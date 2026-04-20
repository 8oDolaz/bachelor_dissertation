"""
VQ-VAE module for BC-Transformer with DINOv2 backbone.

Implements the VQ-VAE pipeline from Porichis et al., "Imitation Learning from
a Single Demonstration Leveraging Vector Quantization for Robotic Harvesting"
(Robotics 2024, 13, 98), adapted to:

  * Use DINOv2 ViT-S/14 as the image encoder (frozen except for the last
    transformer block + final LayerNorm + a trainable 1x1 projection to the
    codebook embedding dimension).
  * Keep the spatial, block-wise EMA vector quantizer described in the paper.
  * Keep the convolutional image decoder that reconstructs the input from the
    quantized feature map (yielding the L_IR reconstruction loss).
  * Replace the Target Position Decoder (LSTM in the paper) with a BC-
    Transformer policy from robomimic. The transformer consumes the flattened
    quantized embedding just like any other RGB feature.

The joint training objective is

    L = L_policy + beta * L_VQ + lambda * L_IR

where L_policy is the negative log-likelihood from the GMM policy, L_VQ is the
commitment loss, and L_IR is the per-pixel binary cross-entropy reconstruction
loss (Equation 4 in the paper).

Gradient flow follows the paper (Figure 2):
  * Policy and L_IR gradients reach the encoder via the straight-through
    estimator (they skip the argmin of the VQ module).
  * L_VQ gradients only flow into the encoder.
  * Codebook entries are updated via EMA, not gradient descent.
"""
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from robomimic.models.base_nets import ConvBase
from robomimic.models.obs_core import VisualCore
from robomimic.algo.bc import BC_Transformer_GMM, PolicyAlgo


# ---------------------------------------------------------------------------
# DINOv2 encoder with only the last block + norm trainable + projection head
# ---------------------------------------------------------------------------
class DINOv2ConvLastTrainable(ConvBase):
    """
    DINOv2 ViT-S/14 backbone that emits a spatial feature map [B, D, 16, 16]
    suitable for the spatial vector quantizer.

    Trainable components:
      * The last transformer block of DINOv2 (blocks[-1])
      * The final LayerNorm (`dino.norm`)
      * A 1x1 conv projection from the DINOv2 embed dim (384) to `out_dim`
        (the codebook embedding dim, i.e. the block length B in the paper).

    All other DINOv2 parameters (patch embed, earlier blocks, pos embed, etc.)
    are frozen.

    The output `[B, out_dim, 16, 16]` is what the paper refers to as e_t: 16*16
    spatial locations, each of which is a `out_dim`-dim block that the
    quantizer will match against a codebook entry.
    """

    def __init__(self, input_channel=3, pretrained=True, out_dim=64, **kwargs):
        super().__init__()

        self.dino = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", pretrained=pretrained,
        )
        self._patch_size = self.dino.patch_size              # 14
        self._target_size = 224                               # 224 / 14 = 16
        self._embed_dim = self.dino.embed_dim                 # 384
        self._num_patches_side = self._target_size // self._patch_size  # 16
        self._num_register_tokens = getattr(self.dino, "num_register_tokens", 0)
        self._out_dim = out_dim

        # ImageNet normalization (robomimic provides float images in [0, 1])
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Freeze everything ...
        for param in self.dino.parameters():
            param.requires_grad = False

        # ... then unfreeze the last transformer block + final LayerNorm.
        last_block = self.dino.blocks[-1]
        for param in last_block.parameters():
            param.requires_grad = True
        for param in self.dino.norm.parameters():
            param.requires_grad = True

        # Trainable 1x1 projection: 384 -> out_dim, acting on the spatial map.
        # This plays the role of the "last layer" that adapts DINOv2 features
        # to the codebook block length.
        self.proj = nn.Conv2d(self._embed_dim, out_dim, kernel_size=1)

        total = sum(p.numel() for p in self.dino.parameters())
        trainable = sum(p.numel() for p in self.dino.parameters() if p.requires_grad)
        proj_trainable = sum(p.numel() for p in self.proj.parameters())
        print(f"\n[DINOv2-VQVAE] DINOv2 params: {total:,} "
              f"(trainable in last block + norm: {trainable:,}) | "
              f"proj params: {proj_trainable:,}\n")

        # ConvBase expects self.nets; unused but required for interface parity.
        self.nets = nn.Identity()

    def output_shape(self, input_shape):
        return [self._out_dim, self._num_patches_side, self._num_patches_side]

    def train(self, mode=True):
        # Full module follows `mode`; this affects BatchNorm / Dropout in proj
        # and in the trainable last block. Earlier blocks have no BN/Dropout
        # that matters, so this is fine.
        super().train(mode)
        return self

    def forward(self, x):
        B = x.shape[0]

        # Resize to 224x224 so we get exactly 16x16 patch tokens.
        if x.shape[-2:] != (self._target_size, self._target_size):
            x = F.interpolate(
                x, size=(self._target_size, self._target_size),
                mode="bilinear", align_corners=False,
            )
        x_norm = (x - self.img_mean) / self.img_std

        # Run frozen prefix (patch embed + all but last block) without tracking
        # gradients to save memory. Then run the trainable last block + norm
        # with gradients enabled.
        with torch.no_grad():
            tokens = self.dino.prepare_tokens_with_masks(x_norm)
            for blk in self.dino.blocks[:-1]:
                tokens = blk(tokens)

        tokens = self.dino.blocks[-1](tokens)
        tokens = self.dino.norm(tokens)

        # Drop CLS + register tokens, keep only patch tokens.
        n_prefix = 1 + self._num_register_tokens
        patch_tokens = tokens[:, n_prefix:]                         # [B, 256, 384]

        h = w = self._num_patches_side
        feat = patch_tokens.transpose(1, 2).reshape(B, self._embed_dim, h, w)
        feat = self.proj(feat)                                      # [B, D, 16, 16]
        return feat


# ---------------------------------------------------------------------------
# Spatial vector quantizer (paper Eq. 1-2, EMA codebook)
# ---------------------------------------------------------------------------
class SpatialVectorQuantizer(nn.Module):
    """
    Spatial vector quantizer with EMA codebook updates.

    Expects a feature map `e_t` of shape [B, D, H, W] where D equals the
    codebook embedding dimension (the block length B in the paper). Each of
    the H*W spatial positions is an independent block that gets matched to
    its nearest codebook entry.

    Returns the quantized feature map `z_t` of the same shape, with gradients
    straight-through copied from the encoder output (paper Figure 2, dashed
    lines).

    Codebook update (paper Eq. 1):
        b_i_tau = gamma * b_i_{tau-1} + (1 - gamma) * r_i

    Commitment loss (paper Eq. 2):
        L_VQ = || e_t - sg[z_t] ||^2

    Args:
        num_embeddings: codebook vocabulary size N.
        embedding_dim:  block length B (dim of each codebook entry).
        commitment_cost: weight beta on the commitment loss.
        decay: EMA coefficient gamma (0.9 ... 1).
        epsilon: Laplace-smoothing epsilon to avoid dead codes.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25,
                 decay=0.99, epsilon=1e-5):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings, 1.0 / num_embeddings
        )

        # Codebook is updated via EMA, not gradient descent.
        self.embedding.weight.requires_grad = False

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', self.embedding.weight.data.clone())

    def forward(self, inputs):
        """
        Args:
            inputs: [B, D, H, W] continuous spatial embedding from the encoder.

        Returns:
            quantized: [B, D, H, W] quantized spatial embedding (straight-through).
            vq_loss:   scalar commitment loss (Eq. 2) already scaled by beta.
            indices:   [B, H, W] LongTensor of selected codebook entries.
            perplexity: scalar codebook-utilization metric.
        """
        assert inputs.dim() == 4 and inputs.shape[1] == self.embedding_dim, (
            f"SpatialVectorQuantizer expects [B, D={self.embedding_dim}, H, W], "
            f"got {tuple(inputs.shape)}"
        )

        B, D, H, W = inputs.shape
        # [B, D, H, W] -> [B, H, W, D] -> [B*H*W, D]
        flat = inputs.permute(0, 2, 3, 1).contiguous().reshape(-1, D)

        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z . e
        distances = (
            torch.sum(flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2.0 * torch.matmul(flat, self.embedding.weight.t())
        )

        indices = torch.argmin(distances, dim=1)                # [B*H*W]
        encodings = F.one_hot(indices, self.num_embeddings).float()

        quantized_flat = self.embedding(indices)                # [B*H*W, D]
        quantized = quantized_flat.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        # EMA codebook update (paper Eq. 1). Only in training mode.
        if self.training:
            with torch.no_grad():
                self._ema_cluster_size.mul_(self.decay).add_(
                    torch.sum(encodings, dim=0), alpha=1.0 - self.decay
                )

                n = torch.sum(self._ema_cluster_size)
                cluster_size = (
                    (self._ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon) * n
                )

                dw = torch.matmul(encodings.t(), flat)
                self._ema_w.mul_(self.decay).add_(dw, alpha=1.0 - self.decay)

                self.embedding.weight.data.copy_(
                    self._ema_w / cluster_size.unsqueeze(1)
                )

        # Commitment loss (Eq. 2): push encoder output towards chosen codebook entry.
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        vq_loss = self.commitment_cost * commitment_loss

        # Straight-through estimator so policy/reconstruction gradients reach encoder.
        quantized = inputs + (quantized - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, vq_loss, indices.reshape(B, H, W), perplexity


# ---------------------------------------------------------------------------
# Convolutional image decoder (paper: "a series of deconvolution layers")
# ---------------------------------------------------------------------------
class ImageDecoderDeconv(nn.Module):
    """
    Reconstructs the input RGB image from the quantized spatial embedding
    z_t ∈ R^{D x H x W}.

    Performs 4 upsampling stages (16x16 -> 32 -> 64 -> 128 -> 224) via
    ConvTranspose2d / bilinear upsampling, ending with a sigmoid so the output
    is in [0, 1] and can be compared to the normalized input image with BCE
    (paper Eq. 4).
    """

    def __init__(self, embed_dim=64, out_channels=3, out_size=224,
                 hidden_channels=(128, 64, 32, 16)):
        super().__init__()

        self.out_size = out_size
        c1, c2, c3, c4 = hidden_channels

        # 16 -> 32 -> 64 -> 128 -> 256 (crop/resize to `out_size` at the end).
        self.net = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, c1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(c3, c4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),

            nn.Conv2d(c4, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, z):
        out = self.net(z)                              # [B, 3, 256, 256]
        if out.shape[-1] != self.out_size:
            out = F.interpolate(
                out, size=(self.out_size, self.out_size),
                mode="bilinear", align_corners=False,
            )
        return torch.sigmoid(out)                      # in [0, 1]


# ---------------------------------------------------------------------------
# Visual core wrapping encoder + VQ + decoder, with global pooling for policy
# ---------------------------------------------------------------------------
class VisualCoreVQVAE(VisualCore):
    """
    RGB encoder core that integrates the VQ-VAE pipeline.

        Image  -->  DINOv2 (last-block-trainable) + 1x1 proj  -->  e_t
                -->  SpatialVectorQuantizer (EMA)               -->  z_t
                -->  ImageDecoderDeconv                         -->  I_tilde
                -->  Flatten/pool(z_t)                          -->  policy input

    The VQ commitment loss and image reconstruction loss are stored as module
    attributes after every forward pass so that BC_Transformer_GMM_VQVAE can
    pull them into the training objective.

    The module is auto-registered under the name "VisualCoreVQVAE" via
    EncoderCore's __init_subclass__ hook, so it can be selected in the config
    with:
        config.observation.encoder.rgb.core_class = "VisualCoreVQVAE"

    Notable kwargs (passed through core_kwargs):
        feature_dimension:   final feature dim fed to the policy. If None, the
                             flattened z_t is used directly.
        vq_num_embeddings:   codebook size N.
        vq_embedding_dim:    block length B (== encoder channel dim).
        vq_commitment_cost:  commitment weight beta inside the VQ module.
        vq_decay:            EMA coefficient gamma.
        decoder_out_size:    spatial size of the reconstructed image.
    """

    def __init__(
        self,
        input_shape,
        backbone_class="DINOv2ConvLastTrainable",
        pool_class=None,
        backbone_kwargs=None,
        pool_kwargs=None,
        flatten=True,
        feature_dimension=128,
        vq_num_embeddings=1024,
        vq_embedding_dim=64,
        vq_commitment_cost=0.25,
        vq_decay=0.99,
        decoder_out_size=224,
    ):
        # Build backbone ourselves to control out_dim. Inject it into kwargs.
        backbone_kwargs = dict(backbone_kwargs or {})
        backbone_kwargs.setdefault("out_dim", vq_embedding_dim)

        # We bypass VisualCore's pooling pipeline because we need the 2D
        # spatial map for quantization + reconstruction. Force pool_class=None
        # and flatten=False, and build the feature MLP manually below.
        super().__init__(
            input_shape=input_shape,
            backbone_class=backbone_class,
            pool_class=None,
            backbone_kwargs=backbone_kwargs,
            pool_kwargs=None,
            flatten=False,
            feature_dimension=None,
        )

        # At this point self.nets contains: [backbone]
        # (VisualCore appends pool/flatten/fc only when requested).
        self._vq = SpatialVectorQuantizer(
            num_embeddings=vq_num_embeddings,
            embedding_dim=vq_embedding_dim,
            commitment_cost=vq_commitment_cost,
            decay=vq_decay,
        )

        self._decoder = ImageDecoderDeconv(
            embed_dim=vq_embedding_dim,
            out_channels=input_shape[0],
            out_size=decoder_out_size,
        )

        # Feature map shape after backbone.
        encoder_out = self.nets[0].output_shape(list(input_shape))  # [D, H, W]
        self._encoder_out_shape = encoder_out
        flat_dim = int(np.prod(encoder_out))

        # Projection head producing the per-timestep observation embedding that
        # the transformer consumes.
        self._feature_dimension = feature_dimension
        if feature_dimension is not None:
            self._fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_dim, feature_dimension),
            )
        else:
            self._fc = nn.Flatten()

        # Scratch space populated during forward() and read by the algo.
        self._vq_loss = None
        self._vq_perplexity = None
        self._recon_loss = None

        self._decoder_out_size = decoder_out_size

    def output_shape(self, input_shape=None):
        if self._feature_dimension is not None:
            return [self._feature_dimension]
        return [int(np.prod(self._encoder_out_shape))]

    def forward(self, inputs):
        """
        inputs: [B, 3, H, W] RGB image in [0, 1] (possibly after augmentation).
        """
        ndim = len(self.input_shape)
        assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape)

        # Keep a copy of the encoder's *actual* input to compare against the
        # reconstruction. This lives in [0, 1] (robomimic feeds normalized pixels).
        target = inputs

        # Encoder: [B, 3, H, W] -> [B, D, 16, 16]
        e_t = self.backbone(inputs)

        # Quantize each spatial block.
        z_t, vq_loss, _, perplexity = self._vq(e_t)

        # Decode to reconstruction, then resize the *target* to match the
        # decoder output for BCE.
        recon = self._decoder(z_t)

        if target.shape[-2:] != recon.shape[-2:]:
            target_resized = F.interpolate(
                target, size=recon.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        else:
            target_resized = target

        # Pixel-wise BCE (paper Eq. 4). Inputs are clamped to avoid log(0)
        # when random crops push values to exactly 0 or 1.
        target_clamped = target_resized.clamp(1e-6, 1.0 - 1e-6)
        recon_clamped = recon.clamp(1e-6, 1.0 - 1e-6)
        recon_loss = F.binary_cross_entropy(recon_clamped, target_clamped)

        # Stash losses for the algo.
        self._vq_loss = vq_loss
        self._vq_perplexity = perplexity
        self._recon_loss = recon_loss

        # Policy-side embedding (flattened / projected quantized map).
        return self._fc(z_t)


# ---------------------------------------------------------------------------
# BC-Transformer-GMM + VQ-VAE joint training
# ---------------------------------------------------------------------------
class BC_Transformer_GMM_VQVAE(BC_Transformer_GMM):
    """
    BC-Transformer with GMM policy head + parallel VQ-VAE training.

    Identical to BC_Transformer_GMM but collects the VQ commitment loss and
    the image reconstruction loss from every VisualCoreVQVAE module in the
    network, and adds them to the action loss with weights `beta` and
    `lambda` (configurable via algo_config.vqvae).

    Joint loss (paper Eq. 5, with LSTM target-position decoder replaced by the
    GMM transformer policy):

        L = L_policy + beta * L_VQ + lambda * L_IR
    """

    def _forward_training(self, batch, epoch=None):
        predictions = super()._forward_training(batch, epoch=epoch)

        vq_loss = torch.tensor(0.0, device=self.device)
        recon_loss = torch.tensor(0.0, device=self.device)
        vq_perplexity = torch.tensor(0.0, device=self.device)
        count = 0
        for module in self.nets.modules():
            if isinstance(module, VisualCoreVQVAE) and module._vq_loss is not None:
                vq_loss = vq_loss + module._vq_loss
                recon_loss = recon_loss + module._recon_loss
                vq_perplexity = vq_perplexity + module._vq_perplexity
                count += 1

        if count > 0:
            predictions["vq_loss"] = vq_loss / count
            predictions["recon_loss"] = recon_loss / count
            predictions["vq_perplexity"] = vq_perplexity / count

        return predictions

    def _compute_losses(self, predictions, batch):
        losses = super()._compute_losses(predictions, batch)

        # Loss weights (fall back to sensible defaults if not configured).
        vqvae_cfg = getattr(self.algo_config, "vqvae", None)
        beta = getattr(vqvae_cfg, "beta", 1.0) if vqvae_cfg is not None else 1.0
        lam = getattr(vqvae_cfg, "recon_weight", 1.0) if vqvae_cfg is not None else 1.0

        if "vq_loss" in predictions:
            losses["vq_loss"] = predictions["vq_loss"]
            losses["recon_loss"] = predictions["recon_loss"]
            losses["action_loss"] = (
                losses["action_loss"]
                + beta * predictions["vq_loss"]
                + lam * predictions["recon_loss"]
            )
        return losses

    def log_info(self, info):
        log = super().log_info(info)
        if "vq_loss" in info["losses"]:
            log["VQ_Loss"] = info["losses"]["vq_loss"].item()
        if "recon_loss" in info["losses"]:
            log["Recon_Loss"] = info["losses"]["recon_loss"].item()
        if "vq_perplexity" in info.get("predictions", {}):
            log["VQ_Perplexity"] = info["predictions"]["vq_perplexity"].item()
        return log
