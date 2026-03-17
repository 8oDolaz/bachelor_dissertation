"""
Vector Quantization module for BC-RNN with image observations.

Implements VQ-VAE style vector quantization (van den Oord et al., 2017)
that sits between the visual encoder and the RNN policy.

Pipeline:
  Image -> ResNet18 (frozen) -> SpatialSoftmax -> Linear(64) -> VQ(codebook) -> LSTM -> GMM

The VQ layer maps each continuous 64-dim visual feature to the nearest
entry in a learned codebook, creating a discrete bottleneck.
"""
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from robomimic.models.obs_core import VisualCore
from robomimic.algo.bc import BC_RNN_GMM, PolicyAlgo


class VectorQuantizer(nn.Module):
    """
    Vector Quantization with Exponential Moving Average (EMA) codebook updates.

    Given continuous encoder outputs z_e, finds the nearest codebook entry e_k
    and returns it. Gradients pass through via the straight-through estimator.

    Training losses:
      - Commitment loss: beta * ||z_e - sg[e_k]||^2
        (encourages encoder outputs to stay close to codebook entries)
      - Codebook is updated via EMA (no gradient needed for codebook loss)

    Args:
        num_embeddings: size of the codebook (K)
        embedding_dim: dimension of each codebook entry (D)
        commitment_cost: weight for commitment loss (beta)
        decay: EMA decay rate for codebook updates
        epsilon: small constant for numerical stability in EMA
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25,
                 decay=0.99, epsilon=1e-5):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Codebook: K entries of dimension D
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings, 1.0 / num_embeddings
        )

        # EMA tracking buffers (not model parameters, updated manually)
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', self.embedding.weight.data.clone())

    def forward(self, inputs):
        """
        Args:
            inputs: [..., D] tensor of continuous features.
                    Typically [B, D] or [B*T, D].

        Returns:
            quantized: same shape as inputs, with features replaced by
                       nearest codebook entries
            vq_loss: scalar commitment loss
            encoding_indices: [...] indices of selected codebook entries
            perplexity: scalar measuring codebook utilization (higher = more
                        entries used)
        """
        input_shape = inputs.shape
        # Flatten to [N, D]
        flat_input = inputs.reshape(-1, self.embedding_dim)

        # Compute pairwise distances: ||z - e||^2
        #   = ||z||^2 + ||e||^2 - 2 * z . e
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2.0 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        # Nearest codebook entry for each input
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        # Look up quantized vectors
        quantized = self.embedding(encoding_indices).reshape(input_shape)

        # EMA codebook update (training only)
        if self.training:
            self._ema_cluster_size.mul_(self.decay).add_(
                torch.sum(encodings, dim=0), alpha=1.0 - self.decay
            )

            # Laplace smoothing to avoid dead codes
            n = torch.sum(self._ema_cluster_size)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w.mul_(self.decay).add_(dw, alpha=1.0 - self.decay)

            self.embedding.weight.data = (
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Commitment loss: push encoder outputs towards codebook entries
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        vq_loss = self.commitment_cost * commitment_loss

        # Straight-through estimator: copy gradients from quantized to inputs
        quantized = inputs + (quantized - inputs).detach()

        # Perplexity: effective codebook usage (exp of entropy)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        return quantized, vq_loss, encoding_indices.reshape(input_shape[:-1]), perplexity


class VisualCoreVQ(VisualCore):
    """
    VisualCore with Vector Quantization bottleneck.

    Inherits the full ResNet18 + SpatialSoftmax + Linear pipeline from
    VisualCore, and appends a VQ layer that discretizes the output features
    using a learned codebook.

    After forward(), the VQ loss and perplexity are stored as attributes
    (_vq_loss, _vq_perplexity) for retrieval during training.

    This class is auto-registered as an encoder core via EncoderCore's
    __init_subclass__ mechanism, so it can be used in config with:
        config.observation.encoder.rgb.core_class = "VisualCoreVQ"

    Additional kwargs (passed via core_kwargs in config):
        vq_num_embeddings: codebook size (default: 512)
        vq_commitment_cost: commitment loss weight (default: 0.25)
        vq_decay: EMA decay rate (default: 0.99)
    """

    def __init__(
        self,
        input_shape,
        backbone_class="ResNet18Conv",
        pool_class="SpatialSoftmax",
        backbone_kwargs=None,
        pool_kwargs=None,
        flatten=True,
        feature_dimension=64,
        vq_num_embeddings=512,
        vq_commitment_cost=0.25,
        vq_decay=0.99,
    ):
        super().__init__(
            input_shape=input_shape,
            backbone_class=backbone_class,
            pool_class=pool_class,
            backbone_kwargs=backbone_kwargs,
            pool_kwargs=pool_kwargs,
            flatten=flatten,
            feature_dimension=feature_dimension,
        )

        # Determine VQ dimension from encoder output
        vq_dim = (
            feature_dimension
            if feature_dimension is not None
            else int(np.prod(self.output_shape(input_shape)))
        )

        self.vq = VectorQuantizer(
            num_embeddings=vq_num_embeddings,
            embedding_dim=vq_dim,
            commitment_cost=vq_commitment_cost,
            decay=vq_decay,
        )

        # Stored after each forward pass for loss computation
        self._vq_loss = None
        self._vq_perplexity = None

    def forward(self, inputs):
        """Forward: VisualCore encoding followed by vector quantization."""
        features = super().forward(inputs)
        quantized, vq_loss, _, perplexity = self.vq(features)

        self._vq_loss = vq_loss
        self._vq_perplexity = perplexity

        return quantized


class BC_RNN_GMM_VQ(BC_RNN_GMM):
    """
    BC-RNN-GMM with Vector Quantization.

    Identical to BC_RNN_GMM but collects VQ losses from VisualCoreVQ
    encoder modules and adds them to the total training loss.
    """

    def _forward_training(self, batch):
        """Run forward pass and collect VQ losses from all VisualCoreVQ modules."""
        predictions = super()._forward_training(batch)

        # Collect VQ losses from all VisualCoreVQ encoder modules
        vq_loss = torch.tensor(0.0, device=self.device)
        vq_perplexity = torch.tensor(0.0, device=self.device)
        count = 0
        for module in self.nets.modules():
            if isinstance(module, VisualCoreVQ) and module._vq_loss is not None:
                vq_loss = vq_loss + module._vq_loss
                vq_perplexity = vq_perplexity + module._vq_perplexity
                count += 1

        if count > 0:
            predictions["vq_loss"] = vq_loss / count
            predictions["vq_perplexity"] = vq_perplexity / count

        return predictions

    def _compute_losses(self, predictions, batch):
        """Add VQ commitment loss to the action loss."""
        losses = super()._compute_losses(predictions, batch)

        if "vq_loss" in predictions:
            losses["vq_loss"] = predictions["vq_loss"]
            losses["action_loss"] = losses["action_loss"] + predictions["vq_loss"]

        return losses

    def log_info(self, info):
        """Log VQ-specific metrics to tensorboard."""
        log = super().log_info(info)

        if "vq_loss" in info["losses"]:
            log["VQ_Loss"] = info["losses"]["vq_loss"].item()
        if "vq_perplexity" in info.get("predictions", {}):
            log["VQ_Perplexity"] = info["predictions"]["vq_perplexity"].item()

        return log
