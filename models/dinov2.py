"""
DINOv2 ViT-S/14 backbone for robomimic VisualCore.

Outputs patch tokens as a spatial feature map [B, 384, 16, 16],
compatible with SpatialSoftmax pooling.

Subclasses ConvBase so it auto-registers in robomimic's backbone registry.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from robomimic.models.base_nets import ConvBase


class DINOv2Conv(ConvBase):
    """
    DINOv2 ViT-S/14 backbone that outputs spatial feature maps.

    Input images are resized to 224x224 and ImageNet-normalized before
    being fed through the frozen DINOv2 encoder. Patch tokens are
    reshaped into a [B, 384, 16, 16] spatial map so that downstream
    pooling layers (e.g. SpatialSoftmax) work unchanged.

    Args:
        input_channel (int): Number of input channels (set automatically by VisualCore).
        pretrained (bool): Load pretrained DINOv2 weights.
        freeze_until_block (int): Freeze the first N transformer blocks (out of 12).
            Blocks [freeze_until_block:] and the final LayerNorm remain trainable.
    """

    def __init__(self, input_channel=3, pretrained=True, **kwargs):
        super().__init__()

        self.dino = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", pretrained=pretrained,
        )
        self._patch_size = self.dino.patch_size          # 14
        self._target_size = 224                           # 224 / 14 = 16 patches per side
        self._embed_dim = self.dino.embed_dim             # 384
        self._num_patches_side = self._target_size // self._patch_size  # 16

        # Handle register tokens (base model has 0, _reg variants have 4)
        self._num_register_tokens = getattr(self.dino, "num_register_tokens", 0)

        # ImageNet normalization (robomimic sends [0, 1] float images)
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Freeze all DINOv2 parameters — use as a fixed feature extractor
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.eval()

        total_params = sum(p.numel() for p in self.dino.parameters())
        print(f"\n[DINOv2 Backbone] All {total_params:,} params frozen (feature extractor mode)\n")

        # ConvBase expects self.nets; not used but required for compatibility
        self.nets = nn.Identity()

    def output_shape(self, input_shape):
        """Output shape is always [384, 16, 16] regardless of input (we resize to 224)."""
        return [self._embed_dim, self._num_patches_side, self._num_patches_side]

    def train(self, mode=True):
        # Keep DINOv2 always in eval mode (frozen batch norm / dropout)
        super().train(mode)
        self.dino.eval()
        return self

    def forward(self, x):
        B = x.shape[0]

        # Resize to 224x224
        if x.shape[-2:] != (self._target_size, self._target_size):
            x = F.interpolate(
                x, size=(self._target_size, self._target_size),
                mode="bilinear", align_corners=False,
            )

        # ImageNet normalization
        x = (x - self.img_mean) / self.img_std

        # Forward through DINOv2 with no grad — skip activation storage for speed
        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(x)
            for blk in self.dino.blocks:
                x = blk(x)
            x = self.dino.norm(x)

        # Extract patch tokens: skip CLS token (+ any register tokens)
        n_prefix = 1 + self._num_register_tokens
        patch_tokens = x[:, n_prefix:]  # [B, 256, 384]

        # Reshape to spatial feature map [B, 384, 16, 16]
        h = w = self._num_patches_side
        return patch_tokens.transpose(1, 2).reshape(B, self._embed_dim, h, w)
