"""
DINOv2 ViT-B/14 backbone for robomimic VisualCore.

Outputs patch tokens as a spatial feature map [B, 768, 16, 16],
compatible with SpatialSoftmax pooling.

Subclasses ConvBase so it auto-registers in robomimic's backbone registry.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from robomimic.models.base_nets import ConvBase


class DINOv2Conv(ConvBase):
    """
    DINOv2 ViT-B/14 backbone that outputs spatial feature maps.

    Input images are resized to 224x224 and ImageNet-normalized before
    being fed through the frozen DINOv2 encoder. Patch tokens are
    reshaped into a [B, 768, 16, 16] spatial map so that downstream
    pooling layers (e.g. SpatialSoftmax) work unchanged.

    Args:
        input_channel (int): Number of input channels (set automatically by VisualCore).
        pretrained (bool): Load pretrained DINOv2 weights.
        freeze_until_block (int): Freeze the first N transformer blocks (out of 12).
            Blocks [freeze_until_block:] and the final LayerNorm remain trainable.
    """

    def __init__(self, input_channel=3, pretrained=True, freeze_until_block=10, **kwargs):
        super().__init__()

        self.dino = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", pretrained=pretrained,
        )
        self._patch_size = self.dino.patch_size          # 14
        self._target_size = 224                           # 224 / 14 = 16 patches per side
        self._embed_dim = self.dino.embed_dim             # 768
        self._num_patches_side = self._target_size // self._patch_size  # 16

        # Handle register tokens (base model has 0, _reg variants have 4)
        self._num_register_tokens = getattr(self.dino, "num_register_tokens", 0)

        # ImageNet normalization (robomimic sends [0, 1] float images)
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # --- Freeze strategy ---
        # Freeze everything first
        for param in self.dino.parameters():
            param.requires_grad = False
        # Unfreeze last (12 - freeze_until_block) transformer blocks + final norm
        frozen_count = sum(p.numel() for p in self.dino.parameters() if not p.requires_grad)
        for block in self.dino.blocks[freeze_until_block:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.dino.norm.parameters():
            param.requires_grad = True
        trainable_count = sum(p.numel() for p in self.dino.parameters() if p.requires_grad)
        frozen_count = sum(p.numel() for p in self.dino.parameters() if not p.requires_grad)
        print(
            f"\n[DINOv2 Backbone] Frozen params: {frozen_count:,} | "
            f"Trainable params (blocks {freeze_until_block}-11 + norm): {trainable_count:,}\n"
        )

        # ConvBase expects self.nets; not used but required for compatibility
        self.nets = nn.Identity()

    def output_shape(self, input_shape):
        """Output shape is always [768, 16, 16] regardless of input (we resize to 224)."""
        return [self._embed_dim, self._num_patches_side, self._num_patches_side]

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

        # Forward through DINOv2 (manually to avoid API differences across versions)
        x = self.dino.prepare_tokens_with_masks(x)
        for blk in self.dino.blocks:
            x = blk(x)
        x = self.dino.norm(x)

        # Extract patch tokens: skip CLS token (+ any register tokens)
        n_prefix = 1 + self._num_register_tokens
        patch_tokens = x[:, n_prefix:]  # [B, 256, 768]

        # Reshape to spatial feature map [B, 768, 16, 16]
        h = w = self._num_patches_side
        return patch_tokens.transpose(1, 2).reshape(B, self._embed_dim, h, w)
