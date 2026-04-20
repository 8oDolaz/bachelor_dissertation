"""
Configuration dataclasses for scene visual variations.

Each config controls one axis of domain randomization:
  - TextureConfig: object/table/floor colors and texture patterns
  - LightingConfig: light position, direction, color intensity
  - CameraConfig: camera position, rotation, field-of-view jitter
  - PlacementConfig: object placement noise (position displacement)
  - SceneConfig: combines all of the above
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class TextureConfig:
    """Controls color and texture randomization."""
    enabled: bool = False
    # If True, vary colors near originals; if False, fully random
    randomize_local: bool = True
    # How far from original color to drift (0=identical, 1=fully random)
    local_rgb_interpolation: float = 0.1
    # Randomize material reflectance/shininess
    randomize_material: bool = False
    local_material_interpolation: float = 0.2
    # Which texture patterns to sample from
    texture_variations: Tuple[str, ...] = ("rgb",)
    # Randomize the skybox/background
    randomize_skybox: bool = False
    # Restrict to specific geoms (None = all)
    geom_names: Optional[List[str]] = None


@dataclass
class LightingConfig:
    """Controls lighting randomization."""
    enabled: bool = False
    randomize_position: bool = True
    randomize_direction: bool = True
    randomize_specular: bool = True
    randomize_ambient: bool = True
    randomize_diffuse: bool = True
    randomize_active: bool = False
    position_perturbation_size: float = 0.1
    direction_perturbation_size: float = 0.35
    specular_perturbation_size: float = 0.1
    ambient_perturbation_size: float = 0.1
    diffuse_perturbation_size: float = 0.1


@dataclass
class CameraConfig:
    """Controls camera viewpoint randomization."""
    enabled: bool = False
    randomize_position: bool = True
    randomize_rotation: bool = True
    randomize_fovy: bool = False
    position_perturbation_size: float = 0.01
    rotation_perturbation_size: float = 0.087  # ~5 degrees
    fovy_perturbation_size: float = 5.0


@dataclass
class PlacementConfig:
    """Controls object position displacement."""
    enabled: bool = True
    # Cube placement ranges (default training range is [-0.03, 0.03])
    x_range: Tuple[float, float] = (-0.03, 0.03)
    y_range: Tuple[float, float] = (-0.03, 0.03)
    # Additional Gaussian noise on top of uniform sampling
    position_noise_std: float = 0.0
    # Rotation randomization (None = uniform random, same as training)
    rotation: Optional[float] = None


@dataclass
class SceneConfig:
    """Complete scene configuration combining all variation axes."""
    name: str = "default"
    seed: Optional[int] = None
    texture: TextureConfig = field(default_factory=TextureConfig)
    lighting: LightingConfig = field(default_factory=LightingConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    placement: PlacementConfig = field(default_factory=PlacementConfig)
    # How often to re-randomize (1 = every step, 0 = only on reset)
    randomize_every_n_steps: int = 0
