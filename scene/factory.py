"""
Scene factory: creates robosuite Lift environments with configurable
visual domain randomization (textures, lighting, camera, object placement).

The environments are fully compatible with robomimic's rollout infrastructure
so trained checkpoints can be evaluated directly.
"""
import os
from copy import deepcopy

import numpy as np
import robosuite as suite
import robosuite.macros as macros
from robosuite.utils.mjmod import LightingModder, CameraModder

# Directory containing user-provided texture images (e.g. table.jpg).
_SCENE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(_SCENE_DIR, "assets")

# Training data was collected with opencv convention (images flipped to standard
# top-left origin). Must match here so the policy sees identical orientations.
macros.IMAGE_CONVENTION = "opencv"

from config import (
    SceneConfig, TextureConfig, LightingConfig, CameraConfig, PlacementConfig,
)


# ---------------------------------------------------------------------------
# SceneLightModder: physically plausible lighting using direct model access.
#
# The arena's default light has castshadow=False and directional=True, so
# position changes produce no shadow movement and look like color changes.
# This modder reconfigures the light at runtime to be a shadow-casting
# positional light, then randomizes position, brightness, and warmth.
# ---------------------------------------------------------------------------
class SceneLightModder:
    """
    Reconfigures scene light index 0 to cast shadows and be positional,
    then randomizes:
      - position: moves the light source so shadows visibly shift
      - brightness: uniform scale across all RGB channels (dim <-> bright)
      - warmth: correlated warm/cool shift (+R+G-B for warm, -R-G+B for cool)
    """

    _WARMTH_AXIS   = np.array([1.0, 0.4, -1.0])
    _BASE_POS      = np.array([0.5,  0.5,  2.5])
    _BASE_DIR      = np.array([-0.5, -0.5, -1.0])
    _BASE_DIFFUSE  = np.array([0.8,  0.8,  0.8])
    _BASE_AMBIENT  = np.array([0.15, 0.15, 0.15])
    _BASE_SPECULAR = np.array([0.4,  0.4,  0.4])

    def __init__(self, sim, random_state,
                 position_perturbation_size=0.5,
                 brightness_perturbation_size=0.2,
                 warmth_perturbation_size=0.15,
                 light_idx=0):
        self.sim = sim
        self.random_state = random_state
        self.position_perturbation_size = position_perturbation_size
        self.brightness_perturbation_size = brightness_perturbation_size
        self.warmth_perturbation_size = warmth_perturbation_size
        self.light_idx = light_idx
        self._configure_light()
        self.save_defaults()

    def _configure_light(self):
        """Enable shadows, make the light positional, set a sensible baseline."""
        m = self.sim.model
        i = self.light_idx
        m.light_castshadow[i] = 1
        # Make light positional so shadows shift with its position.
        # light_type (newer MuJoCo): 0=SPOT, 1=DIRECTIONAL, 2=POINT
        # light_directional (older MuJoCo): bool, 0=positional spot
        # light_mode is unrelated (body-tracking mode) — do NOT touch it.
        if hasattr(m, 'light_type'):
            m.light_type[i] = 0       # mjLIGHT_SPOT
        elif hasattr(m, 'light_directional'):
            m.light_directional[i] = 0
        if hasattr(m, 'light_cutoff'):
            m.light_cutoff[i] = 90.0  # wide cone — covers full scene from above
        m.light_dir[i] = [0.0, 0.0, -1.0]  # point straight down
        m.light_pos[i] = self._BASE_POS.copy()
        m.light_diffuse[i] = self._BASE_DIFFUSE.copy()
        m.light_ambient[i] = self._BASE_AMBIENT.copy()
        m.light_specular[i] = self._BASE_SPECULAR.copy()

    def save_defaults(self):
        m = self.sim.model
        i = self.light_idx
        self._def_pos     = m.light_pos[i].copy()
        self._def_diffuse = m.light_diffuse[i].copy()
        self._def_ambient = m.light_ambient[i].copy()

    def update_sim(self, sim):
        self.sim = sim
        self._configure_light()
        self.save_defaults()

    def randomize(self):
        rs = self.random_state
        m  = self.sim.model
        i  = self.light_idx

        # Move the light to a new position so shadows visibly shift
        p = self.position_perturbation_size
        new_pos = self._def_pos + rs.uniform(-p, p, size=3)
        new_pos[2] = np.clip(new_pos[2], 1.5, 3.5)  # keep above scene
        m.light_pos[i] = new_pos

        # Brightness: uniform shift across all RGB channels
        brightness = rs.uniform(-self.brightness_perturbation_size,
                                self.brightness_perturbation_size)
        # Warmth: correlated warm/cool color temperature shift
        warmth = rs.uniform(-self.warmth_perturbation_size,
                            self.warmth_perturbation_size)

        color_delta = brightness + warmth * self._WARMTH_AXIS
        m.light_diffuse[i] = np.clip(self._def_diffuse + color_delta, 0.0, 1.0)
        m.light_ambient[i] = np.clip(self._def_ambient + color_delta * 0.4, 0.0, 1.0)


# ---------------------------------------------------------------------------
# GeomColorModder: mujoco 3.5-compatible color/material randomization
# (robosuite's TextureModder relies on model.tex_rgb which was removed)
# ---------------------------------------------------------------------------
class GeomColorModder:
    """
    Randomize geom RGBA colors and material properties directly via
    MjModel arrays. Works with mujoco >= 3.x.
    """

    def __init__(self, sim, random_state=None, geom_names=None,
                 randomize_local=True, local_rgb_interpolation=0.1,
                 randomize_material=False, local_material_interpolation=0.2):
        self.sim = sim
        self.random_state = random_state or np.random.RandomState()
        self.randomize_local = randomize_local
        self.local_rgb_interpolation = local_rgb_interpolation
        self.randomize_material = randomize_material
        self.local_material_interpolation = local_material_interpolation

        if geom_names is None:
            geom_names = [self.sim.model.geom(i).name for i in range(self.sim.model.ngeom)]
        self.geom_names = [n for n in geom_names if n]  # filter empty names
        self.save_defaults()

    def update_sim(self, sim):
        self.sim = sim

    def save_defaults(self):
        model = self.sim.model
        self._default_geom_rgba = {}
        self._default_mat_props = {}
        for name in self.geom_names:
            gid = model.geom_name2id(name)
            self._default_geom_rgba[name] = model.geom_rgba[gid].copy()
            mid = model.geom_matid[gid]
            if mid >= 0:
                self._default_mat_props[name] = {
                    "rgba": model.mat_rgba[mid].copy(),
                    "shininess": float(model.mat_shininess[mid]),
                    "specular": float(model.mat_specular[mid]),
                    "reflectance": float(model.mat_reflectance[mid]),
                }

    def restore_defaults(self):
        model = self.sim.model
        for name in self.geom_names:
            gid = model.geom_name2id(name)
            model.geom_rgba[gid] = self._default_geom_rgba[name]
            if name in self._default_mat_props:
                mid = model.geom_matid[gid]
                props = self._default_mat_props[name]
                model.mat_rgba[mid] = props["rgba"]
                model.mat_shininess[mid] = props["shininess"]
                model.mat_specular[mid] = props["specular"]
                model.mat_reflectance[mid] = props["reflectance"]

    def randomize(self):
        model = self.sim.model
        for name in self.geom_names:
            gid = model.geom_name2id(name)
            orig = self._default_geom_rgba[name]

            if self.randomize_local:
                delta = self.random_state.uniform(
                    -self.local_rgb_interpolation,
                    self.local_rgb_interpolation,
                    size=3,
                )
                new_rgb = np.clip(orig[:3] + delta, 0.0, 1.0)
            else:
                new_rgb = self.random_state.uniform(0.0, 1.0, size=3)

            model.geom_rgba[gid] = np.append(new_rgb, orig[3])

            if self.randomize_material and name in self._default_mat_props:
                mid = model.geom_matid[gid]
                props = self._default_mat_props[name]
                eps = self.local_material_interpolation
                model.mat_rgba[mid][:3] = np.clip(
                    props["rgba"][:3] + self.random_state.uniform(-eps, eps, 3), 0, 1
                )
                model.mat_shininess[mid] = np.clip(
                    props["shininess"] + self.random_state.uniform(-eps, eps), 0, 1
                )
                model.mat_specular[mid] = np.clip(
                    props["specular"] + self.random_state.uniform(-eps, eps), 0, 1
                )
                model.mat_reflectance[mid] = np.clip(
                    props["reflectance"] + self.random_state.uniform(-eps, eps), 0, 1
                )


# ---------------------------------------------------------------------------
# ImageTextureModder: replaces a geom's existing material texture with a
# user-provided image, resized to the original texture dimensions.
#
# Works with mujoco 3.x (uses model.tex_data, falls back to tex_rgb on
# older versions). Re-uploads the texture to the GPU after writing so the
# next render call picks it up.
# ---------------------------------------------------------------------------
class ImageTextureModder:
    """
    Map of {geom_name: image_path}. On reset, each image is loaded,
    resized to the existing texture's dimensions and written into the
    MuJoCo texture buffer. Idempotent: calling randomize() multiple
    times re-applies the same image.
    """

    def __init__(self, sim, image_paths):
        self.sim = sim
        self.image_paths = dict(image_paths)
        # geom_name -> (tex_id, tex_adr, byte_size, bitmap_uint8)
        self._entries = {}
        self._load_images()

    def _resolve_texture_id(self, model, geom_name):
        gid = model.geom_name2id(geom_name)
        mid = int(model.geom_matid[gid])
        if mid < 0:
            raise ValueError(
                f"Geom {geom_name!r} has no material assigned; cannot set image texture."
            )
        texid = model.mat_texid[mid]
        # mujoco 3.x stores texture ids per role (e.g. RGB, occlusion, ...)
        if hasattr(texid, "__len__"):
            valid = [int(t) for t in texid if int(t) >= 0]
            if not valid:
                raise ValueError(
                    f"Material for geom {geom_name!r} has no texture in any role."
                )
            return valid[0]
        tid = int(texid)
        if tid < 0:
            raise ValueError(
                f"Material for geom {geom_name!r} has no texture assigned."
            )
        return tid

    def _load_images(self):
        import cv2

        model = self.sim.model
        self._entries.clear()
        for geom_name, path in self.image_paths.items():
            tid = self._resolve_texture_id(model, geom_name)
            h = int(model.tex_height[tid])
            w = int(model.tex_width[tid])
            adr = int(model.tex_adr[tid])
            nchannel = (
                int(model.tex_nchannel[tid])
                if hasattr(model, "tex_nchannel") else 3
            )

            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not load texture image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            if nchannel == 4:
                alpha = np.full((h, w, 1), 255, dtype=np.uint8)
                img = np.concatenate([img, alpha], axis=2)
            bitmap = np.ascontiguousarray(img, dtype=np.uint8)
            size = h * w * nchannel
            self._entries[geom_name] = (tid, adr, size, bitmap)

    def update_sim(self, sim):
        self.sim = sim
        self._load_images()

    def save_defaults(self):
        # No restore: image textures are intentionally permanent for the run.
        pass

    def randomize(self):
        model = self.sim.model
        tex_buf = model.tex_data if hasattr(model, "tex_data") else model.tex_rgb
        ctx = getattr(self.sim, "_render_context_offscreen", None)
        for tid, adr, size, bitmap in self._entries.values():
            tex_buf[adr:adr + size] = bitmap.flatten()
            if ctx is not None:
                ctx.upload_texture(tid)


# ---------------------------------------------------------------------------
# Environment kwargs matching the training dataset (Lift / Panda / OSC_POSE)
# ---------------------------------------------------------------------------
_BASE_ENV_KWARGS = dict(
    env_name="Lift",
    robots=["Panda"],
    has_renderer=False,
    has_offscreen_renderer=True,
    ignore_done=True,
    use_object_obs=True,
    use_camera_obs=True,
    control_freq=20,
    camera_depths=False,
    camera_heights=84,
    camera_widths=84,
    lite_physics=False,
    reward_shaping=False,
    camera_names=["agentview", "robot0_eye_in_hand"],
    controller_configs={
        "type": "BASIC",
        "body_parts": {
            "right": {
                "type": "OSC_POSE",
                "input_max": 1,
                "input_min": -1,
                "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                "kp": 150,
                "damping": 1,
                "impedance_mode": "fixed",
                "kp_limits": [0, 300],
                "damping_limits": [0, 10],
                "position_limits": None,
                "orientation_limits": None,
                "uncouple_pos_ori": True,
                "control_delta": True,
                "interpolation": None,
                "ramp_ratio": 0.2,
                "input_ref_frame": "world",
                "gripper": {"type": "GRIP"},
            }
        },
    },
)


class ModifiedLiftEnv:
    """
    Thin wrapper around a robosuite Lift environment that applies
    visual domain randomization via modders after each reset.

    Compatible with robomimic rollout functions (exposes the same
    interface as the raw robosuite env).
    """

    def __init__(self, config: SceneConfig, has_renderer: bool = False):
        self.config = config
        self._rng = np.random.RandomState(config.seed)

        # Build placement initializer kwargs for modified cube position
        placement_kw = {}
        pc = config.placement
        if pc.enabled:
            placement_kw["placement_initializer"] = self._make_placement_initializer(pc)

        env_kwargs = deepcopy(_BASE_ENV_KWARGS)
        if has_renderer:
            env_kwargs["has_renderer"] = True

        # EGL device selection (same logic as robomimic)
        if not has_renderer:
            try:
                import egl_probe
                devices = egl_probe.get_available_devices()
                if devices:
                    env_kwargs["render_gpu_device_id"] = devices[0]
            except ImportError:
                pass

        self.env = suite.make(**env_kwargs, **placement_kw)
        self._modders_initialized = False
        self._modders = []
        self._step_counter = 0

    # ------------------------------------------------------------------
    # Placement initializer
    # ------------------------------------------------------------------
    def _make_placement_initializer(self, pc: PlacementConfig):
        """Build a UniformRandomSampler with the requested placement ranges."""
        from robosuite.utils.placement_samplers import UniformRandomSampler

        return UniformRandomSampler(
            name="SceneCubeSampler",
            x_range=list(pc.x_range),
            y_range=list(pc.y_range),
            rotation=pc.rotation,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=(0.0, 0.0, 0.8),
            z_offset=0.01,
            rng=self._rng,
        )

    # ------------------------------------------------------------------
    # Modder initialization (deferred until first reset, when sim exists)
    # ------------------------------------------------------------------
    def _init_modders(self):
        if self._modders_initialized:
            # Re-associate modders with the (possibly new) sim
            for m in self._modders:
                m.update_sim(self.env.sim)
                m.save_defaults()
            return

        rs = self._rng
        cfg = self.config

        if cfg.lighting.enabled:
            lc = cfg.lighting
            if lc.brightness_perturbation_size > 0 or lc.warmth_perturbation_size > 0:
                # Use SceneLightModder: reconfigures the scene light to cast
                # shadows and be positional, then randomizes position/brightness/warmth.
                self._light_modder = SceneLightModder(
                    sim=self.env.sim,
                    random_state=rs,
                    position_perturbation_size=lc.position_perturbation_size,
                    brightness_perturbation_size=lc.brightness_perturbation_size,
                    warmth_perturbation_size=lc.warmth_perturbation_size,
                )
            else:
                # Fallback: standard robosuite LightingModder (no shadows)
                self._light_modder = LightingModder(
                    sim=self.env.sim,
                    random_state=rs,
                    randomize_position=lc.randomize_position,
                    randomize_direction=lc.randomize_direction,
                    randomize_specular=lc.randomize_specular,
                    randomize_ambient=lc.randomize_ambient,
                    randomize_diffuse=lc.randomize_diffuse,
                    randomize_active=lc.randomize_active,
                    position_perturbation_size=lc.position_perturbation_size,
                    direction_perturbation_size=lc.direction_perturbation_size,
                    specular_perturbation_size=lc.specular_perturbation_size,
                    ambient_perturbation_size=lc.ambient_perturbation_size,
                    diffuse_perturbation_size=lc.diffuse_perturbation_size,
                )
            self._modders.append(self._light_modder)

        if cfg.camera.enabled:
            cc = cfg.camera
            self._camera_modder = CameraModder(
                sim=self.env.sim,
                random_state=rs,
                randomize_position=cc.randomize_position,
                randomize_rotation=cc.randomize_rotation,
                randomize_fovy=cc.randomize_fovy,
                position_perturbation_size=cc.position_perturbation_size,
                rotation_perturbation_size=cc.rotation_perturbation_size,
                fovy_perturbation_size=cc.fovy_perturbation_size,
            )
            self._modders.append(self._camera_modder)

        if cfg.texture.enabled:
            tc = cfg.texture
            self._color_modder = GeomColorModder(
                sim=self.env.sim,
                random_state=rs,
                geom_names=tc.geom_names,
                randomize_local=tc.randomize_local,
                local_rgb_interpolation=tc.local_rgb_interpolation,
                randomize_material=tc.randomize_material,
                local_material_interpolation=tc.local_material_interpolation,
            )
            self._modders.append(self._color_modder)

        if cfg.texture.image_textures:
            self._image_texture_modder = ImageTextureModder(
                sim=self.env.sim,
                image_paths=cfg.texture.image_textures,
            )
            self._modders.append(self._image_texture_modder)

        self._modders_initialized = True

    def _apply_randomization(self):
        """Apply all enabled modders."""
        for modder in self._modders:
            modder.randomize()

    # ------------------------------------------------------------------
    # Position noise (applied after placement initializer samples)
    # ------------------------------------------------------------------
    def _apply_position_noise(self):
        pc = self.config.placement
        if not pc.enabled or pc.position_noise_std <= 0:
            return
        cube_body_id = self.env.sim.model.body_name2id(self.env.cube.root_body)
        noise = self._rng.normal(0, pc.position_noise_std, size=3)
        noise[2] = abs(noise[2])  # keep z positive (don't push through table)
        self.env.sim.data.body_xpos[cube_body_id] += noise

    @staticmethod
    def _remap_obs(obs):
        """Remap robosuite obs keys to match robomimic training conventions."""
        if "object-state" in obs and "object" not in obs:
            obs["object"] = obs["object-state"]
        return obs

    # ------------------------------------------------------------------
    # Public API (mirrors robosuite env interface)
    # ------------------------------------------------------------------
    def reset(self):
        # Workaround for robosuite GL context leak: with hard_reset=True
        # (default) and renderer="mjviewer" (default), robosuite's reset()
        # creates a new sim via _initialize_sim() without first calling
        # _destroy_sim() — that only happens when renderer=="mujoco".
        # Each leaked GL context holds GPU framebuffer resources; after
        # enough resets the GPU cannot allocate new framebuffers and
        # renders return uninitialised memory (static noise).
        if self.env.hard_reset and self.env.sim is not None:
            self.env._destroy_sim()

        obs = self.env.reset()
        self._init_modders()
        self._apply_position_noise()
        self._apply_randomization()
        self.env.sim.forward()
        self._step_counter = 0
        # Re-render observations with modified visual state
        obs = self.env._get_observations()
        return self._remap_obs(obs)

    def step(self, action):
        self._step_counter += 1
        if (self.config.randomize_every_n_steps > 0
                and self._step_counter % self.config.randomize_every_n_steps == 0):
            self._apply_randomization()
        obs, reward, done, info = self.env.step(action)
        return self._remap_obs(obs), reward, done, info

    def render(self):
        """Render the MuJoCo viewer and show camera feeds in an OpenCV window."""
        # Propagate any pending model changes (modder edits to colors,
        # lights, cameras) before reading pixels.
        self.env.sim.forward()

        try:
            import cv2
        except ImportError:
            self.env.render()
            return  # skip camera display if opencv not available

        # Offscreen renders first — before the on-screen viewer call,
        # which can switch the active GL context and corrupt the
        # offscreen framebuffer.
        frames = []
        for cam in ("agentview", "robot0_eye_in_hand"):
            frame = self.env.sim.render(height=256, width=256, camera_name=cam)
            # .copy() immediately: sim.render() can return a view into a
            # reusable internal buffer; the next render call would
            # silently overwrite it.
            frame = frame[::-1].copy()
            frames.append(frame)
        combined = np.concatenate(frames, axis=1)  # 256 x 512

        if combined.dtype != np.uint8:
            scale = 255.0 if combined.max() <= 1.0 else 1.0
            combined = np.clip(combined * scale, 0, 255).astype(np.uint8)

        # On-screen viewer after offscreen reads are done
        self.env.render()

        # OpenCV expects BGR
        cv2.imshow("Camera Views (agentview | eye_in_hand)",
                   cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def close(self):
        try:
            import cv2
            cv2.destroyAllWindows()
        except ImportError:
            pass
        self.env.close()

    # Forward everything else to the inner env
    def __getattr__(self, name):
        return getattr(self.env, name)


# ---------------------------------------------------------------------------
# Convenience: preset configurations
# ---------------------------------------------------------------------------
PRESETS = {
    "default": SceneConfig(
        name="default",
        placement=PlacementConfig(enabled=True),
    ),
    "color_jitter": SceneConfig(
        name="color_jitter",
        texture=TextureConfig(
            enabled=True,
            randomize_local=False,
            local_rgb_interpolation=1.0,
            randomize_material=True,
            texture_variations=("rgb",),
            randomize_skybox=False,
            geom_names=["cube_g0_vis"],
        ),
        placement=PlacementConfig(enabled=True),
    ),
    "color_jitter_table": SceneConfig(
        name="color_jitter_table",
        texture=TextureConfig(
            enabled=True,
            randomize_local=False,   # fully random color
            texture_variations=("rgb",),
            geom_names=["table_visual"],
        ),
    ),
    "color_jitter_table_n_cube": SceneConfig(
        name="color_jitter_table_n_cube",
        texture=TextureConfig(
            enabled=True,
            randomize_local=False,   # fully random color
            local_rgb_interpolation=1.0,
            texture_variations=("rgb",),
            geom_names=["table_visual", "cube_g0_vis"],
        ),
    ),
    "image_texture_table": SceneConfig(
        name="image_texture_table",
        texture=TextureConfig(
            image_textures={
                "table_visual": os.path.join(ASSETS_DIR, "table.jpg"),
            },
        ),
        placement=PlacementConfig(enabled=True),
    ),
    "lighting_change": SceneConfig(
        name="lighting_change",
        lighting=LightingConfig(
            enabled=True,
            position_perturbation_size=1,
            brightness_perturbation_size=1,
            warmth_perturbation_size=1,
        ),
        placement=PlacementConfig(enabled=True),
    ),
    "camera_jitter": SceneConfig(
        name="camera_jitter",
        camera=CameraConfig(
            enabled=True,
            position_perturbation_size=0.02,
            rotation_perturbation_size=0.05,
        ),
        placement=PlacementConfig(enabled=True),
    ),
    "position_noise": SceneConfig(
        name="position_noise",
        placement=PlacementConfig(
            enabled=True,
            x_range=(-0.05, 0.05),
            y_range=(-0.05, 0.05),
            position_noise_std=0.01,
        ),
    ),
    "hard": SceneConfig(
        name="hard",
        texture=TextureConfig(
            enabled=True,
            randomize_local=True,
            local_rgb_interpolation=0.3,
            randomize_material=True,
            texture_variations=("rgb", "noise", "gradient"),
            randomize_skybox=True,
        ),
        lighting=LightingConfig(
            enabled=True,
            position_perturbation_size=0.3,
            direction_perturbation_size=0.5,
            specular_perturbation_size=0.2,
            ambient_perturbation_size=0.15,
            diffuse_perturbation_size=0.15,
        ),
        camera=CameraConfig(
            enabled=False,
            position_perturbation_size=0.02,
            rotation_perturbation_size=0.05,
        ),
        placement=PlacementConfig(
            enabled=True,
            x_range=(-0.05, 0.05),
            y_range=(-0.05, 0.05),
            position_noise_std=0.01,
        ),
    ),
}


def create_env(config: SceneConfig, has_renderer: bool = False) -> ModifiedLiftEnv:
    """Create a Lift environment with the given scene configuration."""
    return ModifiedLiftEnv(config, has_renderer=has_renderer)


def create_env_from_preset(preset_name: str, seed: int = None) -> ModifiedLiftEnv:
    """Create a Lift environment from a named preset."""
    config = deepcopy(PRESETS[preset_name])
    if seed is not None:
        config.seed = seed
    return create_env(config)
