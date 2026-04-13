"""
Scene factory: creates robosuite Lift environments with configurable
visual domain randomization (textures, lighting, camera, object placement).

The environments are fully compatible with robomimic's rollout infrastructure
so trained checkpoints can be evaluated directly.
"""
from copy import deepcopy

import numpy as np
import robosuite as suite
import robosuite.macros as macros
from robosuite.utils.mjmod import LightingModder, CameraModder

# Training data was collected with opencv convention (images flipped to standard
# top-left origin). Must match here so the policy sees identical orientations.
macros.IMAGE_CONVENTION = "opencv"

from config import (
    SceneConfig, TextureConfig, LightingConfig, CameraConfig, PlacementConfig,
)


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
    "lighting_change": SceneConfig(
        name="lighting_change",
        lighting=LightingConfig(
            enabled=True,
            position_perturbation_size=0.3,
            direction_perturbation_size=0.5,
            specular_perturbation_size=0.2,
            ambient_perturbation_size=0.15,
            diffuse_perturbation_size=0.15,
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
            enabled=True,
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
