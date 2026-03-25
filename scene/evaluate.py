"""
Evaluate a trained checkpoint on modified Lift scenes.

Loads a robomimic checkpoint, creates the standard environment from it,
then applies visual domain randomization via scene configs. Reports
success rate per scene variant and saves rollout videos.

Usage:
    python scene/evaluate.py --agent path/to/model.pth --preset default
    python scene/evaluate.py --agent path/to/model.pth --preset hard --n_rollouts 50
    python scene/evaluate.py --agent path/to/model.pth --all_presets
    python scene/evaluate.py --agent path/to/model.pth --preset default --video_path scene/videos/default.mp4
"""
import argparse
import os
import sys
import traceback
from copy import deepcopy

import numpy as np
import torch
import imageio

# Force deterministic CUDA operations so GMM action sampling is reproducible.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Add project root and scene dir to path
_SCENE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCENE_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCENE_DIR)

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.algo import RolloutPolicy

from config import SceneConfig
from factory import PRESETS, create_env

# Import VQ components so they are registered
from models.vq import VisualCoreVQ, BC_RNN_GMM_VQ


def _load_policy(ckpt_path, device):
    """
    Load policy from checkpoint with VQ model support.

    Standard robomimic's algo_factory doesn't know about BC_RNN_GMM_VQ,
    so we patch it the same way the training script does.
    """
    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
    algo_name, _ = FileUtils.algo_name_from_checkpoint(ckpt_dict=ckpt_dict)
    config, _ = FileUtils.config_from_checkpoint(algo_name=algo_name, ckpt_dict=ckpt_dict)
    ObsUtils.initialize_obs_utils_with_config(config)

    shape_meta = ckpt_dict["shape_metadata"]

    # Convert normalization stats from lists to np.arrays
    # (checkpoint serialization stores them as lists)
    obs_normalization_stats = ckpt_dict.get("obs_normalization_stats", None)
    if obs_normalization_stats is not None:
        for m in obs_normalization_stats:
            for k in obs_normalization_stats[m]:
                obs_normalization_stats[m][k] = np.array(obs_normalization_stats[m][k])

    action_normalization_stats = ckpt_dict.get("action_normalization_stats", None)
    if action_normalization_stats is not None:
        for m in action_normalization_stats:
            for k in action_normalization_stats[m]:
                action_normalization_stats[m][k] = np.array(action_normalization_stats[m][k])

    # Determine if this is a VQ model by checking encoder config
    is_vq = False
    try:
        core_class = config.observation.encoder.rgb.core_class
        is_vq = (core_class == "VisualCoreVQ")
    except AttributeError:
        pass

    if is_vq:
        model = BC_RNN_GMM_VQ(
            algo_config=config.algo,
            obs_config=config.observation,
            global_config=config,
            obs_key_shapes=shape_meta["all_shapes"],
            ac_dim=shape_meta["ac_dim"],
            device=device,
        )
    else:
        from robomimic.algo import algo_factory
        model = algo_factory(
            algo_name,
            config,
            obs_key_shapes=shape_meta["all_shapes"],
            ac_dim=shape_meta["ac_dim"],
            device=device,
        )

    model.deserialize(ckpt_dict["model"])
    model.set_eval()
    policy = RolloutPolicy(
        model,
        obs_normalization_stats=obs_normalization_stats,
        action_normalization_stats=action_normalization_stats,
    )
    return policy, ckpt_dict


def _render_frame(env, camera_names=("agentview", "robot0_eye_in_hand"), height=256, width=256):
    """Render frames from all cameras and concatenate horizontally."""
    # Propagate model state (geom colors, light/camera positions set by
    # modders) into the rendering pipeline before reading pixels.
    env.env.sim.forward()

    frames = []
    for cam in camera_names:
        frame = env.env.sim.render(height=height, width=width, camera_name=cam)
        # .copy() is critical: sim.render() may return a view into an
        # internal pixel buffer that gets overwritten by the next render
        # call.  Without the copy the first camera's data can silently
        # become garbage once the second camera is rendered.
        frame = frame[::-1].copy()
        frames.append(frame)
    result = np.concatenate(frames, axis=1)
    # Normalise to uint8. sim.render() can return float32 or values outside
    # [0, 255] during physics instability on a failed pick attempt — ffmpeg
    # silently encodes garbage when given non-uint8 data, corrupting that
    # episode's video fragment.
    if result.dtype != np.uint8:
        scale = 255.0 if result.max() <= 1.0 else 1.0
        result = np.clip(result * scale, 0, 255).astype(np.uint8)
    return result


def rollout(policy, env, horizon, video_writer=None, video_skip=5, episode_idx=0,
            render=False):
    """
    Run a single rollout episode on a ModifiedLiftEnv.

    Args:
        policy: RolloutPolicy instance
        env: ModifiedLiftEnv instance
        horizon: max steps
        video_writer: imageio writer (or None to skip video)
        video_skip: record every N-th frame
        episode_idx: episode number (for video labeling)

    Returns:
        stats: dict with Return, Horizon, Success_Rate
    """
    policy.start_episode()
    obs = env.reset()

    total_reward = 0.0
    success = False
    step_i = 0
    crashed = False

    _err_frame = np.zeros((256, 512, 3), dtype=np.uint8)
    _err_frame[:, :, 0] = 180  # red tint
    _sep_frame = np.zeros((256, 512, 3), dtype=np.uint8)

    def _write_or_placeholder(frame, ctx):
        """Write frame; on failure print a loud warning and write red placeholder.
        Keeps video write errors separate from simulation errors so a single
        bad frame cannot corrupt the writer state for subsequent episodes."""
        try:
            if frame.dtype != np.uint8:
                raise RuntimeError(f"frame dtype is {frame.dtype}, expected uint8")
            video_writer.append_data(frame)
        except Exception as write_exc:
            print(f"\n  [rollout ep={episode_idx}] VIDEO WRITE ERROR ({ctx}): {write_exc}")
            try:
                video_writer.append_data(_err_frame)
            except Exception:
                pass  # writer pipe is fully broken; nothing recoverable here

    try:
        for step_i in range(horizon):
            # Record video frame
            if video_writer is not None and step_i % video_skip == 0:
                frame = _render_frame(env)
                _write_or_placeholder(frame, f"step={step_i}")

            act = policy(ob=obs)
            next_obs, r, done, info = env.step(act)
            if render:
                env.render()
            total_reward += r
            success = env.env._check_success()
            if done or success:
                if video_writer is not None:
                    frame = _render_frame(env)
                    _write_or_placeholder(frame, "final")
                break
            obs = deepcopy(next_obs)

    except Exception:
        crashed = True
        print(f"\n  [rollout ep={episode_idx} step={step_i}] simulation exception:")
        traceback.print_exc()
        if video_writer is not None:
            for _ in range(20):
                try:
                    video_writer.append_data(_err_frame)
                except Exception as e:
                    print(f"  [rollout ep={episode_idx}] could not write error frame: {e}")
                    break

    # Separator between episodes
    if video_writer is not None:
        for _ in range(10):
            try:
                video_writer.append_data(_sep_frame)
            except Exception as e:
                print(f"  [rollout ep={episode_idx}] could not write separator frame: {e}")
                break

    return dict(
        Return=total_reward,
        Horizon=(step_i + 1),
        Success_Rate=float(success),
        Crashed=float(crashed),
    )


def evaluate_checkpoint(
    ckpt_path,
    scene_config,
    n_rollouts=50,
    horizon=400,
    device=None,
    verbose=True,
    video_path=None,
    video_skip=5,
    render=False,
):
    """
    Evaluate a trained checkpoint on a modified scene.

    Args:
        ckpt_path: path to .pth checkpoint
        scene_config: SceneConfig defining visual modifications
        n_rollouts: number of evaluation episodes
        horizon: max steps per episode
        device: torch device (auto-detected if None)
        verbose: print per-episode progress
        video_path: if set, save all rollouts into this single video file
        video_skip: record every N-th frame to video

    Returns:
        dict with avg_return, avg_horizon, success_rate, num_success
    """
    if device is None:
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    policy, ckpt_dict = _load_policy(ckpt_path, device)
    env = create_env(scene_config, has_renderer=render)

    video_writer = None
    if video_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
        # libx264 with -g 1 forces a keyframe on every frame so each episode's
        # segment is independently decodable when seeking. Without this, a
        # short failed-pick episode may contain no keyframe and appear corrupted.
        video_writer = imageio.get_writer(video_path, fps=20, codec="libx264",
                                          output_params=["-crf", "23", "-g", "1"])

    if verbose:
        print(f"\nEvaluating scene '{scene_config.name}' "
              f"({n_rollouts} rollouts, horizon={horizon})")
        if video_path:
            print(f"  Saving video to {video_path}")

    seed = scene_config.seed if scene_config.seed is not None else 42

    all_stats = []
    try:
        for i in range(n_rollouts):
            # Seed torch per episode so GMM action sampling is
            # deterministic and independent of evaluation order.
            ep_seed = seed * 100_000 + i
            torch.manual_seed(ep_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(ep_seed)

            stats = rollout(policy, env, horizon,
                            video_writer=video_writer, video_skip=video_skip,
                            episode_idx=i, render=render)
            all_stats.append(stats)

            if verbose:
                crashed = stats.get("Crashed", 0) > 0
                status = "CRASH" if crashed else ("SUCCESS" if stats["Success_Rate"] > 0 else "fail")
                print(f"  Episode {i+1:3d}/{n_rollouts}: "
                      f"H={stats['Horizon']:4d}  R={stats['Return']:.3f}  {status}")
    finally:
        # Always close the writer — this flushes ffmpeg and finalises the MOOV
        # atom, keeping the file playable even if we crashed mid-run.
        if video_writer is not None:
            video_writer.close()
        env.close()

    if not all_stats:
        return {"scene": scene_config.name, "success_rate": 0.0,
                "avg_return": 0.0, "avg_horizon": 0.0,
                "num_success": 0, "n_rollouts": 0}

    merged = TensorUtils.list_of_flat_dict_to_dict_of_list(all_stats)
    results = {
        "scene": scene_config.name,
        "avg_return": float(np.mean(merged["Return"])),
        "avg_horizon": float(np.mean(merged["Horizon"])),
        "success_rate": float(np.mean(merged["Success_Rate"])),
        "num_success": int(np.sum(merged["Success_Rate"])),
        "num_crashed": int(np.sum(merged["Crashed"])),
        "n_rollouts": len(all_stats),
    }

    if verbose:
        n = results["n_rollouts"]
        print(f"\n--- Results for '{scene_config.name}' ---")
        print(f"  Success rate: {results['success_rate']:.1%} "
              f"({results['num_success']}/{n})")
        if results["num_crashed"] > 0:
            print(f"  Crashed:      {results['num_crashed']}/{n}")
        print(f"  Avg return:   {results['avg_return']:.3f}")
        print(f"  Avg horizon:  {results['avg_horizon']:.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on modified Lift scenes"
    )
    parser.add_argument("--agent", type=str, required=True,
                        help="path to saved checkpoint .pth file")
    parser.add_argument("--preset", type=str, default=None,
                        choices=list(PRESETS.keys()),
                        help="scene preset name")
    parser.add_argument("--all_presets", action="store_true",
                        help="evaluate on all presets")
    parser.add_argument("--n_rollouts", type=int, default=50,
                        help="number of rollout episodes per scene")
    parser.add_argument("--horizon", type=int, default=400,
                        help="max steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")
    parser.add_argument("--video_path", type=str, default=None,
                        help="path to save single video file with all rollouts "
                             "(e.g. scene/videos/default.mp4)")
    parser.add_argument("--video_skip", type=int, default=5,
                        help="record every N-th frame to video")
    parser.add_argument("--render", action="store_true",
                        help="open on-screen viewer to watch rollouts live")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    if args.all_presets:
        preset_names = list(PRESETS.keys())
    elif args.preset:
        preset_names = [args.preset]
    else:
        preset_names = ["default"]

    all_results = []
    for name in preset_names:
        config = deepcopy(PRESETS[name])
        config.seed = args.seed

        # Auto-generate video path per preset if --video_path is a directory
        vpath = args.video_path
        if vpath is not None and len(preset_names) > 1:
            # When running multiple presets, treat video_path as a directory
            os.makedirs(vpath, exist_ok=True)
            vpath = os.path.join(vpath, f"{name}.mp4")

        results = evaluate_checkpoint(
            ckpt_path=args.agent,
            scene_config=config,
            n_rollouts=args.n_rollouts,
            horizon=args.horizon,
            device=device,
            video_path=vpath,
            video_skip=args.video_skip,
            render=args.render,
        )
        all_results.append(results)

    # Summary table
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print(f"{'Scene':<20} {'Success':>10} {'Avg Return':>12} {'Avg H':>8}")
        print("-" * 60)
        for r in all_results:
            print(f"{r['scene']:<20} "
                  f"{r['success_rate']:>9.1%} "
                  f"{r['avg_return']:>12.3f} "
                  f"{r['avg_horizon']:>8.1f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
