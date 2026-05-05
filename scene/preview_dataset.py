"""
Preview starting positions of a dataset without running any policy.

Restores each episode's saved RNG state, resets the env, and shows the
initial frame. No steps are taken, no agent involved. Press any key in
the OpenCV window (or wait --pause seconds) to advance to the next
episode.

Usage:
    python scene/preview_dataset.py --dataset scene/custom/default
    python scene/preview_dataset.py --dataset scene/custom/hard --n 5
    python scene/preview_dataset.py --dataset scene/custom/default --range 0:3 --video_path preview.mp4
"""
import argparse
import os
import sys

import numpy as np

_SCENE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCENE_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCENE_DIR)

from factory import create_env
from evaluate_dataset import load_dataset, _select_episodes, _parse_range  # noqa: F401


def main():
    p = argparse.ArgumentParser(
        description="Render dataset starting positions without stepping the agent")
    p.add_argument("--dataset", type=str, required=True,
                   help="path to dataset directory (from generate_dataset.py)")

    # Episode selection (mirrors evaluate_dataset.py)
    sel = p.add_argument_group("episode selection")
    sel.add_argument("--n", type=int, default=None,
                     help="number of episodes to preview (default: all)")
    sel.add_argument("--range", type=str, default=None, dest="index_range",
                     help="preview episodes i to j, e.g. '0:10' or '5-15'")
    sel.add_argument("--random", action="store_true",
                     help="randomly select episodes (combine with --n)")
    sel.add_argument("--selection_seed", type=int, default=42,
                     help="seed for random episode selection (default: 42)")

    p.add_argument("--no_render", action="store_true",
                   help="skip the on-screen viewer (useful with --video_path)")
    p.add_argument("--pause", type=float, default=0.0,
                   help="auto-advance after this many seconds "
                        "(0 = wait for keypress; default: 0)")
    p.add_argument("--video_path", type=str, default=None,
                   help="save one frame per episode as an mp4")
    args = p.parse_args()

    config, states, metadata, description = load_dataset(args.dataset)
    n_total = len(states)

    print(f"Dataset: {args.dataset}")
    print(f"  Preset:    {config.name}")
    print(f"  Total eps: {n_total}")
    if description:
        print(f"  Desc:      {description}")

    indices = _select_episodes(n_total, args)
    print(f"Previewing {len(indices)} episode(s): {indices[:10]}"
          f"{' ...' if len(indices) > 10 else ''}")

    env = create_env(config, has_renderer=not args.no_render)

    video_writer = None
    if args.video_path is not None:
        import imageio
        os.makedirs(os.path.dirname(os.path.abspath(args.video_path)) or ".",
                    exist_ok=True)
        video_writer = imageio.get_writer(
            args.video_path, fps=2, codec="libx264",
            output_params=["-crf", "23", "-g", "1"],
        )

    try:
        import cv2
    except ImportError:
        cv2 = None

    try:
        for eval_i, ep_idx in enumerate(indices):
            state = states[ep_idx]
            np.random.set_state(state["global_rng"])
            env._rng.set_state(state["env_rng"])
            env.reset()
            env.env.sim.forward()

            # Offscreen frame for video / fallback display.
            frames = []
            for cam in ("agentview", "robot0_eye_in_hand"):
                f = env.env.sim.render(height=256, width=256, camera_name=cam)
                frames.append(f[::-1].copy())
            combined = np.concatenate(frames, axis=1)
            if combined.dtype != np.uint8:
                scale = 255.0 if combined.max() <= 1.0 else 1.0
                combined = np.clip(combined * scale, 0, 255).astype(np.uint8)

            if video_writer is not None:
                video_writer.append_data(combined)

            if not args.no_render:
                env.env.render()
                if cv2 is not None:
                    cv2.imshow("Starting position (agentview | eye_in_hand)",
                               cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                    wait_ms = int(args.pause * 1000) if args.pause > 0 else 0
                    cv2.waitKey(wait_ms)

            print(f"  [{eval_i + 1}/{len(indices)}] episode {ep_idx}")
    finally:
        if video_writer is not None:
            video_writer.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
