"""
Evaluate a trained checkpoint on a pre-generated test dataset.

Loads per-episode RNG states from a dataset directory and restores them
before each env.reset(), producing identical scenes every time. This
guarantees that different model checkpoints are evaluated on exactly the
same visual conditions.

Usage:
    # Evaluate all episodes in the dataset
    python scene/evaluate_dataset.py --agent path/to/model.pth --dataset scene/datasets/default_50

    # First 10 episodes only
    python scene/evaluate_dataset.py --agent path/to/model.pth --dataset scene/datasets/default_50 --n 10

    # Episodes 5 through 15
    python scene/evaluate_dataset.py --agent path/to/model.pth --dataset scene/datasets/default_50 --range 5:15

    # 20 randomly selected episodes
    python scene/evaluate_dataset.py --agent path/to/model.pth --dataset scene/datasets/default_50 --n 20 --random

    # With video recording
    python scene/evaluate_dataset.py --agent path/to/model.pth --dataset scene/datasets/default_50 --video_path scene/videos/eval.mp4
"""
import argparse
import csv
import dataclasses
import datetime
import json
import os
import pickle
import sys
import uuid
from copy import deepcopy

import numpy as np
import torch
import imageio

_SCENE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCENE_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCENE_DIR)

# Force deterministic CUDA operations. The GMM policy samples actions via
# torch.distributions which consumes from torch's RNG; cuDNN non-determinism
# causes tiny numerical differences in the LSTM that compound and push the
# GMM into a different mixture mode → wildly different actions.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils

from config import SceneConfig, TextureConfig, LightingConfig, CameraConfig, PlacementConfig
from factory import create_env
from evaluate import _load_policy, rollout


def _reconstruct_config(d):
    """Reconstruct a SceneConfig from a plain dict (JSON-deserialized).

    dataclasses.asdict turns tuples into lists, so we convert them back.
    """
    placement = d.get("placement", {})
    for key in ("x_range", "y_range"):
        if key in placement and isinstance(placement[key], list):
            placement[key] = tuple(placement[key])

    texture = d.get("texture", {})
    if "texture_variations" in texture and isinstance(texture["texture_variations"], list):
        texture["texture_variations"] = tuple(texture["texture_variations"])

    return SceneConfig(
        name=d.get("name", "default"),
        seed=d.get("seed"),
        texture=TextureConfig(**texture),
        lighting=LightingConfig(**d.get("lighting", {})),
        camera=CameraConfig(**d.get("camera", {})),
        placement=PlacementConfig(**placement),
        randomize_every_n_steps=d.get("randomize_every_n_steps", 0),
    )


def load_dataset(dataset_dir):
    """Load dataset config, per-episode RNG states, and metadata."""
    with open(os.path.join(dataset_dir, "config.json")) as f:
        config_dict = json.load(f)
    config = _reconstruct_config(config_dict)

    with open(os.path.join(dataset_dir, "states.pkl"), "rb") as f:
        states = pickle.load(f)

    with open(os.path.join(dataset_dir, "metadata.json")) as f:
        metadata = json.load(f)

    description = ""
    desc_path = os.path.join(dataset_dir, "description.txt")
    if os.path.exists(desc_path):
        with open(desc_path) as f:
            description = f.read().strip()

    return config, states, metadata, description


def _parse_range(range_str):
    """Parse 'i:j' or 'i-j' into (start, end) tuple."""
    for sep in [":", "-"]:
        if sep in range_str:
            parts = range_str.split(sep, 1)
            return int(parts[0]), int(parts[1])
    raise ValueError(f"Invalid range format: '{range_str}'. Use 'i:j' or 'i-j'.")


def _select_episodes(n_total, args):
    """Return sorted list of episode indices based on CLI selection flags."""
    if args.index_range is not None:
        start, end = _parse_range(args.index_range)
        start = max(0, start)
        end = min(end, n_total)
        indices = list(range(start, end))
    elif args.n is not None:
        n = min(args.n, n_total)
        if args.random:
            rng = np.random.RandomState(args.selection_seed)
            indices = sorted(rng.choice(n_total, size=n, replace=False).tolist())
        else:
            indices = list(range(n))
    else:
        indices = list(range(n_total))
    return indices


_RUNS_CSV_COLUMNS = [
    "run_id", "timestamp", "agent", "dataset", "scene_name",
    "arg_n", "arg_range", "arg_random", "selection_seed", "seed",
    "horizon", "n_total", "n_evaluated",
    "indices", "successes",
    "num_success", "num_crashed", "success_rate",
    "avg_return", "avg_horizon",
]

_EPISODES_CSV_COLUMNS = [
    "run_id", "episode_idx", "success", "crashed", "return", "horizon",
]


def _append_csv(path, columns, rows):
    """Append rows to CSV, creating header if file doesn't exist."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def write_stats(stats_dir, run_record, episode_records):
    """Append one run summary row and per-episode rows to the stats CSVs."""
    runs_path = os.path.join(stats_dir, "stats_runs.csv")
    episodes_path = os.path.join(stats_dir, "stats_episodes.csv")
    _append_csv(runs_path, _RUNS_CSV_COLUMNS, [run_record])
    _append_csv(episodes_path, _EPISODES_CSV_COLUMNS, episode_records)


def _seed_torch_for_episode(seed, ep_idx):
    """Deterministically seed torch RNG for a specific episode.

    Using (seed, ep_idx) ensures the same episode always gets the same
    torch RNG state regardless of evaluation order or how many other
    episodes were run before it.
    """
    ep_seed = seed * 100_000 + ep_idx
    torch.manual_seed(ep_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(ep_seed)


def evaluate_on_dataset(
    ckpt_path,
    dataset_dir,
    indices,
    config,
    states,
    horizon=400,
    device=None,
    video_path=None,
    video_skip=5,
    render=False,
    seed=42,
):
    """
    Evaluate a checkpoint on selected episodes from a saved dataset.

    For each episode, restores the numpy RNG state captured during dataset
    generation (identical scenes) AND seeds torch deterministically per
    episode (identical GMM action samples).
    """
    if device is None:
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    policy, _ = _load_policy(ckpt_path, device)
    env = create_env(config, has_renderer=render)

    video_writer = None
    if video_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
        video_writer = imageio.get_writer(
            video_path, fps=20, codec="libx264",
            output_params=["-crf", "23", "-g", "1"],
        )

    n_eval = len(indices)
    all_stats = []
    try:
        for eval_i, ep_idx in enumerate(indices):
            state = states[ep_idx]

            # Restore numpy RNG states → identical scene (placement,
            # colors, lighting, camera perturbations).
            np.random.set_state(state["global_rng"])
            env._rng.set_state(state["env_rng"])

            # Seed torch RNG per episode → identical GMM action samples.
            # Uses (seed, ep_idx) so the same episode always gets the
            # same torch state regardless of evaluation order.
            _seed_torch_for_episode(seed, ep_idx)

            stats = rollout(
                policy, env, horizon,
                video_writer=video_writer, video_skip=video_skip,
                episode_idx=ep_idx, render=render,
            )
            all_stats.append(stats)

            crashed = stats.get("Crashed", 0) > 0
            status = ("CRASH" if crashed
                      else ("SUCCESS" if stats["Success_Rate"] > 0 else "fail"))
            print(f"  Episode {ep_idx:3d} ({eval_i + 1}/{n_eval}): "
                  f"H={stats['Horizon']:4d}  R={stats['Return']:.3f}  {status}")
    finally:
        if video_writer is not None:
            video_writer.close()
        env.close()

    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on a pre-generated test dataset",
    )
    parser.add_argument("--agent", type=str, required=True,
                        help="path to saved checkpoint .pth file")
    parser.add_argument("--dataset", type=str, required=True,
                        help="path to dataset directory (from generate_dataset.py)")

    # Episode selection
    select = parser.add_argument_group("episode selection")
    select.add_argument("--n", type=int, default=None,
                        help="number of episodes to evaluate (default: all)")
    select.add_argument("--range", type=str, default=None, dest="index_range",
                        help="evaluate episodes i to j, e.g. '0:10' or '5-15'")
    select.add_argument("--random", action="store_true",
                        help="randomly select episodes (combine with --n)")
    select.add_argument("--selection_seed", type=int, default=42,
                        help="seed for random episode selection (default: 42)")

    # Evaluation settings
    parser.add_argument("--horizon", type=int, default=400,
                        help="max steps per episode (default: 400)")
    parser.add_argument("--video_path", type=str, default=None,
                        help="path to save rollout video (.mp4)")
    parser.add_argument("--video_skip", type=int, default=5,
                        help="record every N-th frame to video (default: 5)")
    parser.add_argument("--render", action="store_true",
                        help="open on-screen viewer to watch rollouts live")
    parser.add_argument("--seed", type=int, default=42,
                        help="torch RNG seed for deterministic policy sampling (default: 42)")

    # Stats recording
    stats = parser.add_argument_group("stats recording")
    stats.add_argument("--stats_dir", type=str,
                       default=os.path.join(_SCENE_DIR, "custom"),
                       help="directory to write stats_runs.csv / stats_episodes.csv "
                            "(default: scene/custom)")
    stats.add_argument("--no_stats", action="store_true",
                       help="do not write CSV stats rows for this evaluation")
    stats.add_argument("--run_id", type=str, default=None,
                       help="explicit run_id to use in the stats CSVs "
                            "(default: auto-generated)")
    args = parser.parse_args()

    # Load dataset
    config, states, metadata, description = load_dataset(args.dataset)
    n_total = len(states)

    print(f"Dataset: {args.dataset}")
    print(f"  Preset:    {config.name}")
    print(f"  Total eps: {n_total}")
    if description:
        print(f"  Desc:      {description}")

    # Select episodes
    indices = _select_episodes(n_total, args)
    n_eval = len(indices)
    if n_eval == 0:
        print("No episodes selected.")
        return

    preview = str(indices[:10])[1:-1]
    if n_eval > 10:
        preview += ", ..."
    print(f"  Selected:  {n_eval} episodes [{preview}]")

    # Evaluate
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    all_stats = evaluate_on_dataset(
        ckpt_path=args.agent,
        dataset_dir=args.dataset,
        indices=indices,
        config=config,
        states=states,
        horizon=args.horizon,
        device=device,
        video_path=args.video_path,
        video_skip=args.video_skip,
        render=args.render,
        seed=args.seed,
    )

    if not all_stats:
        print("No episodes evaluated.")
        return

    merged = TensorUtils.list_of_flat_dict_to_dict_of_list(all_stats)
    results = {
        "dataset": args.dataset,
        "scene": config.name,
        "indices": indices,
        "avg_return": float(np.mean(merged["Return"])),
        "avg_horizon": float(np.mean(merged["Horizon"])),
        "success_rate": float(np.mean(merged["Success_Rate"])),
        "num_success": int(np.sum(merged["Success_Rate"])),
        "num_crashed": int(np.sum(merged["Crashed"])),
        "n_evaluated": n_eval,
    }

    print(f"\n--- Results for '{config.name}' ({n_eval} episodes) ---")
    print(f"  Success rate: {results['success_rate']:.1%} "
          f"({results['num_success']}/{n_eval})")
    if results["num_crashed"] > 0:
        print(f"  Crashed:      {results['num_crashed']}/{n_eval}")
    print(f"  Avg return:   {results['avg_return']:.3f}")
    print(f"  Avg horizon:  {results['avg_horizon']:.1f}")

    # Write stats CSVs
    if not args.no_stats:
        run_id = args.run_id or uuid.uuid4().hex[:12]
        successes = [int(s > 0) for s in merged["Success_Rate"]]
        crashes = [int(c > 0) for c in merged["Crashed"]]
        run_record = {
            "run_id": run_id,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "agent": args.agent,
            "dataset": args.dataset,
            "scene_name": config.name,
            "arg_n": args.n if args.n is not None else "",
            "arg_range": args.index_range if args.index_range is not None else "",
            "arg_random": int(bool(args.random)),
            "selection_seed": args.selection_seed,
            "seed": args.seed,
            "horizon": args.horizon,
            "n_total": n_total,
            "n_evaluated": n_eval,
            "indices": json.dumps(indices),
            "successes": json.dumps(successes),
            "num_success": results["num_success"],
            "num_crashed": results["num_crashed"],
            "success_rate": results["success_rate"],
            "avg_return": results["avg_return"],
            "avg_horizon": results["avg_horizon"],
        }
        episode_records = [
            {
                "run_id": run_id,
                "episode_idx": ep_idx,
                "success": successes[i],
                "crashed": crashes[i],
                "return": float(merged["Return"][i]),
                "horizon": int(merged["Horizon"][i]),
            }
            for i, ep_idx in enumerate(indices)
        ]
        write_stats(args.stats_dir, run_record, episode_records)
        print(f"  Stats:        {args.stats_dir}/stats_runs.csv "
              f"(run_id={run_id})")


if __name__ == "__main__":
    main()
