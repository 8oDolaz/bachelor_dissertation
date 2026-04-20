"""
Generate and save a reproducible test dataset for scene evaluation.

Creates a directory containing the scene configuration, per-episode RNG
states, and a description. The saved dataset can be replayed
deterministically by evaluate_dataset.py to produce identical scenes.

Usage:
    python scene/generate_dataset.py --preset default --n_episodes 50 --output_dir scene/datasets/default_50
    python scene/generate_dataset.py --preset hard --n_episodes 100 --seed 123 --output_dir scene/datasets/hard_100 \
        --description "Hard preset, 100 episodes, seed 123"
"""
import argparse
import dataclasses
import json
import os
import pickle
import sys
from copy import deepcopy

import numpy as np

_SCENE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCENE_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCENE_DIR)

from config import SceneConfig
from factory import PRESETS, create_env


def generate_dataset(scene_config, n_episodes, output_dir, description=""):
    """
    Generate RNG states for n_episodes and save as a reproducible dataset.

    For each episode, captures the full RNG state (both global numpy and
    the env-local RandomState) before env.reset(). Restoring these states
    before replay produces identical scenes — same placement, colors,
    lighting, camera perturbations.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save config as JSON
    config_dict = dataclasses.asdict(scene_config)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save description
    with open(os.path.join(output_dir, "description.txt"), "w") as f:
        f.write(description.strip() + "\n" if description.strip() else "")

    # Seed global RNG to match what evaluate.py does
    master_seed = scene_config.seed if scene_config.seed is not None else 42
    np.random.seed(master_seed)

    env = create_env(scene_config)

    states = []
    for i in range(n_episodes):
        # Capture RNG states BEFORE reset — restoring these and calling
        # reset() will reproduce the exact same scene.
        state = {
            "global_rng": np.random.get_state(),
            "env_rng": env._rng.get_state(),
        }
        states.append(state)

        # Run the reset to advance the RNG (we don't need the obs, just
        # need the RNG to be in the right state for the next episode)
        env.reset()
        print(f"  Generated episode {i + 1}/{n_episodes}")

    env.close()

    # Save states
    with open(os.path.join(output_dir, "states.pkl"), "wb") as f:
        pickle.dump(states, f)

    # Save metadata
    meta = {
        "n_episodes": n_episodes,
        "preset": scene_config.name,
        "seed": master_seed,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDataset saved to {output_dir}")
    print(f"  Episodes:  {n_episodes}")
    print(f"  Preset:    {scene_config.name}")
    print(f"  Seed:      {master_seed}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a reproducible test dataset for scene evaluation"
    )
    parser.add_argument("--preset", type=str, default="default",
                        choices=list(PRESETS.keys()),
                        help="scene preset name (default: default)")
    parser.add_argument("--n_episodes", type=int, default=50,
                        help="number of test episodes to generate (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="master random seed (default: 42)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="directory to save dataset into")
    parser.add_argument("--description", type=str, default="",
                        help="human-readable description of this dataset")
    args = parser.parse_args()

    config = deepcopy(PRESETS[args.preset])
    config.seed = args.seed

    print(f"Generating dataset: preset={args.preset}, "
          f"n_episodes={args.n_episodes}, seed={args.seed}")
    generate_dataset(config, args.n_episodes, args.output_dir, args.description)


if __name__ == "__main__":
    main()
