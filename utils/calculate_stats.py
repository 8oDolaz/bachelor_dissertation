"""
Aggregate stats from scene/custom/stats_runs.csv + stats_episodes.csv.

Primary aggregation is per-(agent, dataset): for each (agent, dataset)
combination, compute the overall success rate across all per-episode
rollouts that pass the CLI filters. Per-episode and per-run breakdowns
are also available.

Filters (repeatable --filter key=pattern) use fnmatch-style globs:
  agent      — glob against the 'agent' column (checkpoint path)
  dataset    — glob against the 'dataset' column (dataset dir path)
  scene_name — glob against the scene preset name recorded at eval time
  seed       — exact match (int)
  run_id     — exact match
  arg_n, arg_range, arg_random, selection_seed, horizon — exact match

Usage:
    # Headline table: SR per (agent, dataset) across everything collected
    python utils/calculate_stats.py

    # Only VQ agents
    python utils/calculate_stats.py --filter agent='*vq*'

    # Only the default dataset
    python utils/calculate_stats.py --filter dataset='*/default'

    # Show per-episode SR (average across repeats of the same episode)
    python utils/calculate_stats.py --group episode

    # Show per-run rows (one row per evaluate_dataset call)
    python utils/calculate_stats.py --group run

    # Write aggregate to CSV
    python utils/calculate_stats.py --output results.csv
"""
import argparse
import csv
import fnmatch
import os
import sys
from collections import defaultdict


_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_UTILS_DIR)
_DEFAULT_STATS_DIR = os.path.join(_PROJECT_ROOT, "scene", "custom")


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _parse_filter(expr):
    if "=" not in expr:
        raise ValueError(f"--filter expects key=value, got {expr!r}")
    k, v = expr.split("=", 1)
    return k.strip(), v.strip()


def _match_run(run, filters):
    for key, pattern in filters:
        if key not in run:
            return False
        val = run[key]
        # Numeric-looking keys: exact match (stringified)
        if key in ("seed", "selection_seed", "horizon", "arg_random",
                   "arg_n", "run_id"):
            if str(val) != str(pattern):
                return False
        else:
            if not fnmatch.fnmatch(str(val), pattern):
                return False
    return True


def _safe_float(x, default=float("nan")):
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def _mean(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs):
    xs = [x for x in xs if x == x]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def load_data(stats_dir, filters):
    runs = _read_csv(os.path.join(stats_dir, "stats_runs.csv"))
    episodes = _read_csv(os.path.join(stats_dir, "stats_episodes.csv"))

    kept_runs = [r for r in runs if _match_run(r, filters)]
    kept_run_ids = {r["run_id"] for r in kept_runs}
    kept_episodes = [e for e in episodes if e["run_id"] in kept_run_ids]
    return kept_runs, kept_episodes


def _agent_dataset_key(run):
    return run["agent"], run["dataset"]


def aggregate_agent_dataset(runs, episodes):
    """Primary aggregation: per (agent, dataset)."""
    runs_by_id = {r["run_id"]: r for r in runs}
    eps_by_group = defaultdict(list)   # (agent, dataset) -> list of ep rows
    run_sr_by_group = defaultdict(list)
    ep_indices_by_group = defaultdict(set)

    for e in episodes:
        run = runs_by_id.get(e["run_id"])
        if run is None:
            continue
        key = _agent_dataset_key(run)
        eps_by_group[key].append(e)
        ep_indices_by_group[key].add(e["episode_idx"])

    for r in runs:
        key = _agent_dataset_key(r)
        run_sr_by_group[key].append(_safe_float(r.get("success_rate")))

    rows = []
    for key in sorted(eps_by_group.keys()):
        eps = eps_by_group[key]
        successes = [int(e["success"]) for e in eps]
        crashes = [int(e.get("crashed") or 0) for e in eps]
        rets = [_safe_float(e["return"]) for e in eps]
        agent, dataset = key
        rows.append({
            "agent": agent,
            "dataset": dataset,
            "n_runs": len(run_sr_by_group[key]),
            "n_unique_episodes": len(ep_indices_by_group[key]),
            "n_rollouts": len(eps),
            "success_rate": _mean(successes),
            "success_rate_per_run_std": _std(run_sr_by_group[key]),
            "num_success": sum(successes),
            "num_crashed": sum(crashes),
            "avg_return": _mean(rets),
        })
    return rows


def aggregate_per_episode(runs, episodes):
    """Per (agent, dataset, episode_idx): SR averaged across repeats."""
    runs_by_id = {r["run_id"]: r for r in runs}
    groups = defaultdict(list)
    for e in episodes:
        run = runs_by_id.get(e["run_id"])
        if run is None:
            continue
        key = (run["agent"], run["dataset"], int(e["episode_idx"]))
        groups[key].append(e)

    rows = []
    for (agent, dataset, ep_idx) in sorted(groups.keys()):
        eps = groups[(agent, dataset, ep_idx)]
        successes = [int(e["success"]) for e in eps]
        rets = [_safe_float(e["return"]) for e in eps]
        rows.append({
            "agent": agent,
            "dataset": dataset,
            "episode_idx": ep_idx,
            "n_rollouts": len(eps),
            "success_rate": _mean(successes),
            "num_success": sum(successes),
            "avg_return": _mean(rets),
        })
    return rows


def aggregate_runs(runs):
    rows = []
    for r in runs:
        rows.append({
            "run_id": r["run_id"],
            "timestamp": r.get("timestamp", ""),
            "agent": r["agent"],
            "dataset": r["dataset"],
            "seed": r.get("seed", ""),
            "n_evaluated": r.get("n_evaluated", ""),
            "success_rate": _safe_float(r.get("success_rate")),
            "num_success": r.get("num_success", ""),
            "num_crashed": r.get("num_crashed", ""),
            "avg_return": _safe_float(r.get("avg_return")),
        })
    return rows


def _trunc(s, n):
    s = str(s)
    return s if len(s) <= n else "…" + s[-(n - 1):]


def _print_table(rows, columns, widths=None):
    if not rows:
        print("(no rows)")
        return
    widths = widths or {}
    hdr = []
    for c in columns:
        w = widths.get(c, 14)
        hdr.append(f"{c:<{w}}")
    print("  ".join(hdr))
    print("-" * (sum(widths.get(c, 14) for c in columns) + 2 * (len(columns) - 1)))
    for row in rows:
        parts = []
        for c in columns:
            w = widths.get(c, 14)
            v = row.get(c, "")
            if isinstance(v, float):
                v = f"{v:.3f}"
            parts.append(f"{_trunc(v, w):<{w}}")
        print("  ".join(parts))


def _write_csv(path, rows):
    if not rows:
        print(f"(no rows to write to {path})")
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {path}")


def main():
    p = argparse.ArgumentParser(
        description="Aggregate stats from evaluate_dataset CSV logs")
    p.add_argument("--stats_dir", default=_DEFAULT_STATS_DIR,
                   help=f"directory holding stats_runs.csv / stats_episodes.csv "
                        f"(default: {_DEFAULT_STATS_DIR})")
    p.add_argument("--filter", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="filter runs by column; supports globs. Repeatable.")
    p.add_argument("--group", choices=["agent_dataset", "episode", "run"],
                   default="agent_dataset",
                   help="aggregation level (default: agent_dataset)")
    p.add_argument("--output", default=None,
                   help="write result rows to this CSV path as well")
    args = p.parse_args()

    try:
        filters = [_parse_filter(f) for f in args.filter]
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    runs, episodes = load_data(args.stats_dir, filters)
    print(f"Loaded {len(runs)} run(s) and {len(episodes)} episode row(s) "
          f"after filtering from {args.stats_dir}")
    if not runs:
        return

    if args.group == "agent_dataset":
        rows = aggregate_agent_dataset(runs, episodes)
        cols = ["agent", "dataset", "n_runs", "n_unique_episodes",
                "n_rollouts", "success_rate", "success_rate_per_run_std",
                "num_success", "num_crashed", "avg_return"]
        widths = {"agent": 40, "dataset": 30, "n_runs": 6,
                  "n_unique_episodes": 8, "n_rollouts": 10,
                  "success_rate": 12, "success_rate_per_run_std": 10,
                  "num_success": 11, "num_crashed": 11, "avg_return": 10}
    elif args.group == "episode":
        rows = aggregate_per_episode(runs, episodes)
        cols = ["agent", "dataset", "episode_idx", "n_rollouts",
                "success_rate", "num_success", "avg_return"]
        widths = {"agent": 40, "dataset": 25, "episode_idx": 11,
                  "n_rollouts": 10, "success_rate": 12,
                  "num_success": 11, "avg_return": 10}
    else:  # run
        rows = aggregate_runs(runs)
        cols = ["run_id", "timestamp", "agent", "dataset", "seed",
                "n_evaluated", "success_rate", "num_success",
                "num_crashed", "avg_return"]
        widths = {"run_id": 14, "timestamp": 19, "agent": 30,
                  "dataset": 22, "seed": 6, "n_evaluated": 10,
                  "success_rate": 12, "num_success": 11,
                  "num_crashed": 11, "avg_return": 10}

    print()
    _print_table(rows, cols, widths)
    if args.output:
        _write_csv(args.output, rows)


if __name__ == "__main__":
    main()
