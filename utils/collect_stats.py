"""
Batch-run scene/evaluate_dataset.py across many (agent, dataset, seed) combos.

Designed for long unattended sessions:
  * Each job runs in an isolated subprocess so a single crash/OOM does not
    kill the whole session.
  * Per-job timeout, retries, and a log file per job under stats_dir/logs/.
  * Ctrl+C is caught so the current session summary still prints.
  * Always appends rows to stats_runs.csv / stats_episodes.csv — resumption
    works simply by re-running; duplicate rows are handled in
    calculate_stats.py via run_id.

Config format (YAML):

    defaults:
      horizon: 400
      seed: 42
      selection_seed: 42
      n: null           # or an int; null -> all episodes
      range: null       # e.g. "0:20"
      random: false
      timeout_sec: 3600 # per-job timeout
      retries: 0        # re-run a job this many times on failure

    jobs:
      - agent: train_runs/resnet_lstm_2000/.../model_epoch_2000.pth
        dataset: scene/custom/default
        n: 20
        repeats: 10             # expands to 10 runs, seed = base_seed + i
      - agent: train_runs/vq_bc_rnn_image/.../model.pth
        dataset: scene/custom/hard
        seeds: [42, 100, 200]   # explicit list of seeds
      - agent: train_runs/dino_transformer_2000/.../model.pth
        dataset: scene/custom/default
        # no seeds / no repeats -> single run with default seed

Usage:
    python utils/collect_stats.py --config my_sweep.yaml
    python utils/collect_stats.py --config my_sweep.yaml --dry_run
"""
import argparse
import datetime
import os
import signal
import subprocess
import sys
import time
import uuid
from copy import deepcopy

try:
    import yaml
except ImportError:
    print("PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_UTILS_DIR)
_EVAL_SCRIPT = os.path.join(_PROJECT_ROOT, "scene", "evaluate_dataset.py")
_DEFAULT_STATS_DIR = os.path.join(_PROJECT_ROOT, "scene", "custom")


def _load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "jobs" not in cfg or not isinstance(cfg["jobs"], list):
        raise ValueError(f"Config {path} must have a top-level 'jobs' list")
    return cfg


def _expand_jobs(cfg):
    """Turn each YAML job entry into one or more concrete runs.

    A single YAML job may expand into N runs via `repeats` or `seeds`.
    Each returned run is a self-contained dict of params ready to convert
    into a CLI invocation.
    """
    defaults = cfg.get("defaults") or {}
    runs = []
    for job_i, job in enumerate(cfg["jobs"]):
        merged = deepcopy(defaults)
        merged.update(job)

        if "agent" not in merged or "dataset" not in merged:
            raise ValueError(f"Job #{job_i} is missing 'agent' or 'dataset'")

        base_seed = int(merged.get("seed", 42))
        if "seeds" in merged and merged["seeds"]:
            seeds = [int(s) for s in merged["seeds"]]
        elif "repeats" in merged and merged["repeats"]:
            repeats = int(merged["repeats"])
            seeds = [base_seed + i for i in range(repeats)]
        else:
            seeds = [base_seed]

        for rep_i, seed in enumerate(seeds):
            run = {
                "job_index": job_i,
                "repeat_index": rep_i,
                "n_repeats": len(seeds),
                "agent": merged["agent"],
                "dataset": merged["dataset"],
                "seed": seed,
                "selection_seed": int(merged.get("selection_seed", 42)),
                "horizon": int(merged.get("horizon", 400)),
                "n": merged.get("n"),
                "range": merged.get("range"),
                "random": bool(merged.get("random", False)),
                "video_path": merged.get("video_path"),
                "timeout_sec": merged.get("timeout_sec"),
                "retries": int(merged.get("retries", 0)),
                "extra_args": list(merged.get("extra_args") or []),
            }
            runs.append(run)
    return runs


def _build_cmd(run, stats_dir, run_id):
    cmd = [
        sys.executable, _EVAL_SCRIPT,
        "--agent", str(run["agent"]),
        "--dataset", str(run["dataset"]),
        "--seed", str(run["seed"]),
        "--selection_seed", str(run["selection_seed"]),
        "--horizon", str(run["horizon"]),
        "--stats_dir", stats_dir,
        "--run_id", run_id,
    ]
    if run["n"] is not None:
        cmd += ["--n", str(run["n"])]
    if run["range"]:
        cmd += ["--range", str(run["range"])]
    if run["random"]:
        cmd += ["--random"]
    if run["video_path"]:
        cmd += ["--video_path", str(run["video_path"])]
    cmd += list(run["extra_args"])
    return cmd


def _format_run_header(run, i, total):
    parts = [
        f"agent={os.path.basename(str(run['agent']))}",
        f"dataset={os.path.basename(str(run['dataset']))}",
        f"seed={run['seed']}",
    ]
    if run["n"] is not None:
        parts.append(f"n={run['n']}")
    if run["range"]:
        parts.append(f"range={run['range']}")
    if run["random"]:
        parts.append("random")
    return f"[{i}/{total}] " + "  ".join(parts)


def _run_once(cmd, log_path, timeout_sec):
    """Run one subprocess, streaming output to log file AND stdout."""
    with open(log_path, "w") as log_f:
        log_f.write("$ " + " ".join(cmd) + "\n\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        start = time.time()
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)
                log_f.flush()
                if timeout_sec and (time.time() - start) > timeout_sec:
                    proc.kill()
                    proc.wait()
                    log_f.write(f"\n[TIMEOUT after {timeout_sec}s]\n")
                    return "timeout", proc.returncode or -1
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            log_f.write("\n[INTERRUPTED by user]\n")
            raise
    if proc.returncode == 0:
        return "ok", 0
    return "fail", proc.returncode


def main():
    p = argparse.ArgumentParser(
        description="Batch-run evaluate_dataset.py jobs from a YAML config")
    p.add_argument("--config", required=True, help="path to YAML config")
    p.add_argument("--stats_dir", default=_DEFAULT_STATS_DIR,
                   help=f"where to write stats CSVs (default: {_DEFAULT_STATS_DIR})")
    p.add_argument("--dry_run", action="store_true",
                   help="print the expanded commands and exit, don't run")
    p.add_argument("--start", type=int, default=0,
                   help="skip the first N expanded runs (useful to resume manually)")
    p.add_argument("--limit", type=int, default=None,
                   help="only run this many runs and stop")
    args = p.parse_args()

    cfg = _load_config(args.config)
    runs = _expand_jobs(cfg)

    if args.start:
        runs = runs[args.start:]
    if args.limit is not None:
        runs = runs[:args.limit]

    total = len(runs)
    if total == 0:
        print("No runs to execute.")
        return

    os.makedirs(args.stats_dir, exist_ok=True)
    log_dir = os.path.join(args.stats_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Session {session_id}: {total} run(s) planned")
    print(f"Stats dir: {args.stats_dir}")
    print(f"Logs dir:  {log_dir}")
    print("-" * 72)

    if args.dry_run:
        for i, run in enumerate(runs, 1):
            run_id = uuid.uuid4().hex[:12]
            cmd = _build_cmd(run, args.stats_dir, run_id)
            print(_format_run_header(run, i, total))
            print("  $ " + " ".join(cmd))
        return

    summary = {"ok": 0, "fail": 0, "timeout": 0}
    t_session = time.time()
    interrupted = False

    for i, run in enumerate(runs, 1):
        print("\n" + _format_run_header(run, i, total))
        status = "fail"
        rc = -1
        for attempt in range(run["retries"] + 1):
            run_id = uuid.uuid4().hex[:12]
            cmd = _build_cmd(run, args.stats_dir, run_id)
            log_name = (f"{session_id}_run{i:04d}_attempt{attempt}_"
                        f"{run_id}.log")
            log_path = os.path.join(log_dir, log_name)
            if attempt > 0:
                print(f"  retry {attempt}/{run['retries']}")
            t0 = time.time()
            try:
                status, rc = _run_once(cmd, log_path, run["timeout_sec"])
            except KeyboardInterrupt:
                interrupted = True
                print("\n[Ctrl+C] stopping session")
                break
            except Exception as e:  # subprocess launch failure, etc.
                status, rc = "fail", -1
                with open(log_path, "a") as f:
                    f.write(f"\n[launcher exception] {e!r}\n")
            dt = time.time() - t0
            print(f"  -> {status} (rc={rc}, {dt:.1f}s, log={log_path})")
            if status == "ok":
                break
        if interrupted:
            break
        summary[status] = summary.get(status, 0) + 1

    dt_session = time.time() - t_session
    print("\n" + "=" * 72)
    print(f"Session {session_id} done in {dt_session:.1f}s")
    print(f"  ok:      {summary.get('ok', 0)}")
    print(f"  fail:    {summary.get('fail', 0)}")
    print(f"  timeout: {summary.get('timeout', 0)}")
    if interrupted:
        print("  (interrupted by user before all runs completed)")


if __name__ == "__main__":
    main()
