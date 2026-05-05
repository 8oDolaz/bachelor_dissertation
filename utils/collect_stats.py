"""
Batch-run scene/evaluate_dataset.py across many (agent, dataset, seed) combos.

Designed for long unattended sessions:
  * Workers run in parallel (--jobs N) using a ProcessPoolExecutor.
  * Each run writes to a private staging dir -> no CSV race conditions.
  * Pre-session backup of canonical CSVs to scene/custom/backups/.
  * Atomic manifest.json tracks completed runs -> --resume_session restarts.
  * Per-job timeout, retries, and a log file per attempt under stats_dir/logs/.
  * Ctrl+C is caught: in-flight workers terminated, completed runs preserved,
    final merge still runs so partial sessions are usable immediately.

Layout produced by a session:
    <stats_dir>/
        stats_runs.csv                           (canonical, appended at end)
        stats_episodes.csv                       (canonical, appended at end)
        backups/
            stats_runs.csv.bak.<session_id>
            stats_episodes.csv.bak.<session_id>
        _staging/<session_id>/
            manifest.json                        (planned + status per run)
            logs/run####_attempt#_<run_id>.log
            <run_id>/
                stats_runs.csv                   (1 row)
                stats_episodes.csv               (~100 rows)

Usage:
    # Dry-run: print expanded commands and exit
    python utils/collect_stats.py --config sweep.yaml --dry_run

    # Run with 4 parallel workers
    python utils/collect_stats.py --config sweep.yaml --jobs 4

    # Resume an interrupted session
    python utils/collect_stats.py --config sweep.yaml --jobs 4 \\
        --resume_session 20260505_143022

    # Re-merge staging -> canonical without re-running anything
    python utils/collect_stats.py --finalize 20260505_143022
"""
import argparse
import csv
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
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

_RUNS_CSV = "stats_runs.csv"
_EPISODES_CSV = "stats_episodes.csv"


# ===========================================================================
# Config loading and job expansion
# ===========================================================================

def _load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "jobs" not in cfg or not isinstance(cfg["jobs"], list):
        raise ValueError(f"Config {path} must have a top-level 'jobs' list")
    return cfg


def _expand_jobs(cfg):
    """Turn each YAML job entry into one or more concrete runs."""
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
            runs.append({
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
            })
    return runs


def _run_identity(run):
    """Stable identity tuple used for resume matching."""
    return (
        str(run["agent"]),
        str(run["dataset"]),
        int(run["seed"]),
        int(run["selection_seed"]),
        int(run["horizon"]),
        run["n"] if run["n"] is not None else "",
        str(run["range"] or ""),
        int(bool(run["random"])),
    )


# ===========================================================================
# Manifest (atomic write/read)
# ===========================================================================

def _write_json_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def _load_manifest(staging_dir):
    path = os.path.join(staging_dir, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save_manifest(staging_dir, manifest):
    path = os.path.join(staging_dir, "manifest.json")
    _write_json_atomic(path, manifest)


def _init_manifest(staging_dir, session_id, runs):
    manifest = {
        "session_id": session_id,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "runs": [
            {
                "index": i,
                "identity": list(_run_identity(run)),
                "agent": run["agent"],
                "dataset": run["dataset"],
                "seed": run["seed"],
                "status": "pending",
                "run_id": None,
                "attempts": 0,
                "duration_sec": None,
                "rc": None,
                "log_path": None,
            }
            for i, run in enumerate(runs)
        ],
    }
    _save_manifest(staging_dir, manifest)
    return manifest


def _update_manifest_entry(staging_dir, manifest, idx, **fields):
    manifest["runs"][idx].update(fields)
    _save_manifest(staging_dir, manifest)


# ===========================================================================
# Backup
# ===========================================================================

def _backup_canonical_csvs(stats_dir, session_id):
    backup_dir = os.path.join(stats_dir, "backups")
    os.makedirs(backup_dir, exist_ok=True)
    saved = []
    for name in (_RUNS_CSV, _EPISODES_CSV):
        src = os.path.join(stats_dir, name)
        if os.path.exists(src):
            dst = os.path.join(backup_dir, f"{name}.bak.{session_id}")
            shutil.copy2(src, dst)
            saved.append(dst)
    return saved


# ===========================================================================
# CSV merge (staging -> canonical)
# ===========================================================================

def _read_csv_rows(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _append_rows(path, header, rows):
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerows(rows)


def _merge_staging_csvs(staging_dir, stats_dir):
    """Append every staging <run_id>/stats_*.csv into the canonical CSVs."""
    runs_dst = os.path.join(stats_dir, _RUNS_CSV)
    episodes_dst = os.path.join(stats_dir, _EPISODES_CSV)
    runs_total = 0
    episodes_total = 0

    # Sort by run_id for deterministic output ordering across resumed sessions.
    run_dirs = sorted(
        d for d in os.listdir(staging_dir)
        if os.path.isdir(os.path.join(staging_dir, d)) and d != "logs"
    )

    for run_id in run_dirs:
        d = os.path.join(staging_dir, run_id)

        header, rows = _read_csv_rows(os.path.join(d, _RUNS_CSV))
        if rows:
            _append_rows(runs_dst, header, rows)
            runs_total += len(rows)

        header, rows = _read_csv_rows(os.path.join(d, _EPISODES_CSV))
        if rows:
            _append_rows(episodes_dst, header, rows)
            episodes_total += len(rows)

    return runs_total, episodes_total


# ===========================================================================
# Subprocess execution (one attempt)
# ===========================================================================

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


def _worker_env():
    """Env vars for a worker subprocess.

    Critical: prevent BLAS/OpenMP from spawning a thread per core in every
    subprocess (4 procs * 16 BLAS threads on an 8-core CPU = thrashing).
    """
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env.setdefault("MUJOCO_GL", "egl")
    return env


def _run_subprocess(cmd, log_path, timeout_sec, env):
    """Run one subprocess, streaming output to a log file."""
    with open(log_path, "w") as log_f:
        log_f.write("$ " + " ".join(cmd) + "\n\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )
        start = time.time()
        try:
            for line in proc.stdout:
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
            log_f.write("\n[INTERRUPTED]\n")
            return "interrupted", -1
    if proc.returncode == 0:
        return "ok", 0
    return "fail", proc.returncode


# ===========================================================================
# Worker entry point (runs in subprocess pool)
# ===========================================================================

def _worker(run, staging_dir, log_dir, run_index, total):
    """Execute one logical run (with retries). Returns a result dict.

    Runs in a child process. Must not touch the parent's manifest;
    the parent updates the manifest based on the returned dict.
    """
    # Suppress SIGINT in workers; the orchestrator handles Ctrl+C and
    # propagates cancellation via executor.shutdown.
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass

    env = _worker_env()
    last_status, last_rc = "fail", -1
    last_log_path = None
    run_id = None
    t0 = time.time()
    attempts = 0

    for attempt in range(run["retries"] + 1):
        attempts += 1
        attempt_run_id = uuid.uuid4().hex[:12]
        run_dir = os.path.join(staging_dir, attempt_run_id)
        os.makedirs(run_dir, exist_ok=True)

        log_name = (f"run{run_index:04d}_attempt{attempt}_"
                    f"{attempt_run_id}.log")
        log_path = os.path.join(log_dir, log_name)
        last_log_path = log_path

        cmd = _build_cmd(run, run_dir, attempt_run_id)
        try:
            status, rc = _run_subprocess(cmd, log_path, run["timeout_sec"], env)
        except Exception as e:
            status, rc = "fail", -1
            try:
                with open(log_path, "a") as f:
                    f.write(f"\n[launcher exception] {e!r}\n")
            except Exception:
                pass

        last_status, last_rc = status, rc
        if status == "ok":
            run_id = attempt_run_id
            break
        # Failed attempt: drop its (possibly partial) staging dir so it
        # never gets merged.
        try:
            shutil.rmtree(run_dir, ignore_errors=True)
        except Exception:
            pass

    return {
        "run_index": run_index,
        "status": last_status,
        "rc": last_rc,
        "run_id": run_id,
        "attempts": attempts,
        "duration_sec": time.time() - t0,
        "log_path": last_log_path,
    }


# ===========================================================================
# Orchestrator
# ===========================================================================

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
    return f"[{i}/{total}] " + "  ".join(parts)


def _filter_pending_for_resume(runs, manifest):
    """Return list of (idx, run) for runs not yet completed."""
    completed_identities = {
        tuple(entry["identity"])
        for entry in manifest["runs"]
        if entry["status"] == "ok"
    }
    pending = []
    for i, run in enumerate(runs):
        if tuple(_run_identity(run)) not in completed_identities:
            pending.append((i, run))
    return pending


def _print_summary(manifest, dt_session):
    counts = {}
    for entry in manifest["runs"]:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    print("\n" + "=" * 72)
    print(f"Session {manifest['session_id']} done in {dt_session:.1f}s "
          f"({dt_session / 60:.1f} min)")
    for k in ("ok", "fail", "timeout", "interrupted", "pending"):
        if counts.get(k):
            print(f"  {k:12s} {counts[k]}")


def _do_finalize(stats_dir, session_id):
    """Merge a (possibly past) session's staging dir into canonical CSVs."""
    staging_dir = os.path.join(stats_dir, "_staging", session_id)
    if not os.path.isdir(staging_dir):
        print(f"No staging dir for session {session_id}: {staging_dir}",
              file=sys.stderr)
        return 1
    print(f"Finalizing session {session_id} -> {stats_dir}")
    runs_n, eps_n = _merge_staging_csvs(staging_dir, stats_dir)
    print(f"  appended {runs_n} run rows, {eps_n} episode rows")
    return 0


def main():
    p = argparse.ArgumentParser(
        description="Batch-run evaluate_dataset.py jobs from a YAML config")
    p.add_argument("--config", help="path to YAML config")
    p.add_argument("--stats_dir", default=_DEFAULT_STATS_DIR,
                   help=f"where to write stats CSVs (default: {_DEFAULT_STATS_DIR})")
    p.add_argument("--jobs", type=int, default=1,
                   help="number of parallel evaluation workers (default: 1)")
    p.add_argument("--dry_run", action="store_true",
                   help="print the expanded commands and exit")
    p.add_argument("--start", type=int, default=0,
                   help="skip the first N expanded runs")
    p.add_argument("--limit", type=int, default=None,
                   help="only run this many runs and stop")
    p.add_argument("--resume_session", type=str, default=None,
                   help="session_id to resume (re-runs only missing runs)")
    p.add_argument("--no_backup", action="store_true",
                   help="skip pre-session backup of canonical CSVs")
    p.add_argument("--no_merge", action="store_true",
                   help="skip the post-session merge step")
    p.add_argument("--finalize", type=str, default=None,
                   help="merge an existing staging dir into canonical CSVs and exit")
    args = p.parse_args()

    # Standalone finalize: no config needed.
    if args.finalize:
        sys.exit(_do_finalize(args.stats_dir, args.finalize))

    if not args.config:
        p.error("--config is required (unless using --finalize)")

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

    if args.dry_run:
        print(f"{total} run(s) planned (dry-run, jobs={args.jobs})")
        for i, run in enumerate(runs, 1):
            run_id = uuid.uuid4().hex[:12]
            cmd = _build_cmd(run, "<staging>", run_id)
            print(_format_run_header(run, i, total))
            print("  $ " + " ".join(cmd))
        return

    # ---- Set up session: staging dir, manifest, backup -----------------
    if args.resume_session:
        session_id = args.resume_session
        staging_dir = os.path.join(args.stats_dir, "_staging", session_id)
        if not os.path.isdir(staging_dir):
            print(f"Cannot resume: no staging dir {staging_dir}",
                  file=sys.stderr)
            sys.exit(1)
        manifest = _load_manifest(staging_dir)
        if manifest is None:
            print(f"Cannot resume: no manifest in {staging_dir}",
                  file=sys.stderr)
            sys.exit(1)
        pending = _filter_pending_for_resume(runs, manifest)
        already_ok = total - len(pending)
        print(f"Resuming session {session_id}: "
              f"{already_ok} already done, {len(pending)} pending")
    else:
        session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        staging_dir = os.path.join(args.stats_dir, "_staging", session_id)
        os.makedirs(staging_dir, exist_ok=True)
        manifest = _init_manifest(staging_dir, session_id, runs)
        if not args.no_backup:
            backups = _backup_canonical_csvs(args.stats_dir, session_id)
            for b in backups:
                print(f"  backup: {b}")
        pending = [(i, run) for i, run in enumerate(runs)]
        print(f"Session {session_id}: {total} run(s) planned, "
              f"jobs={args.jobs}")

    log_dir = os.path.join(staging_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"  staging: {staging_dir}")
    print("-" * 72)

    # ---- Execute -------------------------------------------------------
    t_session = time.time()
    interrupted = False

    try:
        with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            future_to_idx = {
                pool.submit(_worker, run, staging_dir, log_dir, i, total): i
                for i, run in pending
            }
            try:
                done = 0
                for fut in as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    done += 1
                    try:
                        result = fut.result()
                    except Exception as e:
                        result = {
                            "run_index": idx, "status": "fail", "rc": -1,
                            "run_id": None, "attempts": 0, "duration_sec": 0.0,
                            "log_path": None,
                        }
                        print(f"[{done}/{len(pending)}] worker exception "
                              f"on run {idx}: {e!r}")
                    _update_manifest_entry(
                        staging_dir, manifest, idx,
                        status=result["status"],
                        run_id=result["run_id"],
                        attempts=result["attempts"],
                        duration_sec=result["duration_sec"],
                        rc=result["rc"],
                        log_path=result["log_path"],
                    )
                    print(f"[{done}/{len(pending)}] "
                          f"{_format_run_header(runs[idx], idx + 1, total)} "
                          f"-> {result['status']} "
                          f"(rc={result['rc']}, "
                          f"{result['duration_sec']:.1f}s, "
                          f"attempts={result['attempts']})")
            except KeyboardInterrupt:
                interrupted = True
                print("\n[Ctrl+C] cancelling pending work and waiting "
                      "for in-flight workers to terminate...")
                for f in future_to_idx:
                    if not f.done():
                        f.cancel()
                pool.shutdown(wait=True, cancel_futures=True)
    finally:
        if not args.no_merge:
            print("\nMerging staging -> canonical CSVs...")
            try:
                runs_n, eps_n = _merge_staging_csvs(staging_dir, args.stats_dir)
                print(f"  appended {runs_n} run rows, {eps_n} episode rows "
                      f"to {args.stats_dir}")
            except Exception as e:
                print(f"  merge failed: {e!r}", file=sys.stderr)
                print(f"  staging preserved at {staging_dir}; "
                      f"retry with --finalize {session_id}", file=sys.stderr)

    dt_session = time.time() - t_session
    _print_summary(manifest, dt_session)
    if interrupted:
        print("  (interrupted; resume with "
              f"--resume_session {session_id})")


if __name__ == "__main__":
    main()
