"""
Batch-run training scripts across many (model, codebook, dataset_percent,
rollout_rate) combos.

Shares the design of utils/collect_stats.py:
  * Each job runs in an isolated subprocess so a single crash/OOM does not
    kill the whole session.
  * Per-job timeout, retries, and a log file per job under train_runs/logs/.
  * Ctrl+C is caught so the session summary still prints.
  * --dry_run prints the expanded commands and exits.

Each job is one training run. Output lands under
    train_runs/{model}_{codebook|none}_{pct}pct_r{rollout_rate}/<timestamp>/...
(robomimic always appends a timestamp subfolder, so re-running the same
job produces a sibling folder rather than overwriting.)

Dataset percent handling (option A):
  * Given --dataset_percent P in (0, 100], we first sub-select P% of all
    episodes under "data/" in the HDF5 (deterministic, fixed seed), then
    split that subset into train / valid filter keys written to
    "mask/train_{P}pct" / "mask/valid_{P}pct". 100 always maps to the
    original "train" / "valid" keys (no rewriting).
  * The split uses the robomimic default val ratio of 0.1; the picked
    subset is deterministic across all runs so 50% is always the same 50%.

Config format (YAML):

    defaults:
      dataset: /path/to/dataset.hdf5   # required (job-level override allowed)
      rollout_rate: 15                 # required
      rollout_n: 10                    # optional, default 10
      dataset_percent: 100             # optional, default 100
      timeout_sec: null                # null = no timeout
      retries: 0
      val_ratio: 0.1                   # fraction of subset used for validation
      subset_seed: 42                  # seed for the dataset subset draw

    jobs:
      - model: resnet_lstm
      - model: dino_transformer
        dataset_percent: 50
      - model: vq_resnet_lstm
        codebook_size: 512
      - model: vq_resnet_lstm
        codebook_size: 1024
        dataset_percent: 50
      - model: dino_vqvae_transformer
        codebook_size: 1024
        rollout_rate: 25

Usage:
    python utils/train_models.py --config utils/train_sweep.yaml
    python utils/train_models.py --config utils/train_sweep.yaml --dry_run
"""
import argparse
import datetime
import os
import random
import subprocess
import sys
import time
from copy import deepcopy

try:
    import yaml
except ImportError:
    print("PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_UTILS_DIR)
_TRAIN_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "train_scripts")
_TRAIN_RUNS_DIR = os.path.join(_PROJECT_ROOT, "train_runs")

# Models that accept --codebook_size. Others must NOT be given one.
_VQ_MODELS = {"vq_resnet_lstm", "dino_vqvae_transformer"}

# All models the runner knows about (== files in train_scripts/).
_KNOWN_MODELS = {
    "resnet_lstm",
    "resnet_transformer",
    "dino_transformer",
    "vq_resnet_lstm",
    "dino_vqvae_transformer",
}


def _load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "jobs" not in cfg or not isinstance(cfg["jobs"], list):
        raise ValueError(f"Config {path} must have a top-level 'jobs' list")
    return cfg


def _expand_jobs(cfg):
    """Merge defaults into each job and validate."""
    defaults = cfg.get("defaults") or {}
    runs = []
    for job_i, job in enumerate(cfg["jobs"]):
        merged = deepcopy(defaults)
        merged.update(job)

        model = merged.get("model")
        if model not in _KNOWN_MODELS:
            raise ValueError(
                f"Job #{job_i}: unknown model {model!r}. "
                f"Expected one of: {sorted(_KNOWN_MODELS)}")

        dataset = merged.get("dataset")
        if not dataset:
            raise ValueError(f"Job #{job_i}: missing 'dataset' (required)")

        rollout_rate = merged.get("rollout_rate")
        if rollout_rate is None:
            raise ValueError(f"Job #{job_i}: missing 'rollout_rate' (required)")

        dataset_percent = float(merged.get("dataset_percent", 100))
        if not (0 < dataset_percent <= 100):
            raise ValueError(
                f"Job #{job_i}: dataset_percent must be in (0, 100], "
                f"got {dataset_percent}")

        codebook_size = merged.get("codebook_size")
        is_vq = model in _VQ_MODELS
        if is_vq and codebook_size is None:
            raise ValueError(
                f"Job #{job_i}: model {model!r} requires 'codebook_size'")
        if not is_vq and codebook_size is not None:
            raise ValueError(
                f"Job #{job_i}: model {model!r} does not accept 'codebook_size'")

        runs.append({
            "job_index": job_i,
            "model": model,
            "dataset": dataset,
            "dataset_percent": dataset_percent,
            "rollout_rate": int(rollout_rate),
            "rollout_n": int(merged.get("rollout_n", 10)),
            "codebook_size": int(codebook_size) if codebook_size is not None else None,
            "val_ratio": float(merged.get("val_ratio", 0.1)),
            "subset_seed": int(merged.get("subset_seed", 42)),
            "timeout_sec": merged.get("timeout_sec"),
            "retries": int(merged.get("retries", 0)),
            "extra_args": list(merged.get("extra_args") or []),
        })
    return runs


def _pct_tag(pct):
    """50.0 -> '50', 12.5 -> '12_5' (filesystem/key friendly)."""
    if float(pct).is_integer():
        return str(int(pct))
    return str(pct).replace(".", "_")


def _filter_key_names(pct):
    tag = _pct_tag(pct)
    return f"train_{tag}pct", f"valid_{tag}pct"


def _exp_name(run):
    parts = [
        run["model"],
        str(run["codebook_size"]) if run["codebook_size"] is not None else "none",
        f"{_pct_tag(run['dataset_percent'])}pct",
        f"r{run['rollout_rate']}",
    ]
    return "_".join(parts)


def ensure_filter_keys(dataset_path, pct, val_ratio, subset_seed):
    """
    Make sure mask/train_{pct}pct and mask/valid_{pct}pct exist in the HDF5.

    Draw (pct/100) * N episodes from data/ (deterministic via subset_seed),
    then split that subset into train/valid by val_ratio. No-op if keys
    already exist. pct==100 uses the stock 'train'/'valid' keys and is a
    no-op.

    Returns (train_key, valid_key).
    """
    if pct >= 100:
        return "train", "valid"

    train_key, valid_key = _filter_key_names(pct)

    import h5py  # local import so --dry_run works without h5py installed

    with h5py.File(dataset_path, "a") as f:
        mask = f.require_group("mask")
        if train_key in mask and valid_key in mask:
            return train_key, valid_key

        demos = sorted(
            f["data"].keys(),
            key=lambda k: int(k.split("_")[-1]) if k.startswith("demo_") else k,
        )
        n_total = len(demos)
        n_subset = max(1, int(round(n_total * (pct / 100.0))))

        rng = random.Random(subset_seed)
        shuffled = demos[:]
        rng.shuffle(shuffled)
        subset = sorted(
            shuffled[:n_subset],
            key=lambda k: int(k.split("_")[-1]) if k.startswith("demo_") else k,
        )

        # Split subset into train / valid using the same seed so the split
        # is deterministic across re-runs.
        split_rng = random.Random(subset_seed + 1)
        shuffled_subset = subset[:]
        split_rng.shuffle(shuffled_subset)
        n_valid = max(1, int(round(n_subset * val_ratio))) if val_ratio > 0 else 0
        n_valid = min(n_valid, n_subset - 1)  # leave at least one for training
        valid_names = sorted(shuffled_subset[:n_valid])
        train_names = sorted(shuffled_subset[n_valid:])

        def _write(key, names):
            if key in mask:
                del mask[key]
            mask.create_dataset(
                key,
                data=[n.encode("utf-8") for n in names],
                dtype=h5py.special_dtype(vlen=str),
            )

        _write(train_key, train_names)
        _write(valid_key, valid_names)
        print(f"[filter-keys] {dataset_path}: wrote mask/{train_key} "
              f"({len(train_names)}) + mask/{valid_key} ({len(valid_names)}) "
              f"from {n_subset}/{n_total} demos (pct={pct})")

    return train_key, valid_key


def _build_cmd(run, exp_name, train_fk, valid_fk):
    script = os.path.join(_TRAIN_SCRIPTS_DIR, f"{run['model']}.py")
    cmd = [
        sys.executable, script,
        "--dataset", str(run["dataset"]),
        "--output", _TRAIN_RUNS_DIR,
        "--exp_name", exp_name,
        "--rollout_rate", str(run["rollout_rate"]),
        "--rollout_n", str(run["rollout_n"]),
        "--train_filter_key", train_fk,
        "--valid_filter_key", valid_fk,
    ]
    if run["codebook_size"] is not None:
        cmd += ["--codebook_size", str(run["codebook_size"])]
    cmd += list(run["extra_args"])
    return cmd


def _format_run_header(run, exp_name, i, total):
    parts = [
        f"model={run['model']}",
        f"cb={run['codebook_size'] if run['codebook_size'] is not None else 'none'}",
        f"pct={_pct_tag(run['dataset_percent'])}",
        f"r={run['rollout_rate']}",
        f"n={run['rollout_n']}",
        f"exp={exp_name}",
    ]
    return f"[{i}/{total}] " + "  ".join(parts)


def _run_once(cmd, log_path, timeout_sec):
    """Run one subprocess, streaming output to log file AND stdout."""
    with open(log_path, "w") as log_f:
        log_f.write("$ " + " ".join(cmd) + "\n\n")
        log_f.flush()
        env = os.environ.copy()
        env["TQDM_DISABLE"] = "1"
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
        description="Batch-run training scripts from a YAML config")
    p.add_argument("--config", required=True, help="path to YAML config")
    p.add_argument("--output_dir", default=_TRAIN_RUNS_DIR,
                   help=f"where train_runs are written (default: {_TRAIN_RUNS_DIR})")
    p.add_argument("--dry_run", action="store_true",
                   help="print the expanded commands and exit, don't run")
    p.add_argument("--start", type=int, default=0,
                   help="skip the first N expanded runs")
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

    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Session {session_id}: {total} run(s) planned")
    print(f"Train runs dir: {args.output_dir}")
    print(f"Logs dir:       {log_dir}")
    print("-" * 72)

    if args.dry_run:
        for i, run in enumerate(runs, 1):
            exp_name = _exp_name(run)
            train_fk, valid_fk = _filter_key_names(run["dataset_percent"]) \
                if run["dataset_percent"] < 100 else ("train", "valid")
            cmd = _build_cmd(run, exp_name, train_fk, valid_fk)
            print(_format_run_header(run, exp_name, i, total))
            print("  $ " + " ".join(cmd))
        return

    summary = {"ok": 0, "fail": 0, "timeout": 0}
    t_session = time.time()
    interrupted = False

    for i, run in enumerate(runs, 1):
        exp_name = _exp_name(run)
        print("\n" + _format_run_header(run, exp_name, i, total))

        try:
            train_fk, valid_fk = ensure_filter_keys(
                run["dataset"], run["dataset_percent"],
                run["val_ratio"], run["subset_seed"],
            )
        except Exception as e:
            print(f"  -> filter-key prep failed: {e!r}")
            summary["fail"] += 1
            continue

        status = "fail"
        rc = -1
        for attempt in range(run["retries"] + 1):
            log_name = f"{session_id}_run{i:04d}_attempt{attempt}_{exp_name}.log"
            log_path = os.path.join(log_dir, log_name)
            cmd = _build_cmd(run, exp_name, train_fk, valid_fk)
            if attempt > 0:
                print(f"  retry {attempt}/{run['retries']}")
            t0 = time.time()
            try:
                status, rc = _run_once(cmd, log_path, run["timeout_sec"])
            except KeyboardInterrupt:
                interrupted = True
                print("\n[Ctrl+C] stopping session")
                break
            except Exception as e:
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
