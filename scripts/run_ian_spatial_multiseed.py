from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (42, 123, 456, 789, 1011)
DEFAULT_MODES = ("street_only", "remote_only", "concat", "crossview")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume-safe converged five-seed suite for the repaired Ian spatial split."
    )
    parser.add_argument("--split-dir", default="data/splits/ian_hurricane_original")
    parser.add_argument("--output-root", default="outputs/multiseed_ian_spatial_v1")
    parser.add_argument("--dataset-name", default="ian_original")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent GPU training processes.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--adopt-existing",
        action="store_true",
        help=(
            "Stamp legacy artifacts that lack a run fingerprint. Use only after "
            "independently verifying they were produced from the current split/config."
        ),
    )
    return parser.parse_args()


def _comma_values(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def _validate_spatial_protocol(split_dir: Path) -> dict[str, object]:
    summary_path = split_dir / "split_summary.json"
    audit_path = split_dir / "leakage_audit.json"
    if not summary_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError(
            f"Repaired split requires {summary_path} and {audit_path}; rebuild and audit first."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if summary.get("split_strategy") != "spatial-block":
        raise ValueError("Ian multiseed run requires split_strategy=spatial-block")
    if float(summary.get("spatial_buffer_m", 0.0)) < 25.0:
        raise ValueError("Ian multiseed run requires a recorded >=25 m spatial buffer")
    if audit.get("leakage_found") is not False or audit.get("audit_complete") is not True:
        raise ValueError("Ian split leakage audit is not complete and clean")
    for split in ("train", "val", "test"):
        if not (split_dir / f"{split}.csv").is_file():
            raise FileNotFoundError(split_dir / f"{split}.csv")
    return summary


def _run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
        log.flush()
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_fingerprint(
    split_dir: Path,
    *,
    mode: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "ian-spatial-run-v1",
        "mode": mode,
        "seed": seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "image_size": 224,
        "device": args.device,
        "class_weighting": "balanced",
        "street_augment": True,
        "overhead_augment": True,
        "split_sha256": {
            split: _sha256_file(split_dir / f"{split}.csv")
            for split in ("train", "val", "test")
        },
        "entrypoint_sha256": {
            "train_triage.py": _sha256_file(REPO_ROOT / "scripts" / "train_triage.py"),
            "eval_triage.py": _sha256_file(REPO_ROOT / "scripts" / "eval_triage.py"),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _execute_run(
    *,
    split_dir: Path,
    run_dir: Path,
    mode: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "triage_best.pt"
    history = run_dir / "triage_history.json"
    run_config_path = run_dir / "run_config.json"
    expected_config = _run_fingerprint(
        split_dir, mode=mode, seed=seed, args=args
    )
    reusable_artifacts = checkpoint.is_file() and history.is_file()
    if run_config_path.is_file():
        existing_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing_config.get("fingerprint_sha256") != expected_config["fingerprint_sha256"]:
            raise ValueError(
                f"Run fingerprint mismatch for {run_dir}; use a new output directory "
                "instead of reusing stale artifacts."
            )
    elif reusable_artifacts and not args.adopt_existing:
        raise ValueError(
            f"Existing run lacks a fingerprint: {run_dir}. Re-run with --adopt-existing "
            "only after verifying its split and training parameters."
        )
    run_config_path.write_text(
        json.dumps(expected_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    trained = reusable_artifacts
    if not trained:
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_triage.py"),
            "--train-csv",
            str(split_dir / "train.csv"),
            "--val-csv",
            str(split_dir / "val.csv"),
            "--output-dir",
            str(run_dir),
            "--mode",
            mode,
            "--label-col",
            "label",
            "--class-weighting",
            "balanced",
            "--street-augment",
            "--overhead-augment",
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--image-size",
            "224",
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(seed),
            "--device",
            args.device,
        ]
        _run_logged(command, run_dir / "train.log")

    for split in ("test", "val"):
        prediction_path = run_dir / f"{split}_predictions.csv"
        metrics_path = run_dir / f"{split}_metrics.json"
        if prediction_path.is_file() and metrics_path.is_file():
            continue
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "eval_triage.py"),
            "--checkpoint",
            str(checkpoint),
            "--split-csv",
            str(split_dir / f"{split}.csv"),
            "--output-json",
            str(metrics_path),
            "--predictions-csv",
            str(prediction_path),
            "--batch-size",
            str(args.batch_size),
            "--image-size",
            "224",
            "--num-workers",
            "0",
            "--device",
            args.device,
        ]
        _run_logged(command, run_dir / f"eval_{split}.log")
    metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    return {
        "mode": mode,
        "seed": seed,
        "run_dir": str(run_dir),
        "fingerprint_sha256": expected_config["fingerprint_sha256"],
        "trained_this_run": not trained,
        "test_accuracy": metrics.get("accuracy"),
        "test_macro_f1": metrics.get("macro_f1", metrics.get("f1")),
    }


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    modes = _comma_values(args.modes)
    unknown = sorted(set(modes) - set(DEFAULT_MODES))
    if unknown:
        raise ValueError(f"Unknown modes: {unknown}")
    seeds = [int(value) for value in _comma_values(args.seeds)]
    split_dir = Path(args.split_dir)
    if not split_dir.is_absolute():
        split_dir = REPO_ROOT / split_dir
    split_dir = split_dir.resolve()
    protocol = _validate_spatial_protocol(split_dir)
    output_base = Path(args.output_root)
    if not output_base.is_absolute():
        output_base = REPO_ROOT / output_base
    output_base = output_base.resolve()
    output_root = output_base / args.dataset_name
    tasks = [
        (mode, seed, output_root / f"{mode}_seed{seed}")
        for seed in seeds
        for mode in modes
    ]

    results: list[dict[str, object]] = []
    print_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                _execute_run,
                split_dir=split_dir,
                run_dir=run_dir,
                mode=mode,
                seed=seed,
                args=args,
            ): (mode, seed)
            for mode, seed, run_dir in tasks
        }
        for future in as_completed(futures):
            mode, seed = futures[future]
            result = future.result()
            results.append(result)
            with print_lock:
                print(
                    f"DONE {mode} seed={seed} "
                    f"accuracy={result['test_accuracy']:.4f} f1={result['test_macro_f1']:.4f}",
                    flush=True,
                )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "split_dir": str(split_dir),
        "split_protocol": protocol,
        "jobs": args.jobs,
        "runs": sorted(results, key=lambda row: (int(row["seed"]), str(row["mode"]))),
    }
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "run_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Completed {len(results)} Ian spatial runs.")


if __name__ == "__main__":
    main()
