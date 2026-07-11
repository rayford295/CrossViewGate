from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "cvian-sequence-role-isolated-base-v1"
SEEDS = (42, 123, 456, 789, 1011)
ROLES = ("base_fit", "selector_fit", "validation", "prospective_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train five fresh CVIAN cross-view base encoders on the sequence/block-clean "
            "base_fit role. The prospective test is never evaluated by this script."
        )
    )
    parser.add_argument(
        "--protocol-dir",
        default="data/splits/ian_hurricane_sequence_four_role_v1",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/cvian_sequence_active_v2/base_encoders",
    )
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _resolve(path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _validate_protocol(protocol_dir: Path) -> dict[str, Any]:
    summary_path = protocol_dir / "protocol_summary.json"
    commitment_path = protocol_dir / "test_commitment.json"
    summary = _load_json(summary_path)
    commitment = _load_json(commitment_path)
    if summary.get("schema_version") != "cvian-sequence-four-role-summary-v1":
        raise ValueError("Unsupported CVIAN four-role protocol summary")
    if commitment.get("schema_version") != "cvian-sequence-test-commitment-v1":
        raise ValueError("Unsupported CVIAN test commitment")
    status = summary.get("test_status", {})
    if status.get("status") != "selector_selection_holdout_with_historical_base_exposure":
        raise ValueError("Unexpected prospective-test status")
    if status.get("old_base_or_checkpoint_reuse_permitted") is not False or commitment.get(
        "historical_checkpoint_reuse_permitted"
    ) is not False:
        raise ValueError("Protocol must forbid historical checkpoint reuse")
    if status.get("new_protocol_test_scored") is not False:
        raise ValueError("Prospective test is already marked scored")
    for field in (
        "record_overlap_zero",
        "sequence_overlap_zero",
        "spatial_block_overlap_zero",
        "spatial_buffer_clear",
    ):
        if summary.get(field) is not True:
            raise ValueError(f"Protocol audit failed: {field}")
    recorded_hashes = summary.get("role_manifest_sha256", {})
    frames: dict[str, pd.DataFrame] = {}
    for role in ROLES:
        path = protocol_dir / f"{role}.csv"
        if not path.is_file() or recorded_hashes.get(path.name) != _sha256(path):
            raise ValueError(f"Role manifest hash mismatch: {path}")
        frame = pd.read_csv(
            path,
            dtype={"sample_id": str, "sequence_id": str, "spatial_block_id": str},
            usecols=["sample_id", "sequence_id", "spatial_block_id", "label"],
        )
        if frame["sample_id"].duplicated().any() or frame["label"].nunique() != 3:
            raise ValueError(f"Role is duplicate or not class-complete: {role}")
        frames[role] = frame
    for left_index, left in enumerate(ROLES):
        for right in ROLES[left_index + 1 :]:
            for column in ("sample_id", "sequence_id", "spatial_block_id"):
                overlap = set(frames[left][column]) & set(frames[right][column])
                if overlap:
                    raise ValueError(f"{column} overlap between {left} and {right}")
    return {
        "summary": summary,
        "summary_sha256": _sha256(summary_path),
        "commitment": commitment,
        "commitment_sha256": _sha256(commitment_path),
        "role_sha256": {
            role: _sha256(protocol_dir / f"{role}.csv") for role in ROLES
        },
    }


def _fingerprint(
    args: argparse.Namespace,
    *,
    seed: int,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    sources = {
        "orchestrator": REPO_ROOT / "scripts" / "run_cvian_sequence_base_multiseed.py",
        "trainer": REPO_ROOT / "scripts" / "train_triage.py",
        "training_loops": REPO_ROOT / "crossview_conflict" / "training" / "loops.py",
        "triage_model": REPO_ROOT / "crossview_conflict" / "models" / "triage.py",
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "mode": "crossview",
        "train_role": "base_fit",
        "validation_role": "validation",
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "image_size": 224,
        "class_weighting": "balanced",
        "street_augment": True,
        "overhead_augment": True,
        "device": args.device,
        "protocol_summary_sha256": protocol["summary_sha256"],
        "test_commitment_sha256": protocol["commitment_sha256"],
        "role_sha256": protocol["role_sha256"],
        "source_sha256": {name: _sha256(path) for name, path in sources.items()},
        "test_evaluated": False,
    }
    payload["fingerprint_sha256"] = _canonical_hash(payload)
    return payload


def _validate_completion(
    run_dir: Path, completion: dict[str, Any], expected: dict[str, Any]
) -> None:
    if completion.get("fingerprint_sha256") != expected["fingerprint_sha256"]:
        raise ValueError(f"Base run fingerprint mismatch: {run_dir}")
    for filename, field in (
        ("triage_best.pt", "checkpoint_sha256"),
        ("triage_history.json", "history_sha256"),
        ("train.log", "train_log_sha256"),
    ):
        path = run_dir / filename
        if not path.is_file() or _sha256(path) != completion.get(field):
            raise ValueError(f"Base run artifact hash mismatch: {path}")


def _run_seed(
    args: argparse.Namespace,
    *,
    seed: int,
    protocol_dir: Path,
    output_root: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    run_dir = output_root / f"crossview_seed{seed}"
    completion_path = run_dir / "training_complete.json"
    expected = _fingerprint(args, seed=seed, protocol=protocol)
    if completion_path.is_file():
        completion = _load_json(completion_path)
        _validate_completion(run_dir, completion, expected)
        return completion
    if any(
        (run_dir / name).exists()
        for name in ("triage_best.pt", "triage_history.json", "train.log")
    ):
        raise ValueError(
            f"Unattested base artifacts exist in {run_dir}; use a new output root"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_triage.py"),
        "--train-csv",
        str(protocol_dir / "base_fit.csv"),
        "--val-csv",
        str(protocol_dir / "validation.csv"),
        "--output-dir",
        str(run_dir),
        "--mode",
        "crossview",
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
        "--lr",
        str(args.learning_rate),
        "--image-size",
        "224",
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(seed),
        "--device",
        args.device,
    ]
    environment = dict(os.environ)
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    log_path = run_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
        log.flush()
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            env=environment,
        )
    completion = {
        **expected,
        "checkpoint_sha256": _sha256(run_dir / "triage_best.pt"),
        "history_sha256": _sha256(run_dir / "triage_history.json"),
        "train_log_sha256": _sha256(log_path),
    }
    completion_path.write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    return completion


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.patience < 0 or args.batch_size < 1:
        raise ValueError("Training counts must be positive")
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if seeds != SEEDS:
        raise ValueError(f"This protocol fixes the registered seeds to {SEEDS}")
    protocol_dir = _resolve(args.protocol_dir)
    output_root = _resolve(args.output_root)
    protocol = _validate_protocol(protocol_dir)
    runs = []
    for seed in seeds:
        print(f"sequence-role-isolated base seed={seed}", flush=True)
        runs.append(
            _run_seed(
                args,
                seed=seed,
                protocol_dir=protocol_dir,
                output_root=output_root,
                protocol=protocol,
            )
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "training_summary.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "runs": runs}, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
