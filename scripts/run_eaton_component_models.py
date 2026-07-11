#!/usr/bin/env python
"""Train fresh Eaton single-view models and export development predictions only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import PIL
import torch
import torchvision
import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_eaton_component_direction_protocol import load_protocol, sha256_file


PREDICTION_SCHEMA = "eaton-component-direction-predictions-v1"
CLASS_ORDER = (
    "no_or_trace_damage",
    "damaged_repairable",
    "destroyed",
)
PROBABILITY_COLUMNS = {
    "street": tuple(f"street_prob_{name}" for name in CLASS_ORDER),
    "overhead": tuple(f"overhead_prob_{name}" for name in CLASS_ORDER),
}


def executed_code_paths() -> tuple[Path, ...]:
    """Return the repository files transitively executed by train/evaluation."""

    return tuple(
        REPO_ROOT / relative
        for relative in (
            "scripts/run_eaton_component_models.py",
            "scripts/build_eaton_component_direction_protocol.py",
            "scripts/train_triage.py",
            "scripts/eval_triage.py",
            "crossview_conflict/factory.py",
            "crossview_conflict/data/datasets.py",
            "crossview_conflict/models/triage.py",
            "crossview_conflict/models/backbones.py",
            "crossview_conflict/training/loops.py",
            "crossview_conflict/training/metrics.py",
            "crossview_conflict/training/conflict_focal_loss.py",
            "crossview_conflict/utils/image.py",
            "crossview_conflict/utils/io.py",
            "crossview_conflict/utils/seed.py",
        )
    )


def code_hashes() -> dict[str, str]:
    paths = executed_code_paths()
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return {
        str(path.relative_to(REPO_ROOT)).replace("\\", "/"): sha256_file(path)
        for path in paths
    }


def _write_json_lf(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    )


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if logits.ndim != 2 or logits.shape[1] != len(CLASS_ORDER):
        raise ValueError(f"Expected logits with shape (n, {len(CLASS_ORDER)})")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    shifted = logits / temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    probabilities = _softmax(logits, temperature)
    return float(
        -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0)).mean()
    )


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Fit one positive scalar temperature by deterministic golden-section search."""

    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if len(logits) != len(targets) or not len(targets):
        raise ValueError("Calibration logits and targets must be non-empty and aligned")
    if not np.isfinite(logits).all() or not np.isin(targets, np.arange(len(CLASS_ORDER))).all():
        raise ValueError("Calibration inputs are invalid")
    low, high = -4.0, 4.0
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = high - ratio * (high - low)
    x2 = low + ratio * (high - low)
    y1 = _nll(logits, targets, math.exp(x1))
    y2 = _nll(logits, targets, math.exp(x2))
    for _ in range(80):
        if y1 <= y2:
            high, x2, y2 = x2, x1, y1
            x1 = high - ratio * (high - low)
            y1 = _nll(logits, targets, math.exp(x1))
        else:
            low, x1, y1 = x1, x2, y2
            x2 = low + ratio * (high - low)
            y2 = _nll(logits, targets, math.exp(x2))
    temperature = math.exp((low + high) / 2.0)
    return float(temperature)


def _read_eval_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sample_id": str})
    required = {"sample_id", "target", *(f"logit_{index}" for index in range(3))}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Evaluation export {path} is missing columns: {missing}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Evaluation export {path} has duplicate sample_id")
    return frame


def calibrated_probabilities(
    frame: pd.DataFrame, temperature: float, *, view: str
) -> pd.DataFrame:
    if view not in PROBABILITY_COLUMNS:
        raise ValueError(f"Unknown view {view!r}")
    logits = frame[[f"logit_{index}" for index in range(3)]].to_numpy(dtype=np.float64)
    probabilities = _softmax(logits, temperature)
    output = frame[["sample_id", "target"]].copy()
    for index, column in enumerate(PROBABILITY_COLUMNS[view]):
        output[column] = probabilities[:, index]
    output[f"{view}_prediction"] = probabilities.argmax(axis=1).astype(int)
    return output


def build_seed_prediction_export(
    *,
    seed: int,
    street: pd.DataFrame,
    overhead: pd.DataFrame,
    manifest: pd.DataFrame,
    street_temperature: float,
    overhead_temperature: float,
    protocol_role: str = "study_development",
) -> pd.DataFrame:
    if protocol_role != "study_development":
        raise ValueError("This development runner refuses every role except study_development")
    street_calibrated = calibrated_probabilities(
        street, street_temperature, view="street"
    )
    overhead_calibrated = calibrated_probabilities(
        overhead, overhead_temperature, view="overhead"
    )
    merged = street_calibrated.merge(
        overhead_calibrated,
        on=["sample_id", "target"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Street and overhead prediction ids/targets are not identical")
    merged = merged.drop(columns="_merge")
    manifest_ids = manifest["pair_id"].astype(str)
    if manifest_ids.duplicated().any():
        raise ValueError("Role manifest pair_id must be unique")
    prediction_ids = set(merged["sample_id"].astype(str))
    expected_ids = set(manifest_ids)
    if prediction_ids != expected_ids:
        missing = sorted(expected_ids - prediction_ids)[:5]
        extra = sorted(prediction_ids - expected_ids)[:5]
        raise ValueError(f"Prediction/manifest id mismatch; missing={missing}, extra={extra}")
    metadata = manifest[["pair_id", "spatial_block_id", "label"]].copy()
    metadata["pair_id"] = metadata["pair_id"].astype(str)
    result = metadata.merge(
        merged,
        left_on="pair_id",
        right_on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if not result["label"].astype(int).eq(result["target"].astype(int)).all():
        raise ValueError("Evaluation targets do not reproduce the frozen manifest labels")
    result = result.drop(columns=["sample_id", "label"])
    result.insert(1, "seed", int(seed))
    result.insert(3, "protocol_role", protocol_role)
    result.insert(5, "class_order", json.dumps(CLASS_ORDER, separators=(",", ":")))
    result["street_temperature"] = float(street_temperature)
    result["overhead_temperature"] = float(overhead_temperature)
    probability_columns = [*PROBABILITY_COLUMNS["street"], *PROBABILITY_COLUMNS["overhead"]]
    probabilities = result[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise RuntimeError("Calibrated probabilities are invalid")
    if not np.allclose(result[list(PROBABILITY_COLUMNS["street"])].sum(axis=1), 1.0):
        raise RuntimeError("Street probabilities do not sum to one")
    if not np.allclose(result[list(PROBABILITY_COLUMNS["overhead"])].sum(axis=1), 1.0):
        raise RuntimeError("Overhead probabilities do not sum to one")
    return result.sort_values("pair_id").reset_index(drop=True)


def _run(command: Sequence[str], *, cwd: Path) -> None:
    printable = subprocess.list2cmdline(list(command))
    print(f"RUN {printable}", flush=True)
    subprocess.run(list(command), cwd=cwd, check=True)


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain distinct integers")
    return seeds


def _verify_role_hashes(split_dir: Path, summary: Mapping[str, Any]) -> None:
    expected = summary.get("role_manifest_sha256", {})
    for role in ("model_fit", "model_validation", "study_development"):
        path = split_dir / f"{role}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected.get(role):
            raise ValueError(f"Frozen {role} manifest hash changed")


def _verify_protocol_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("schema_version") != "eaton-component-direction-spatial-summary-v1":
        raise ValueError("Spatial protocol summary has the wrong schema")
    if summary.get("analysis_status") != "PROTOCOL_READY_CONFIRMATION_UNTOUCHED":
        raise ValueError("Spatial protocol is not ready with confirmation untouched")
    confirmation = summary.get("confirmation_commitment", {})
    if confirmation.get("status") != "UNSCORED_RESERVED_SAME_EVENT_SPATIAL_CONFIRMATION":
        raise ValueError("Spatial confirmation commitment is not untouched")
    pairwise = summary.get("pairwise_role_audit", [])
    if len(pairwise) != 6:
        raise ValueError("Spatial protocol must contain six pairwise role audits")
    zero_fields = (
        "pair_id_overlap",
        "dependency_group_overlap",
        "spatial_block_overlap",
        "street_media_sha_overlap",
        "remote_media_sha_overlap",
        "overlapping_remote_crop_pairs",
    )
    for row in pairwise:
        if any(int(row.get(field, -1)) != 0 for field in zero_fields):
            raise ValueError("Spatial protocol pairwise overlap audit failed")
        if row.get("strictly_beyond_buffer") is not True:
            raise ValueError("Spatial protocol distance buffer audit failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=REPO_ROOT / "data" / "splits" / "eaton_component_direction_v1",
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=REPO_ROOT / "configs" / "eaton_component_direction_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "eaton_component_direction_v1" / "development_models",
    )
    parser.add_argument("--seeds", help="Comma-separated subset; defaults to frozen seeds")
    parser.add_argument("--epochs", type=int, help="Explicit override, intended only for smoke tests")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    protocol = load_protocol(args.protocol_config)
    summary_path = args.split_dir / "protocol_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _verify_protocol_summary(summary)
    if summary.get("protocol_config_sha256") != sha256_file(args.protocol_config):
        raise ValueError("Protocol config hash does not match the frozen split summary")
    _verify_role_hashes(args.split_dir, summary)
    start_protocol_hash = sha256_file(args.protocol_config)
    start_code_hashes = code_hashes()

    model_config = protocol["models"]
    frozen_seeds = tuple(int(value) for value in model_config["seeds"])
    seeds = _parse_seeds(args.seeds) if args.seeds else frozen_seeds
    if any(seed not in frozen_seeds for seed in seeds):
        raise ValueError("Requested seed is not in the frozen protocol")
    epochs = int(args.epochs if args.epochs is not None else model_config["epochs"])
    batch_size = int(
        args.batch_size if args.batch_size is not None else model_config["batch_size"]
    )
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    is_protocol_run = (
        seeds == frozen_seeds
        and epochs == int(model_config["epochs"])
        and batch_size == int(model_config["batch_size"])
    )
    if not is_protocol_run:
        print("STATUS NON_PROTOCOL_SMOKE_OR_PARTIAL_RUN", flush=True)
    elif args.reuse:
        raise ValueError(
            "A protocol run must train every checkpoint fresh; --reuse is smoke/partial only"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fit_csv = args.split_dir / "model_fit.csv"
    validation_csv = args.split_dir / "model_validation.csv"
    development_csv = args.split_dir / "study_development.csv"
    modes = ("street_only", "remote_only")
    artifacts: dict[str, dict[str, object]] = {}
    temperatures: dict[str, dict[str, float]] = {}
    seed_exports: list[pd.DataFrame] = []
    development_manifest = pd.read_csv(
        development_csv, dtype={"pair_id": str, "spatial_block_id": str}
    )

    for seed in seeds:
        mode_frames: dict[str, dict[str, pd.DataFrame]] = {}
        temperatures[str(seed)] = {}
        for mode in modes:
            run_dir = args.output_dir / "models" / f"{mode}_seed{seed}"
            checkpoint = run_dir / "triage_best.pt"
            validation_predictions = run_dir / "model_validation_predictions.csv"
            development_predictions = run_dir / "study_development_predictions.csv"
            if checkpoint.exists() and (is_protocol_run or not args.reuse):
                raise FileExistsError(
                    f"Refusing to reuse existing checkpoint {checkpoint}"
                )
            if not checkpoint.exists():
                command = [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "train_triage.py"),
                    "--train-csv",
                    str(fit_csv),
                    "--val-csv",
                    str(validation_csv),
                    "--output-dir",
                    str(run_dir),
                    "--mode",
                    mode,
                    "--label-col",
                    "label",
                    "--street-backbone",
                    str(model_config["backbone"]),
                    "--overhead-backbone",
                    str(model_config["backbone"]),
                    "--image-size",
                    str(model_config["image_size"]),
                    "--epochs",
                    str(epochs),
                    "--patience",
                    str(model_config["patience"]),
                    "--batch-size",
                    str(batch_size),
                    "--lr",
                    str(model_config["learning_rate"]),
                    "--num-workers",
                    str(args.num_workers),
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                    "--class-weighting",
                    str(model_config["class_weighting"]),
                ]
                command.append("--street-augment" if mode == "street_only" else "--overhead-augment")
                _run(command, cwd=REPO_ROOT)
            for split_name, split_csv, output_csv in (
                ("model_validation", validation_csv, validation_predictions),
                ("study_development", development_csv, development_predictions),
            ):
                if not output_csv.exists() or not args.reuse:
                    _run(
                        [
                            sys.executable,
                            str(REPO_ROOT / "scripts" / "eval_triage.py"),
                            "--checkpoint",
                            str(checkpoint),
                            "--split-csv",
                            str(split_csv),
                            "--image-size",
                            str(model_config["image_size"]),
                            "--batch-size",
                            str(batch_size),
                            "--num-workers",
                            str(args.num_workers),
                            "--device",
                            args.device,
                            "--output-json",
                            str(run_dir / f"{split_name}_metrics.json"),
                            "--predictions-csv",
                            str(output_csv),
                        ],
                        cwd=REPO_ROOT,
                    )
            validation_frame = _read_eval_predictions(validation_predictions)
            development_frame = _read_eval_predictions(development_predictions)
            logits = validation_frame[[f"logit_{index}" for index in range(3)]].to_numpy()
            targets = validation_frame["target"].to_numpy(dtype=np.int64)
            temperature = fit_temperature(logits, targets)
            view = "street" if mode == "street_only" else "overhead"
            temperatures[str(seed)][view] = temperature
            mode_frames[view] = {
                "validation": validation_frame,
                "development": development_frame,
            }
            key = f"{mode}_seed{seed}"
            artifacts[key] = {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "validation_predictions_sha256": sha256_file(validation_predictions),
                "development_predictions_sha256": sha256_file(development_predictions),
                "temperature": temperature,
            }

        seed_exports.append(
            build_seed_prediction_export(
                seed=seed,
                street=mode_frames["street"]["development"],
                overhead=mode_frames["overhead"]["development"],
                manifest=development_manifest,
                street_temperature=temperatures[str(seed)]["street"],
                overhead_temperature=temperatures[str(seed)]["overhead"],
            )
        )

    export = pd.concat(seed_exports, ignore_index=True).sort_values(["seed", "pair_id"])
    export_path = args.output_dir / "study_development_predictions.csv"
    export.to_csv(export_path, index=False)
    if sha256_file(args.protocol_config) != start_protocol_hash:
        raise RuntimeError("Protocol config changed while the model run was executing")
    _verify_role_hashes(args.split_dir, summary)
    end_code_hashes = code_hashes()
    if end_code_hashes != start_code_hashes:
        changed = sorted(
            key
            for key in set(start_code_hashes) | set(end_code_hashes)
            if start_code_hashes.get(key) != end_code_hashes.get(key)
        )
        raise RuntimeError(f"Executed code changed during the model run: {changed}")
    metadata: dict[str, object] = {
        "schema_version": PREDICTION_SCHEMA,
        "protocol_version": protocol["protocol_version"],
        "status": "PROTOCOL_DEVELOPMENT_RUN" if is_protocol_run else "NON_PROTOCOL_SMOKE_OR_PARTIAL_RUN",
        "protocol_config_sha256": start_protocol_hash,
        "role_manifest_sha256": {
            role: summary["role_manifest_sha256"][role]
            for role in ("model_fit", "model_validation", "study_development")
        },
        "prediction_role": "study_development",
        "spatial_confirmation_read_or_scored": False,
        "class_order": list(CLASS_ORDER),
        "seeds": list(seeds),
        "epochs": epochs,
        "batch_size": batch_size,
        "temperature_scaling_role": "model_validation",
        "ensemble_rule": model_config["ensemble_rule"],
        "temperatures": temperatures,
        "artifacts": artifacts,
        "prediction_csv": str(export_path.resolve()),
        "prediction_csv_sha256": sha256_file(export_path),
        "runner_sha256": start_code_hashes["scripts/run_eaton_component_models.py"],
        "code_sha256": start_code_hashes,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": PIL.__version__,
            "tqdm": tqdm.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device": args.device,
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if str(args.device).startswith("cuda") and torch.cuda.is_available()
                else None
            ),
        },
    }
    _write_json_lf(args.output_dir / "prediction_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
