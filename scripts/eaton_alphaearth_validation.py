#!/usr/bin/env python
"""External AlphaEarth validation on the Eaton wildfire split protocol.

The experiment keeps the existing Eaton roles intact:

- fit on ``model_fit``
- choose/report development performance on ``model_validation``
- apply the same fitted model to the untouched ``spatial_confirmation`` role

Feature extraction samples Google Earth Engine's AlphaEarth Foundations annual
embedding at each structure location. For Eaton 2025, the default pre-event
embedding year is 2024.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASET_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
EMBEDDING_BANDS = [f"A{index:02d}" for index in range(64)]
CLASS_ORDER = ("no_or_trace_damage", "damaged_repairable", "destroyed")
ROLE_NAMES = ("model_fit", "model_validation", "study_development", "spatial_confirmation")


def _chunked(frame: pd.DataFrame, size: int) -> Iterable[pd.DataFrame]:
    for start in range(0, len(frame), size):
        yield frame.iloc[start : start + size]


def _read_role(split_dir: Path, role: str) -> pd.DataFrame:
    path = split_dir / f"{role}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(
        path,
        dtype={
            "sample_id": str,
            "pair_id": str,
            "spatial_block_id": str,
            "dependency_group_id": str,
            "split_dependency_group_id": str,
        },
    )


def build_locations(split_dir: Path) -> pd.DataFrame:
    rows = []
    for role in ROLE_NAMES:
        frame = _read_role(split_dir, role)
        required = {"sample_id", "latitude", "longitude", "label", "label_name"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{role}.csv is missing columns: {missing}")
        frame = frame.copy()
        frame["protocol_role"] = role
        rows.append(frame)
    all_roles = pd.concat(rows, ignore_index=True)
    if all_roles["sample_id"].duplicated().any():
        dupes = all_roles.loc[all_roles["sample_id"].duplicated(), "sample_id"].head().tolist()
        raise ValueError(f"Sample IDs are not unique across roles: {dupes}")
    locations = all_roles[
        [
            "sample_id",
            "protocol_role",
            "latitude",
            "longitude",
            "label",
            "label_name",
            "category",
            "spatial_block_id",
            "dependency_group_id",
            "split_dependency_group_id",
        ]
    ].rename(columns={"latitude": "lat", "longitude": "lon"})
    locations["lat"] = pd.to_numeric(locations["lat"], errors="coerce")
    locations["lon"] = pd.to_numeric(locations["lon"], errors="coerce")
    if locations[["lat", "lon"]].isna().any().any():
        raise ValueError("At least one Eaton row has missing latitude/longitude")
    return locations.sort_values("sample_id").reset_index(drop=True)


def extract_alphaearth(
    *,
    locations: pd.DataFrame,
    out_csv: Path,
    metadata_out: Path,
    year: int,
    ee_project: str | None,
    chunk_size: int,
    neighborhood_m: int,
) -> dict[str, Any]:
    import ee

    ee_project = ee_project or os.environ.get("EE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if ee_project:
        ee.Initialize(project=ee_project)
    else:
        raise RuntimeError(
            "Earth Engine credentials exist, but no Google Cloud project was provided. "
            "Pass --ee-project <project-id> or set EE_PROJECT / GOOGLE_CLOUD_PROJECT."
        )

    image = (
        ee.ImageCollection(DATASET_ID)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .mosaic()
        .select(EMBEDDING_BANDS)
    )

    rows: list[dict[str, Any]] = []
    total = len(locations)
    for index, chunk in enumerate(_chunked(locations, chunk_size), start=1):
        records = chunk[["sample_id", "lon", "lat"]].to_dict("records")
        points = ee.FeatureCollection(
            [
                ee.Feature(
                    ee.Geometry.Point([record["lon"], record["lat"]]),
                    {"sample_id": record["sample_id"]},
                )
                for record in records
            ]
        )
        point_values = image.reduceRegions(
            collection=points, reducer=ee.Reducer.first(), scale=10
        ).getInfo()["features"]
        buffered = points.map(lambda feature: feature.setGeometry(feature.geometry().buffer(neighborhood_m)))
        neighborhood_values = image.reduceRegions(
            collection=buffered, reducer=ee.Reducer.mean(), scale=10
        ).getInfo()["features"]
        neighborhood_by_id = {
            feature["properties"].get("sample_id"): feature["properties"]
            for feature in neighborhood_values
        }
        for feature in point_values:
            props = feature["properties"]
            sample_id = props.get("sample_id")
            row: dict[str, Any] = {"sample_id": sample_id}
            for band_index, band in enumerate(EMBEDDING_BANDS):
                row[f"ae{year}_e{band_index:02d}"] = props.get(band)
            nbhd = neighborhood_by_id.get(sample_id, {})
            for band_index, band in enumerate(EMBEDDING_BANDS):
                row[f"ae{year}_nbhd{neighborhood_m}m_e{band_index:02d}"] = nbhd.get(band)
            rows.append(row)
        print(f"Extracted AlphaEarth chunk {index}: {min(index * chunk_size, total)}/{total}", flush=True)

    features = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_csv, index=False)
    profile = {
        "dataset": DATASET_ID,
        "model": "AlphaEarth Foundations",
        "event": "Eaton Fire 2025",
        "embedding_year": int(year),
        "leakage_note": "Eaton Fire occurred in 2025; 2024 annual embedding is pre-event.",
        "point_scale_m": 10,
        "neighborhood_mean_m": int(neighborhood_m),
        "n_samples": int(len(features)),
        "n_missing_point_values": int(features[f"ae{year}_e00"].isna().sum()),
        "n_feature_columns": int(len(features.columns) - 1),
        "extracted_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output_csv": str(out_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def _classification_models(seed: int) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logistic_balanced": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
        ),
        "random_forest_balanced": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

    labels = np.arange(len(CLASS_ORDER), dtype=int)
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        **{f"f1_{name}": float(per_class[index]) for index, name in enumerate(CLASS_ORDER)},
        "confusion_matrix": json.dumps(confusion_matrix(y_true, y_pred, labels=labels).tolist(), separators=(",", ":")),
    }


def _feature_columns(features: pd.DataFrame, year: int) -> list[str]:
    columns = [column for column in features.columns if column.startswith(f"ae{year}_")]
    if not columns:
        raise ValueError(f"No AlphaEarth columns found for year {year}")
    return columns


def run_validation(
    *,
    split_dir: Path,
    features_csv: Path,
    output_dir: Path,
    year: int,
    seed: int,
) -> dict[str, Any]:
    locations = build_locations(split_dir)
    features = pd.read_csv(features_csv, dtype={"sample_id": str})
    feature_columns = _feature_columns(features, year)
    data = locations.merge(features[["sample_id", *feature_columns]], on="sample_id", how="left", validate="one_to_one")
    missing_rows = int(data[feature_columns].isna().all(axis=1).sum())
    if missing_rows:
        print(f"Warning: {missing_rows} rows have no AlphaEarth point features", flush=True)

    role_frames = {
        role: data[data["protocol_role"].eq(role)].copy().reset_index(drop=True)
        for role in ROLE_NAMES
    }
    train = role_frames["model_fit"]
    X_train = train[feature_columns].apply(pd.to_numeric, errors="coerce")
    y_train = train["label"].astype(int).to_numpy()

    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for model_name, model in _classification_models(seed).items():
        model.fit(X_train, y_train)
        for role in ("model_fit", "model_validation", "study_development", "spatial_confirmation"):
            frame = role_frames[role]
            X = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
            y_true = frame["label"].astype(int).to_numpy()
            y_pred = model.predict(X)
            probs = model.predict_proba(X)
            row = {
                "feature_set": "alphaearth_overhead",
                "model": model_name,
                "fit_role": "model_fit",
                "eval_role": role,
                "n_samples": int(len(frame)),
                "n_spatial_blocks": int(frame["spatial_block_id"].nunique()),
                "missing_feature_rows": int(X.isna().all(axis=1).sum()),
            }
            row.update(_metrics(y_true, y_pred))
            metrics_rows.append(row)

            preds = frame[
                [
                    "sample_id",
                    "protocol_role",
                    "label",
                    "label_name",
                    "category",
                    "spatial_block_id",
                    "dependency_group_id",
                    "split_dependency_group_id",
                ]
            ].copy()
            preds.insert(1, "feature_set", "alphaearth_overhead")
            preds.insert(2, "model", model_name)
            preds["prediction"] = y_pred.astype(int)
            preds["prediction_name"] = [CLASS_ORDER[int(value)] for value in y_pred]
            for index, class_name in enumerate(CLASS_ORDER):
                if index < probs.shape[1]:
                    preds[f"prob_{class_name}"] = probs[:, index]
                else:
                    preds[f"prob_{class_name}"] = 0.0
            prediction_rows.append(preds)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics_path = output_dir / "metrics.csv"
    predictions_path = output_dir / "predictions.csv"
    profile_path = output_dir / "run_profile.json"
    summary_path = output_dir / "summary.md"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    summary = build_summary(metrics, features_csv, year, seed, missing_rows)
    summary_path.write_text(summary, encoding="utf-8")
    payload = {
        "schema_version": "eaton-alphaearth-validation-v1",
        "feature_set": "alphaearth_overhead",
        "split_dir": str(split_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "features_csv": str(features_csv.relative_to(REPO_ROOT)).replace("\\", "/") if features_csv.is_relative_to(REPO_ROOT) else str(features_csv),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "embedding_year": int(year),
        "seed": int(seed),
        "n_samples": int(len(data)),
        "n_feature_columns": int(len(feature_columns)),
        "missing_rows": int(missing_rows),
        "files": {
            "metrics": str(metrics_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "predictions": str(predictions_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "summary": str(summary_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        },
    }
    profile_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_summary(metrics: pd.DataFrame, features_csv: Path, year: int, seed: int, missing_rows: int) -> str:
    lines = [
        "# Eaton AlphaEarth External Validation",
        "",
        f"Features: `{features_csv}`",
        f"Embedding year: {year} (pre-event for Eaton 2025)",
        f"Seed: {seed}",
        f"Rows with all AlphaEarth features missing: {missing_rows}",
        "",
        "Trained on `model_fit`; evaluated on all frozen Eaton roles.",
        "",
        "| Model | Eval role | n | Macro-F1 | Balanced acc | Accuracy | F1 no/trace | F1 repairable | F1 destroyed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.sort_values(["model", "eval_role"]).to_dict("records"):
        lines.append(
            "| {model} | {eval_role} | {n_samples} | {macro_f1:.4f} | {balanced_accuracy:.4f} | "
            "{accuracy:.4f} | {f1_no_or_trace_damage:.4f} | {f1_damaged_repairable:.4f} | "
            "{f1_destroyed:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Reading guide:",
            "",
            "- `model_validation` is the development validation role.",
            "- `spatial_confirmation` is the reserved same-event spatial confirmation role.",
            "- Eaton class imbalance is severe in the middle class; macro-F1 is the primary score.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=REPO_ROOT / "data" / "splits" / "eaton_component_direction_v1",
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=REPO_ROOT / "data" / "features" / "eaton_alphaearth_2024.csv",
    )
    parser.add_argument(
        "--locations-out",
        type=Path,
        default=REPO_ROOT / "data" / "features" / "eaton_alphaearth_locations.csv",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=REPO_ROOT / "metadata" / "eaton_alphaearth_2024_profile.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "experiments" / "eaton_alphaearth_2024",
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--ee-project")
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--neighborhood-m", type=int, default=30)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--extract", action="store_true", help="Extract AlphaEarth features with Earth Engine before running.")
    parser.add_argument("--locations-only", action="store_true", help="Only write the Eaton location table for external extraction.")
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    locations = build_locations(args.split_dir)
    args.locations_out.parent.mkdir(parents=True, exist_ok=True)
    locations.to_csv(args.locations_out, index=False)
    if args.locations_only:
        print(f"Wrote {len(locations)} locations to {args.locations_out}")
        return
    if args.extract:
        if args.features_csv.exists() and not args.force_extract:
            print(f"Using existing AlphaEarth features: {args.features_csv}")
        else:
            extract_alphaearth(
                locations=locations,
                out_csv=args.features_csv,
                metadata_out=args.metadata_out,
                year=args.year,
                ee_project=args.ee_project,
                chunk_size=args.chunk_size,
                neighborhood_m=args.neighborhood_m,
            )
    if not args.features_csv.is_file():
        raise FileNotFoundError(
            f"{args.features_csv} not found. Run with --extract after Earth Engine auth, "
            "or provide --features-csv."
        )
    payload = run_validation(
        split_dir=args.split_dir,
        features_csv=args.features_csv,
        output_dir=args.output_dir,
        year=args.year,
        seed=args.seed,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
