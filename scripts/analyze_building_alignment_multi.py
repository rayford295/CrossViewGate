from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Building alignment and target-visibility proxy analysis.")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset spec as name=conflict_csv. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="nvidia/segformer-b5-finetuned-ade-640-640")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--center-fraction", type=float, default=0.5)
    parser.add_argument("--visible-center-threshold", type=float, default=0.05)
    parser.add_argument("--visible-centroid-threshold", type=float, default=0.65)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _parse_dataset_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Dataset spec must be name=csv_path, got: {spec}")
        name, path = spec.split("=", 1)
        parsed.append((name.strip(), Path(path.strip())))
    return parsed


def _bootstrap_mean_ci(values: np.ndarray, n_resamples: int, seed: int, confidence_level: float = 0.95) -> dict[str, float]:
    values = values[np.isfinite(values)]
    mean = float(values.mean()) if len(values) else float("nan")
    if len(values) <= 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for idx in range(n_resamples):
        draws[idx] = rng.choice(values, size=len(values), replace=True).mean()
    alpha = 1.0 - confidence_level
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"mean": mean, "ci_low": float(low), "ci_high": float(high)}


def _spearman_with_permutation(x: np.ndarray, y: np.ndarray, permutations: int, seed: int) -> dict[str, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) <= 2 or np.unique(x).size <= 1 or np.unique(y).size <= 1:
        return {"spearman_r": float("nan"), "p_value_two_sided": float("nan"), "n": int(len(x))}
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    observed = float(np.corrcoef(x_rank, y_rank)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for idx in range(permutations):
        null[idx] = np.corrcoef(x_rank, rng.permutation(y_rank))[0, 1]
    return {
        "spearman_r": observed,
        "p_value_two_sided": float((np.abs(null) >= abs(observed)).mean()),
        "n": int(len(x)),
    }


def _prepare_model(model_id: str, device: str):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(device)
    model.eval()
    id2label = {int(key): value for key, value in model.config.id2label.items()}
    building_id = None
    for index, label in id2label.items():
        if str(label).lower() == "building":
            building_id = index
            break
    if building_id is None:
        raise KeyError(f"No 'building' label found in {model_id}.")
    return processor, model, building_id


def _load_images(paths: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def _metrics_from_mask(mask: np.ndarray, center_fraction: float) -> dict[str, float]:
    building = mask.astype(bool)
    height, width = building.shape
    building_ratio = float(building.mean())
    crop_h = max(1, int(height * center_fraction))
    crop_w = max(1, int(width * center_fraction))
    y0 = max(0, (height - crop_h) // 2)
    x0 = max(0, (width - crop_w) // 2)
    center = building[y0 : y0 + crop_h, x0 : x0 + crop_w]
    center_ratio = float(center.mean())
    if building.any():
        ys, xs = np.nonzero(building)
        centroid_y = float(ys.mean())
        centroid_x = float(xs.mean())
        center_y = (height - 1.0) / 2.0
        center_x = (width - 1.0) / 2.0
        radius = math.sqrt(center_y**2 + center_x**2)
        centroid_distance_norm = math.sqrt((centroid_y - center_y) ** 2 + (centroid_x - center_x) ** 2) / max(radius, 1e-6)
    else:
        centroid_distance_norm = 1.0
    return {
        "building_ratio": building_ratio,
        "center_building_ratio": center_ratio,
        "center_minus_global": float(center_ratio - building_ratio),
        "building_present": float(building_ratio > 0.0),
        "centroid_distance_norm": float(centroid_distance_norm),
    }


def _target_column(df: pd.DataFrame) -> str:
    for candidate in ("label", "binary_label", "street_target", "target"):
        if candidate in df.columns:
            return candidate
    raise KeyError("No target column found.")


def _add_gain_columns(df: pd.DataFrame) -> pd.DataFrame:
    target_col = _target_column(df)
    target = df[target_col].to_numpy(dtype=np.int64)
    df["street_correct"] = (df["street_prediction"].to_numpy(dtype=np.int64) == target).astype(int)
    df["remote_correct"] = (df["remote_prediction"].to_numpy(dtype=np.int64) == target).astype(int)
    df["crossview_correct"] = (df["crossview_prediction"].to_numpy(dtype=np.int64) == target).astype(int)
    df["best_named_single_correct"] = df["street_correct"]
    if df["remote_correct"].mean() > df["street_correct"].mean():
        df["best_named_single_correct"] = df["remote_correct"]
    df["oracle_single_correct"] = np.maximum(df["street_correct"], df["remote_correct"])
    df["crossview_gain_over_best_named_single"] = df["crossview_correct"] - df["best_named_single_correct"]
    df["crossview_gain_over_oracle_single"] = df["crossview_correct"] - df["oracle_single_correct"]
    if "concat_prediction" in df.columns:
        df["concat_correct"] = (df["concat_prediction"].to_numpy(dtype=np.int64) == target).astype(int)
        df["crossview_gain_over_concat"] = df["crossview_correct"] - df["concat_correct"]
    return df


@torch.no_grad()
def _analyze_one(
    name: str,
    csv_path: Path,
    output_dir: Path,
    processor,
    model,
    building_id: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.read_csv(csv_path)
    if "street_view_path" not in df.columns:
        raise KeyError(f"{csv_path} is missing street_view_path.")

    rows: list[dict[str, object]] = []
    for start in range(0, len(df), args.batch_size):
        batch = df.iloc[start : start + args.batch_size].copy()
        images = _load_images(batch["street_view_path"].astype(str).tolist())
        inputs = processor(images=images, return_tensors="pt").to(args.device)
        outputs = model(**inputs)
        segmentations = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1] for image in images])
        for record, segmentation in zip(batch.to_dict(orient="records"), segmentations):
            mask = segmentation.detach().cpu().numpy() == building_id
            rows.append({**record, **_metrics_from_mask(mask, args.center_fraction)})

    result = pd.DataFrame(rows)
    result["target_visibility_proxy"] = (
        (result["center_building_ratio"] >= args.visible_center_threshold)
        & (result["centroid_distance_norm"] <= args.visible_centroid_threshold)
    ).astype(int)
    result = _add_gain_columns(result)
    result["dataset"] = name

    csv_out = output_dir / f"{name}_alignment.csv"
    result.to_csv(csv_out, index=False)

    summary: dict[str, object] = {
        "dataset": name,
        "num_examples": int(len(result)),
        "csv": str(csv_out),
        "building_ratio": _bootstrap_mean_ci(result["building_ratio"].to_numpy(dtype=np.float64), args.bootstrap_resamples, args.seed),
        "center_building_ratio": _bootstrap_mean_ci(
            result["center_building_ratio"].to_numpy(dtype=np.float64), args.bootstrap_resamples, args.seed
        ),
        "target_visibility_proxy_rate": _bootstrap_mean_ci(
            result["target_visibility_proxy"].to_numpy(dtype=np.float64), args.bootstrap_resamples, args.seed
        ),
        "centroid_distance_norm": _bootstrap_mean_ci(
            result["centroid_distance_norm"].to_numpy(dtype=np.float64), args.bootstrap_resamples, args.seed
        ),
        "crossview_gain_over_best_named_single": _bootstrap_mean_ci(
            result["crossview_gain_over_best_named_single"].to_numpy(dtype=np.float64), args.bootstrap_resamples, args.seed
        ),
        "correlations": {
            "building_ratio_vs_crossview_correct": _spearman_with_permutation(
                result["building_ratio"].to_numpy(dtype=np.float64),
                result["crossview_correct"].to_numpy(dtype=np.float64),
                args.bootstrap_resamples,
                args.seed,
            ),
            "center_building_ratio_vs_crossview_gain": _spearman_with_permutation(
                result["center_building_ratio"].to_numpy(dtype=np.float64),
                result["crossview_gain_over_best_named_single"].to_numpy(dtype=np.float64),
                args.bootstrap_resamples,
                args.seed + 1,
            ),
            "visibility_proxy_vs_crossview_gain": _spearman_with_permutation(
                result["target_visibility_proxy"].to_numpy(dtype=np.float64),
                result["crossview_gain_over_best_named_single"].to_numpy(dtype=np.float64),
                args.bootstrap_resamples,
                args.seed + 2,
            ),
        },
    }
    return result, summary


def _save_plot(combined: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for dataset, group in combined.groupby("dataset", sort=True):
        axes[0].hist(group["building_ratio"], bins=12, alpha=0.55, label=dataset)
        if group["center_building_ratio"].nunique(dropna=True) <= 1:
            trend = group.agg(
                center_building_ratio=("center_building_ratio", "mean"),
                crossview_gain=("crossview_gain_over_best_named_single", "mean"),
            ).to_frame().T
        else:
            bins = pd.qcut(group["center_building_ratio"], q=min(4, max(1, len(group))), duplicates="drop")
            trend = group.groupby(bins, observed=True).agg(
                center_building_ratio=("center_building_ratio", "mean"),
                crossview_gain=("crossview_gain_over_best_named_single", "mean"),
            )
        axes[1].plot(trend["center_building_ratio"], trend["crossview_gain"], marker="o", label=dataset)
    axes[0].set_xlabel("building pixel ratio")
    axes[0].set_ylabel("count")
    axes[0].set_title("Street-view building coverage")
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("center building ratio")
    axes[1].set_ylabel("crossview gain over best single")
    axes[1].set_title("Alignment proxy vs conflict gain")
    axes[1].legend(frameon=False)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    processor, model, building_id = _prepare_model(args.model_id, args.device)
    frames: list[pd.DataFrame] = []
    summaries: dict[str, object] = {}
    for name, path in _parse_dataset_specs(args.dataset):
        frame, summary = _analyze_one(name, path, output_dir, processor, model, building_id, args)
        frames.append(frame)
        summaries[name] = summary
    combined = pd.concat(frames, ignore_index=True)
    combined_csv = output_dir / "building_alignment_combined.csv"
    combined.to_csv(combined_csv, index=False)
    plot_path = output_dir / "building_alignment_multi.png"
    _save_plot(combined, plot_path)
    summary = {
        "model_id": args.model_id,
        "building_label_id": int(building_id),
        "center_fraction": args.center_fraction,
        "visible_center_threshold": args.visible_center_threshold,
        "visible_centroid_threshold": args.visible_centroid_threshold,
        "combined_csv": str(combined_csv),
        "plot": str(plot_path),
        "datasets": summaries,
    }
    save_json(summary, output_dir / "alignment_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
