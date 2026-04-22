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
    parser = argparse.ArgumentParser(
        description="Estimate building coverage and alignment on conflict subsets using semantic segmentation."
    )
    parser.add_argument("--wildfire-csv", required=True)
    parser.add_argument("--hurricane-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="nvidia/segformer-b5-finetuned-ade-640-640")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--center-fraction", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _bootstrap_mean_ci(values: np.ndarray, n_resamples: int, seed: int, confidence_level: float = 0.95) -> dict[str, float]:
    mean = float(values.mean()) if len(values) else float("nan")
    if len(values) <= 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for idx in range(n_resamples):
        sample = rng.choice(values, size=len(values), replace=True)
        draws[idx] = sample.mean()
    alpha = 1.0 - confidence_level
    ci_low, ci_high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"mean": mean, "ci_low": float(ci_low), "ci_high": float(ci_high)}


def _prepare_model(model_id: str, device: str):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    building_id = None
    for idx, label in id2label.items():
        if label.lower() == "building":
            building_id = idx
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


def _compute_metrics_from_mask(mask: np.ndarray, center_fraction: float) -> dict[str, float]:
    building = mask.astype(bool)
    height, width = building.shape
    coverage = float(building.mean())

    crop_h = max(1, int(height * center_fraction))
    crop_w = max(1, int(width * center_fraction))
    y0 = max(0, (height - crop_h) // 2)
    x0 = max(0, (width - crop_w) // 2)
    center_mask = building[y0 : y0 + crop_h, x0 : x0 + crop_w]
    center_coverage = float(center_mask.mean())

    if building.any():
        ys, xs = np.nonzero(building)
        centroid_y = ys.mean()
        centroid_x = xs.mean()
        center_y = (height - 1) / 2.0
        center_x = (width - 1) / 2.0
        radius = math.sqrt(center_y**2 + center_x**2)
        centroid_distance = math.sqrt((centroid_y - center_y) ** 2 + (centroid_x - center_x) ** 2)
        centroid_distance_norm = float(centroid_distance / max(radius, 1e-6))
    else:
        centroid_distance_norm = 1.0

    return {
        "building_ratio": coverage,
        "center_building_ratio": center_coverage,
        "center_minus_global": float(center_coverage - coverage),
        "building_present": float(coverage > 0.0),
        "centroid_distance_norm": centroid_distance_norm,
    }


@torch.no_grad()
def _analyze_dataset(
    dataset_name: str,
    csv_path: str | Path,
    processor,
    model,
    building_id: int,
    device: str,
    center_fraction: float,
    batch_size: int,
    bootstrap_resamples: int,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    if "street_view_path" not in df.columns:
        raise KeyError(f"{csv_path} is missing street_view_path.")

    rows: list[dict[str, object]] = []
    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start : start + batch_size].copy()
        paths = batch_df["street_view_path"].astype(str).tolist()
        images = _load_images(paths)
        inputs = processor(images=images, return_tensors="pt").to(device)
        outputs = model(**inputs)
        segmentation = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[image.size[::-1] for image in images],
        )

        for row_dict, seg in zip(batch_df.to_dict(orient="records"), segmentation):
            seg_np = seg.detach().cpu().numpy()
            metrics = _compute_metrics_from_mask(seg_np == building_id, center_fraction=center_fraction)
            rows.append({**row_dict, **metrics})

    result_df = pd.DataFrame(rows)
    result_path = output_dir / f"{dataset_name}_alignment.csv"
    result_df.to_csv(result_path, index=False)

    building_ratio = result_df["building_ratio"].to_numpy(dtype=np.float64)
    center_ratio = result_df["center_building_ratio"].to_numpy(dtype=np.float64)
    center_gain = result_df["center_minus_global"].to_numpy(dtype=np.float64)
    centroid_dist = result_df["centroid_distance_norm"].to_numpy(dtype=np.float64)
    building_present_rate = float(result_df["building_present"].mean())

    summary = {
        "dataset": dataset_name,
        "num_examples": int(len(result_df)),
        "building_ratio_mean": float(building_ratio.mean()),
        "building_ratio_median": float(np.median(building_ratio)),
        "building_ratio_mean_ci": _bootstrap_mean_ci(building_ratio, bootstrap_resamples, seed),
        "center_building_ratio_mean": float(center_ratio.mean()),
        "center_building_ratio_mean_ci": _bootstrap_mean_ci(center_ratio, bootstrap_resamples, seed),
        "center_minus_global_mean": float(center_gain.mean()),
        "center_minus_global_mean_ci": _bootstrap_mean_ci(center_gain, bootstrap_resamples, seed),
        "centroid_distance_norm_mean": float(centroid_dist.mean()),
        "centroid_distance_norm_mean_ci": _bootstrap_mean_ci(centroid_dist, bootstrap_resamples, seed),
        "building_present_rate": building_present_rate,
        "csv": str(result_path),
    }
    return summary


def _save_plot(wildfire_df: pd.DataFrame, hurricane_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].hist(wildfire_df["building_ratio"], bins=12, alpha=0.75, label="Wildfire")
    axes[0].hist(hurricane_df["building_ratio"], bins=12, alpha=0.75, label="Hurricane")
    axes[0].set_title("Building Pixel Ratio")
    axes[0].set_xlabel("ratio")
    axes[0].set_ylabel("count")
    axes[0].legend()

    axes[1].hist(wildfire_df["centroid_distance_norm"], bins=12, alpha=0.75, label="Wildfire")
    axes[1].hist(hurricane_df["centroid_distance_norm"], bins=12, alpha=0.75, label="Hurricane")
    axes[1].set_title("Normalized Building Centroid Distance")
    axes[1].set_xlabel("distance to image center")
    axes[1].set_ylabel("count")
    axes[1].legend()

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    processor, model, building_id = _prepare_model(args.model_id, args.device)

    wildfire_summary = _analyze_dataset(
        "wildfire_conflict",
        args.wildfire_csv,
        processor,
        model,
        building_id,
        args.device,
        args.center_fraction,
        args.batch_size,
        args.bootstrap_resamples,
        args.seed,
        output_dir,
    )
    hurricane_summary = _analyze_dataset(
        "hurricane_conflict",
        args.hurricane_csv,
        processor,
        model,
        building_id,
        args.device,
        args.center_fraction,
        args.batch_size,
        args.bootstrap_resamples,
        args.seed,
        output_dir,
    )

    wildfire_df = pd.read_csv(wildfire_summary["csv"])
    hurricane_df = pd.read_csv(hurricane_summary["csv"])
    plot_path = output_dir / "building_alignment_comparison.png"
    _save_plot(wildfire_df, hurricane_df, plot_path)

    combined = {
        "model_id": args.model_id,
        "building_label_id": building_id,
        "center_fraction": args.center_fraction,
        "wildfire_conflict": wildfire_summary,
        "hurricane_conflict": hurricane_summary,
        "plot": str(plot_path),
        "working_interpretation": {
            "building_ratio_higher_in_wildfire": wildfire_summary["building_ratio_mean"]
            > hurricane_summary["building_ratio_mean"],
            "centroid_distance_lower_in_wildfire": wildfire_summary["centroid_distance_norm_mean"]
            < hurricane_summary["centroid_distance_norm_mean"],
        },
    }
    save_json(combined, output_dir / "alignment_summary.json")
    print(combined)


if __name__ == "__main__":
    main()
