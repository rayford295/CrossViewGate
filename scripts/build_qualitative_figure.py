from __future__ import annotations

import argparse
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
        description="Build a qualitative figure for CrossViewConflict conflict examples."
    )
    parser.add_argument("--wildfire-csv", required=True)
    parser.add_argument("--hurricane-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="nvidia/segformer-b5-finetuned-ade-640-640")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _prepare_model(model_id: str, device: str):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    building_id = next(idx for idx, label in id2label.items() if label.lower() == "building")
    return processor, model, building_id


def _load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _overlay_building(image: Image.Image, mask: np.ndarray) -> np.ndarray:
    base = np.asarray(image).astype(np.float32) / 255.0
    overlay = base.copy()
    overlay[mask] = 0.6 * overlay[mask] + 0.4 * np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return overlay


@torch.no_grad()
def _segment_building(image: Image.Image, processor, model, building_id: int, device: str) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    seg = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    return (seg.detach().cpu().numpy() == building_id)


def _pick_examples(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    target = df["binary_label"].to_numpy()
    df = df.copy()
    df["street_correct"] = (df["street_prediction"].to_numpy() == target).astype(int)
    df["remote_correct"] = (df["remote_prediction"].to_numpy() == target).astype(int)
    df["crossview_correct"] = (df["crossview_prediction"].to_numpy() == target).astype(int)
    df["crossview_margin"] = np.abs(df["crossview_probability"] - 0.5)
    success = df[df["crossview_correct"] == 1].sort_values("crossview_margin", ascending=False)
    failure = df[df["crossview_correct"] == 0].sort_values("crossview_margin", ascending=False)
    if dataset == "wildfire":
        chosen = pd.concat([success.head(2), failure.head(1)], ignore_index=True)
    else:
        chosen = pd.concat([success.head(2)], ignore_index=True)
    chosen["dataset"] = dataset
    return chosen


def _dataset_color(name: str) -> str:
    return "#c87b2a" if name == "wildfire" else "#2d6f95"


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    processor, model, building_id = _prepare_model(args.model_id, args.device)

    wildfire = _pick_examples(pd.read_csv(args.wildfire_csv), "wildfire")
    hurricane = _pick_examples(pd.read_csv(args.hurricane_csv), "hurricane")
    selected = pd.concat([wildfire, hurricane], ignore_index=True)

    n_rows = len(selected)
    fig, axes = plt.subplots(
        n_rows,
        4,
        figsize=(16.5, 3.9 * n_rows),
        gridspec_kw={"width_ratios": [1.05, 1.0, 1.05, 0.95]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#f8f6f0")
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    records: list[dict[str, object]] = []
    for row_idx, row in enumerate(selected.to_dict(orient="records")):
        street_image = _load_image(row["street_view_path"])
        overhead_image = _load_image(row["remote_sensing_path"])
        building_mask = _segment_building(street_image, processor, model, building_id, args.device)
        overlay = _overlay_building(street_image, building_mask)

        dataset = row["dataset"]
        color = _dataset_color(dataset)

        axes[row_idx, 0].imshow(street_image)
        axes[row_idx, 0].set_title(f"{dataset} street", fontsize=10, pad=6)
        axes[row_idx, 1].imshow(overhead_image)
        axes[row_idx, 1].set_title("overhead", fontsize=10, pad=6)
        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title("building overlay", fontsize=10, pad=6)
        axes[row_idx, 3].set_facecolor("#f3efe5")
        for spine in axes[row_idx, 3].spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_edgecolor(color)
        axes[row_idx, 3].set_xticks([])
        axes[row_idx, 3].set_yticks([])
        axes[row_idx, 3].text(
            0.05,
            0.96,
            "\n".join(
                [
                    f"dataset: {dataset}",
                    f"GT: {row['binary_label']}",
                    f"street: {row['street_prediction']} ({row['street_probability']:.2f})",
                    f"remote: {row['remote_prediction']} ({row['remote_probability']:.2f})",
                    f"crossview: {row['crossview_prediction']} ({row['crossview_probability']:.2f})",
                    f"sample_id: {row['sample_id']}",
                ]
            ),
            va="top",
            family="monospace",
            fontsize=10.5,
        )
        for col in range(3):
            axes[row_idx, col].axis("off")
            for spine in axes[row_idx, col].spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_edgecolor("#d9d1bf")

        records.append(
            {
                "dataset": row["dataset"],
                "sample_id": row["sample_id"],
                "binary_label": int(row["binary_label"]),
                "street_prediction": int(row["street_prediction"]),
                "remote_prediction": int(row["remote_prediction"]),
                "crossview_prediction": int(row["crossview_prediction"]),
            }
        )

    fig.suptitle(
        "Qualitative Conflict Cases Across Wildfire and Hurricane Regimes",
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )
    figure_path = output_dir / "qualitative_conflict_examples.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    save_json({"selected_examples": records, "figure": str(figure_path)}, output_dir / "qualitative_conflict_examples.json")
    print({"selected_examples": records, "figure": str(figure_path)})


if __name__ == "__main__":
    main()
