from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-sample street-view building-visibility features for every "
            "sample in a split CSV. These features feed the reliability gate, so "
            "they are computed for full splits rather than conflict subsets."
        )
    )
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--model-id", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--center-fraction", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


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
        centroid_distance_norm = math.sqrt(
            (centroid_y - center_y) ** 2 + (centroid_x - center_x) ** 2
        ) / max(radius, 1e-6)
    else:
        centroid_distance_norm = 1.0
    return {
        "building_ratio": building_ratio,
        "center_building_ratio": center_ratio,
        "center_minus_global": float(center_ratio - building_ratio),
        "building_present": float(building_ratio > 0.0),
        "centroid_distance_norm": float(centroid_distance_norm),
    }


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.split_csv)
    if "street_view_path" not in df.columns:
        raise KeyError(f"No street_view_path column in {args.split_csv}")
    processor, model, building_id = _prepare_model(args.model_id, args.device)

    rows: list[dict[str, object]] = []
    paths = df["street_view_path"].tolist()
    sample_ids = df["sample_id"].tolist()
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        batch_ids = sample_ids[start : start + args.batch_size]
        images = []
        valid = []
        for index, path in enumerate(batch_paths):
            try:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
                valid.append(index)
            except (FileNotFoundError, OSError) as error:
                rows.append({"sample_id": str(batch_ids[index]), "error": str(error)})
        if not images:
            continue
        inputs = processor(images=images, return_tensors="pt").to(args.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        for image_index, batch_index in enumerate(valid):
            size = images[image_index].size[::-1]
            upsampled = torch.nn.functional.interpolate(
                logits[image_index : image_index + 1], size=size, mode="bilinear", align_corners=False
            )
            mask = (upsampled.argmax(dim=1)[0] == building_id).cpu().numpy()
            row: dict[str, object] = {"sample_id": str(batch_ids[batch_index])}
            row.update(_metrics_from_mask(mask, args.center_fraction))
            rows.append(row)
        if (start // args.batch_size) % 25 == 0:
            print(f"{start + len(batch_paths)}/{len(paths)}", flush=True)

    output_path = Path(args.output_csv)
    ensure_dir(output_path.parent)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"wrote {output_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
