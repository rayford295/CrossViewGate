from __future__ import annotations

import argparse
import hashlib
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
            "FOV intervention for panoramic street views. Produces two crops per "
            "panorama with identical geometry: 'building' centers the window on "
            "the circular-mean horizontal centroid of the SegFormer building mask; "
            "'random' uses a uniform random horizontal center (reproducible per "
            "sample). Only target alignment differs between variants, so the "
            "downstream comparison isolates the view-regime effect causally."
        )
    )
    parser.add_argument("--split-dir", required=True, help="Directory with train/val/test.csv")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-image-root", default="data/fov_intervention")
    parser.add_argument("--output-split-root", default="data/splits")
    parser.add_argument("--crop-width", type=int, default=256)
    parser.add_argument("--crop-height", type=int, default=256)
    parser.add_argument("--model-id", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def _prepare_model(model_id: str, device: str):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(device)
    model.eval()
    id2label = {int(key): value for key, value in model.config.id2label.items()}
    for index, label in id2label.items():
        if str(label).lower() == "building":
            return processor, model, index
    raise KeyError(f"No 'building' label found in {model_id}.")


def circular_centroid_x(mask: np.ndarray) -> float | None:
    """Horizontal centroid of a panoramic mask using the circular mean."""
    column_mass = mask.sum(axis=0).astype(np.float64)
    total = column_mass.sum()
    if total <= 0:
        return None
    width = mask.shape[1]
    angles = 2.0 * math.pi * np.arange(width) / width
    sin_sum = float((column_mass * np.sin(angles)).sum())
    cos_sum = float((column_mass * np.cos(angles)).sum())
    angle = math.atan2(sin_sum, cos_sum) % (2.0 * math.pi)
    return angle / (2.0 * math.pi) * width


def _random_center_x(sample_id: str, width: int) -> float:
    digest = hashlib.sha256(f"fov_random_{sample_id}".encode()).hexdigest()
    return (int(digest[:12], 16) / float(16**12)) * width


def crop_panorama(image: Image.Image, center_x: float, crop_width: int, crop_height: int) -> Image.Image:
    array = np.asarray(image)
    height, width = array.shape[:2]
    rolled = np.roll(array, shift=int(round(width / 2.0 - center_x)), axis=1)
    x0 = (width - crop_width) // 2
    y0 = max(0, (height - crop_height) // 2)
    return Image.fromarray(rolled[y0 : y0 + crop_height, x0 : x0 + crop_width])


def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir)
    image_root = Path(args.output_image_root) / args.dataset_name
    processor, model, building_id = _prepare_model(args.model_id, args.device)

    frames = {split: pd.read_csv(split_dir / f"{split}.csv") for split in ("train", "val", "test")}
    all_rows = pd.concat(
        [frame.assign(_split=name) for name, frame in frames.items()], ignore_index=True
    )
    log_rows: list[dict[str, object]] = []
    new_paths: dict[str, dict[str, str]] = {"building": {}, "random": {}}

    paths = all_rows["street_view_path"].tolist()
    sample_ids = all_rows["sample_id"].astype(str).tolist()
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        batch_ids = sample_ids[start : start + args.batch_size]
        images: list[Image.Image] = []
        for path in batch_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt").to(args.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        for index, (image, sample_id) in enumerate(zip(images, batch_ids)):
            size = image.size[::-1]
            upsampled = torch.nn.functional.interpolate(
                logits[index : index + 1], size=size, mode="bilinear", align_corners=False
            )
            mask = (upsampled.argmax(dim=1)[0] == building_id).cpu().numpy()
            centroid = circular_centroid_x(mask)
            fallback = centroid is None
            width = image.size[0]
            building_x = width / 2.0 if fallback else centroid
            random_x = _random_center_x(sample_id, width)
            for variant, center_x in (("building", building_x), ("random", random_x)):
                out_dir = image_root / variant
                ensure_dir(out_dir)
                out_path = out_dir / f"{sample_id}.png"
                if not out_path.exists():
                    crop_panorama(image, center_x, args.crop_width, args.crop_height).save(out_path)
                new_paths[variant][sample_id] = str(out_path.resolve())
            log_rows.append(
                {
                    "sample_id": sample_id,
                    "building_center_x": building_x,
                    "random_center_x": random_x,
                    "building_ratio": float(mask.mean()),
                    "no_building_fallback": fallback,
                }
            )
        if (start // args.batch_size) % 25 == 0:
            print(f"{start + len(batch_paths)}/{len(paths)}", flush=True)

    for variant in ("building", "random"):
        out_split_dir = Path(args.output_split_root) / f"{args.dataset_name}_fov_{variant}"
        ensure_dir(out_split_dir)
        for split, frame in frames.items():
            updated = frame.copy()
            updated["street_view_path"] = updated["sample_id"].astype(str).map(new_paths[variant])
            missing = updated["street_view_path"].isna().sum()
            if missing:
                raise ValueError(f"{missing} samples without cropped image in {split}")
            updated.to_csv(out_split_dir / f"{split}.csv", index=False)
        print(f"wrote splits to {out_split_dir}")

    log_path = image_root / "crop_log.csv"
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    fallback_rate = float(pd.DataFrame(log_rows)["no_building_fallback"].mean())
    print(f"wrote {log_path}; no-building fallback rate = {fallback_rate:.4f}")


if __name__ == "__main__":
    main()
