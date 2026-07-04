from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.data.datasets import CrossViewTriageDataset
from crossview_conflict.factory import load_triage_from_checkpoint
from crossview_conflict.training.metrics import classification_metrics, classification_predictions
from crossview_conflict.utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CrossViewConflict triage model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json")
    parser.add_argument("--predictions-csv", help="Optional per-sample predictions CSV.")
    return parser.parse_args()


def _class_name(class_names: list[str], index: int) -> str:
    if 0 <= index < len(class_names):
        return class_names[index]
    return str(index)


def main() -> None:
    args = parse_args()
    model, checkpoint = load_triage_from_checkpoint(args.checkpoint, device=args.device)
    config = checkpoint.get("config", {})
    class_names = list(config.get("class_names", []))
    num_classes = int(config.get("num_classes", 1))
    if not class_names:
        class_names = [str(index) for index in range(max(num_classes, 2 if num_classes == 1 else num_classes))]
    dataset = CrossViewTriageDataset(
        args.split_csv,
        street_size=args.image_size,
        overhead_size=args.image_size,
        street_backbone=config.get("street_backbone", "resnet18"),
        overhead_backbone=config.get("overhead_backbone", "resnet18"),
        label_col=config.get("label_col", "auto"),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model.eval()
    logits_list = []
    targets_list = []
    prediction_rows: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            street = batch["street"].to(args.device)
            overhead = batch["overhead"].to(args.device)
            target = batch["target"].to(args.device)
            kwargs = {"street": street, "overhead": overhead}
            if "generated" in batch:
                kwargs["generated"] = batch["generated"].to(args.device)
            logits = model(**kwargs)
            predictions, probabilities = classification_predictions(logits)
            logits_list.append(logits.cpu())
            targets_list.append(target.cpu())
            for row_index, (sample_id, target_value, pred_value) in enumerate(zip(
                batch["sample_id"],
                target.cpu().tolist(),
                predictions.cpu().tolist(),
            )):
                target_index = int(target_value)
                prediction_index = int(pred_value)
                row = {
                    "sample_id": str(sample_id),
                    "target": target_index,
                    "target_name": _class_name(class_names, target_index),
                    "prediction": prediction_index,
                    "prediction_name": _class_name(class_names, prediction_index),
                }
                if logits.ndim == 1 or (logits.ndim == 2 and logits.size(-1) == 1):
                    logit_value = float(logits.cpu().reshape(-1)[row_index].item())
                    probability = float(probabilities.cpu().reshape(-1)[row_index].item())
                    row.update(
                        {
                            "logit": logit_value,
                            "probability": probability,
                            "confidence": probability if prediction_index == 1 else 1.0 - probability,
                        }
                    )
                else:
                    probability_row = probabilities.cpu()[row_index].tolist()
                    logit_row = logits.cpu()[row_index].tolist()
                    row["confidence"] = float(max(probability_row))
                    row["probability"] = row["confidence"]
                    for class_index, logit in enumerate(logit_row):
                        row[f"logit_{class_index}"] = float(logit)
                    for class_index, probability in enumerate(probability_row):
                        row[f"prob_{class_index}"] = float(probability)
                prediction_rows.append(row)

    metrics = classification_metrics(torch.cat(logits_list), torch.cat(targets_list), class_names=class_names)
    metrics["checkpoint_epoch"] = checkpoint.get("epoch")
    print(metrics)
    if args.output_json:
        save_json(metrics, args.output_json)
    if args.predictions_csv:
        predictions_path = Path(args.predictions_csv)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)


if __name__ == "__main__":
    main()
