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
from crossview_conflict.training.metrics import binary_classification_metrics
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


def main() -> None:
    args = parse_args()
    model, checkpoint = load_triage_from_checkpoint(args.checkpoint, device=args.device)
    config = checkpoint.get("config", {})
    dataset = CrossViewTriageDataset(
        args.split_csv,
        street_size=args.image_size,
        overhead_size=args.image_size,
        street_backbone=config.get("street_backbone", "resnet18"),
        overhead_backbone=config.get("overhead_backbone", "resnet18"),
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
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()
            logits_list.append(logits.cpu())
            targets_list.append(target.cpu())
            for sample_id, target_value, logit_value, prob_value, pred_value in zip(
                batch["sample_id"],
                target.cpu().tolist(),
                logits.cpu().tolist(),
                probabilities.cpu().tolist(),
                predictions.cpu().tolist(),
            ):
                prediction_rows.append(
                    {
                        "sample_id": str(sample_id),
                        "target": int(target_value),
                        "logit": float(logit_value),
                        "probability": float(prob_value),
                        "prediction": int(pred_value),
                    }
                )

    metrics = binary_classification_metrics(torch.cat(logits_list), torch.cat(targets_list))
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
