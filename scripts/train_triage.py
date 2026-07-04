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
from crossview_conflict.factory import build_triage
from crossview_conflict.training.loops import eval_triage, train_triage_epoch
from crossview_conflict.utils.io import ensure_dir, save_checkpoint, save_json
from crossview_conflict.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CrossViewConflict rapid triage model.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--street-backbone", default="resnet18")
    parser.add_argument("--overhead-backbone", default="resnet18")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--mode", default="crossview", choices=["crossview", "street_only", "remote_only"])
    parser.add_argument("--label-col", default="auto", help="Target column. Defaults to 'label' when present, else 'binary_label'.")
    parser.add_argument("--use-generated", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--street-augment", action="store_true")
    parser.add_argument("--overhead-augment", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--conflict-gamma", type=float, default=None)
    parser.add_argument(
        "--class-weighting",
        choices=["none", "balanced"],
        default="none",
        help="Optional class weighting for imbalanced original-class experiments.",
    )
    return parser.parse_args()


def _resolve_label_col(df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        if requested not in df.columns:
            raise KeyError(f"Column '{requested}' not found in training CSV.")
        return requested
    for candidate in ("label", "binary_label"):
        if candidate in df.columns:
            return candidate
    raise KeyError("Training CSV must contain either 'label' or 'binary_label'.")


def _infer_class_names(df: pd.DataFrame, label_col: str) -> list[str]:
    num_classes = int(df[label_col].max()) + 1
    class_names = [str(index) for index in range(num_classes)]
    if label_col == "binary_label":
        name_columns = ("binary_name", "label_name", "category")
    else:
        name_columns = ("label_name", "category", "binary_name")
    for name_col in name_columns:
        if name_col not in df.columns:
            continue
        for label_value, group in df.groupby(label_col, sort=True):
            label_index = int(label_value)
            if 0 <= label_index < num_classes:
                class_names[label_index] = str(group[name_col].dropna().astype(str).iloc[0])
        return class_names
    return class_names


def _balanced_class_weights(df: pd.DataFrame, label_col: str, num_classes: int) -> torch.Tensor:
    counts = df[label_col].value_counts().to_dict()
    total = float(sum(int(value) for value in counts.values()))
    weights = []
    for class_index in range(num_classes):
        count = float(counts.get(class_index, 0))
        weights.append(total / max(count * num_classes, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    train_df = pd.read_csv(args.train_csv)
    label_col = _resolve_label_col(train_df, args.label_col)
    class_names = _infer_class_names(train_df, label_col)
    num_classes = len(class_names)

    train_dataset = CrossViewTriageDataset(
        args.train_csv,
        street_size=args.image_size,
        overhead_size=args.image_size,
        include_generated=args.use_generated,
        street_augment=args.street_augment,
        overhead_augment=args.overhead_augment,
        street_backbone=args.street_backbone,
        overhead_backbone=args.overhead_backbone,
        label_col=label_col,
    )
    val_dataset = CrossViewTriageDataset(
        args.val_csv,
        street_size=args.image_size,
        overhead_size=args.image_size,
        include_generated=args.use_generated,
        street_backbone=args.street_backbone,
        overhead_backbone=args.overhead_backbone,
        label_col=label_col,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    config = {
        "street_backbone": args.street_backbone,
        "overhead_backbone": args.overhead_backbone,
        "embedding_dim": args.embedding_dim,
        "pretrained": not args.no_pretrained,
        "mode": args.mode,
        "label_col": label_col,
        "num_classes": num_classes,
        "class_names": class_names,
        "use_generated": args.use_generated,
        "street_augment": args.street_augment,
        "overhead_augment": args.overhead_augment,
        "conflict_gamma": args.conflict_gamma,
        "class_weighting": args.class_weighting,
    }
    model = build_triage(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    class_weights = None
    if args.class_weighting == "balanced":
        class_weights = _balanced_class_weights(train_df, label_col, num_classes).to(args.device)
        config["class_weights"] = [float(value) for value in class_weights.cpu().tolist()]

    history: list[dict[str, object]] = []
    best_f1 = -1.0
    best_checkpoint = output_dir / "triage_best.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = train_triage_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            conflict_gamma=args.conflict_gamma,
            class_weights=class_weights,
        )
        metrics = eval_triage(model, val_loader, args.device, class_names=class_names)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        if float(metrics["f1"]) > best_f1:
            best_f1 = float(metrics["f1"])
            save_checkpoint(
                best_checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                config=config,
            )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}"
        )

    save_json({"history": history}, output_dir / "triage_history.json")
    print(f"Best triage checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
