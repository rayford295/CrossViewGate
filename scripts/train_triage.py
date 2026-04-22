from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)

    train_dataset = CrossViewTriageDataset(
        args.train_csv,
        street_size=args.image_size,
        overhead_size=args.image_size,
        include_generated=args.use_generated,
        street_augment=args.street_augment,
        overhead_augment=args.overhead_augment,
    )
    val_dataset = CrossViewTriageDataset(
        args.val_csv,
        street_size=args.image_size,
        overhead_size=args.image_size,
        include_generated=args.use_generated,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    config = {
        "street_backbone": args.street_backbone,
        "overhead_backbone": args.overhead_backbone,
        "embedding_dim": args.embedding_dim,
        "pretrained": not args.no_pretrained,
        "mode": args.mode,
        "use_generated": args.use_generated,
        "street_augment": args.street_augment,
        "overhead_augment": args.overhead_augment,
    }
    model = build_triage(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history: list[dict[str, float]] = []
    best_f1 = -1.0
    best_checkpoint = output_dir / "triage_best.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = train_triage_epoch(model, train_loader, optimizer, args.device)
        metrics = eval_triage(model, val_loader, args.device)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
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
