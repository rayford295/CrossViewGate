from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a prioritized backbone-ablation run plan."
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--wildfire-train-csv", required=True)
    parser.add_argument("--wildfire-val-csv", required=True)
    parser.add_argument("--hurricane-train-csv", required=True)
    parser.add_argument("--hurricane-val-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size-resnet50", type=int, default=24)
    parser.add_argument("--batch-size-clip", type=int, default=16)
    parser.add_argument("--batch-size-dino", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    run_specs = [
        ("wildfire", "crossview", "resnet50", args.batch_size_resnet50, 1),
        ("wildfire", "crossview", "clip_vit_b_32", args.batch_size_clip, 2),
        ("wildfire", "crossview", "dinov2_vits14", args.batch_size_dino, 3),
        ("hurricane", "crossview", "resnet50", args.batch_size_resnet50, 4),
        ("hurricane", "crossview", "clip_vit_b_32", args.batch_size_clip, 5),
        ("hurricane", "crossview", "dinov2_vits14", args.batch_size_dino, 6),
    ]
    split_map = {
        "wildfire": (args.wildfire_train_csv, args.wildfire_val_csv),
        "hurricane": (args.hurricane_train_csv, args.hurricane_val_csv),
    }

    for dataset, mode, backbone, batch_size, priority in run_specs:
        train_csv, val_csv = split_map[dataset]
        output_dir = Path(args.output_root) / dataset / "backbone_ablation" / backbone
        command = [
            "python",
            "scripts/train_triage.py",
            "--train-csv",
            train_csv,
            "--val-csv",
            val_csv,
            "--output-dir",
            str(output_dir),
            "--mode",
            mode,
            "--street-backbone",
            backbone,
            "--overhead-backbone",
            backbone,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(batch_size),
            "--device",
            args.device,
            "--street-augment",
            "--overhead-augment",
        ]
        rows.append(
            {
                "priority": priority,
                "dataset": dataset,
                "mode": mode,
                "backbone": backbone,
                "batch_size": batch_size,
                "train_csv": train_csv,
                "val_csv": val_csv,
                "output_dir": str(output_dir),
                "command": " ".join(f'"{part}"' if " " in part else part for part in command),
            }
        )

    df = pd.DataFrame(rows).sort_values("priority").reset_index(drop=True)
    output_csv = Path(args.output_csv)
    ensure_dir(output_csv.parent)
    df.to_csv(output_csv, index=False)
    print(f"Saved backbone ablation plan to {output_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
