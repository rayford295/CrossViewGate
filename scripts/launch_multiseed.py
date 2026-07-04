from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or execute a multi-seed CrossViewConflict training plan."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--modes", nargs="+", default=["crossview", "street_only", "remote_only"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--street-backbone", default="resnet18")
    parser.add_argument("--overhead-backbone", default="resnet18")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--label-col", default="auto")
    parser.add_argument("--class-weighting", choices=["none", "balanced"], default="none")
    parser.add_argument("--street-augment", action="store_true")
    parser.add_argument("--overhead-augment", action="store_true")
    parser.add_argument("--execute-index", type=int)
    return parser.parse_args()


def _build_command(args: argparse.Namespace, mode: str, seed: int) -> list[str]:
    output_dir = Path(args.output_root) / args.dataset_name / f"{mode}_seed{seed}"
    command = [
        "python",
        "scripts/train_triage.py",
        "--train-csv",
        args.train_csv,
        "--val-csv",
        args.val_csv,
        "--output-dir",
        str(output_dir),
        "--mode",
        mode,
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--street-backbone",
        args.street_backbone,
        "--overhead-backbone",
        args.overhead_backbone,
        "--image-size",
        str(args.image_size),
        "--label-col",
        args.label_col,
        "--class-weighting",
        args.class_weighting,
    ]
    if args.street_augment:
        command.append("--street-augment")
    if args.overhead_augment:
        command.append("--overhead-augment")
    return command


def main() -> None:
    args = parse_args()
    plan_rows: list[dict[str, object]] = []
    for mode in args.modes:
        for seed in args.seeds:
            command = _build_command(args, mode, seed)
            plan_rows.append(
                {
                    "dataset": args.dataset_name,
                    "mode": mode,
                    "seed": seed,
                    "output_dir": str(Path(args.output_root) / args.dataset_name / f"{mode}_seed{seed}"),
                    "command": " ".join(f'"{part}"' if " " in part else part for part in command),
                }
            )

    plan_df = pd.DataFrame(plan_rows)
    plan_path = ensure_dir(args.output_root) / f"{args.dataset_name}_multiseed_plan.csv"
    plan_df.to_csv(plan_path, index=False)
    print(f"Saved plan: {plan_path}")

    if args.execute_index is not None:
        row = plan_rows[args.execute_index]
        command = _build_command(args, str(row["mode"]), int(row["seed"]))
        print("Executing:", " ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
