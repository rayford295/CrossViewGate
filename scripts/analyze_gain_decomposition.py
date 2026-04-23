"""
Insight 1 + 3: Gain decomposition and S→O correction type analysis.

Computes:
  - overhead_advantage  = remote - street   (on conflict subset)
  - street_contribution = crossview - remote (on conflict subset)
  - street_share        = street_contribution / (crossview - 0.5)

Then breaks the conflict subset into three types:
  - Remote wins: remote correct, street wrong  (overhead is the truth)
  - Street wins: street correct, remote wrong  (S→O correction cases)
  - Both wrong:  neither correct

Reports crossview accuracy on each type separately.

Usage:
    python scripts/analyze_gain_decomposition.py \
        --wildfire-street  outputs/eaton_wildfire/triage_street_only_resnet18/test_predictions.csv \
        --wildfire-remote  outputs/eaton_wildfire/triage_remote_only_resnet18/test_predictions.csv \
        --wildfire-cross   outputs/eaton_wildfire/triage_crossview_resnet18/test_predictions.csv \
        --wildfire-split   data/splits/eaton_wildfire/test.csv \
        --hurricane-street outputs/ian_hurricane/triage_street_only_resnet18/test_predictions.csv \
        --hurricane-remote outputs/ian_hurricane/triage_remote_only_resnet18/test_predictions.csv \
        --hurricane-cross  outputs/ian_hurricane/triage_crossview_resnet18/test_predictions.csv \
        --hurricane-split  data/splits/ian_hurricane_minor_vs_severe/test.csv \
        --output-dir       outputs/analysis/gain_decomposition
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_merged(street_path, remote_path, cross_path, split_path):
    s = pd.read_csv(street_path)[["sample_id", "pred_label"]].rename(columns={"pred_label": "pred_s"})
    r = pd.read_csv(remote_path)[["sample_id", "pred_label"]].rename(columns={"pred_label": "pred_r"})
    c = pd.read_csv(cross_path) [["sample_id", "pred_label"]].rename(columns={"pred_label": "pred_c"})
    sp = pd.read_csv(split_path)[["sample_id", "label"]]
    return sp.merge(s, on="sample_id").merge(r, on="sample_id").merge(c, on="sample_id")


def analyze(df: pd.DataFrame, label: str) -> dict:
    conflict = df[df["pred_s"] != df["pred_r"]].copy()
    n = len(conflict)
    gt = conflict["label"].values

    acc_s = (conflict["pred_s"] == conflict["label"]).mean()
    acc_r = (conflict["pred_r"] == conflict["label"]).mean()
    acc_c = (conflict["pred_c"] == conflict["label"]).mean()

    overhead_adv   = acc_r - acc_s
    street_contrib = acc_c - acc_r
    total_gain     = acc_c - 0.5
    street_share   = street_contrib / total_gain if total_gain > 0 else 0.0

    remote_wins = (conflict["pred_r"] == conflict["label"]) & (conflict["pred_s"] != conflict["label"])
    street_wins = (conflict["pred_s"] == conflict["label"]) & (conflict["pred_r"] != conflict["label"])
    both_wrong  = (conflict["pred_s"] != conflict["label"]) & (conflict["pred_r"] != conflict["label"])

    sw_subset = conflict[street_wins]
    cross_acc_on_sw = (sw_subset["pred_c"] == sw_subset["label"]).mean() if len(sw_subset) > 0 else float("nan")

    rw_subset = conflict[remote_wins]
    cross_acc_on_rw = (rw_subset["pred_c"] == rw_subset["label"]).mean() if len(rw_subset) > 0 else float("nan")

    print(f"\n{'='*58}")
    print(f"  {label}  (n_conflict={n})")
    print(f"{'='*58}")
    print(f"  street accuracy on conflict:  {acc_s:.4f}")
    print(f"  remote accuracy on conflict:  {acc_r:.4f}")
    print(f"  crossview accuracy:           {acc_c:.4f}")
    print(f"")
    print(f"  overhead advantage  (remote − street): +{overhead_adv:.4f}")
    print(f"  street contribution (cross  − remote): +{street_contrib:.4f}")
    print(f"  street share of total gain:             {street_share:.1%}")
    print(f"")
    print(f"  Conflict taxonomy:")
    print(f"    Remote wins (overhead corrects street): {remote_wins.sum():3d}  ({remote_wins.mean():.1%})")
    print(f"    Street wins (street corrects overhead): {street_wins.sum():3d}  ({street_wins.mean():.1%})")
    print(f"    Both wrong:                             {both_wrong.sum():3d}  ({both_wrong.mean():.1%})")
    print(f"")
    print(f"  Crossview on Remote-wins cases: {cross_acc_on_rw:.3f}")
    print(f"  Crossview on Street-wins cases: {cross_acc_on_sw:.3f}  ← key")
    print(f"{'='*58}")

    return {
        "dataset": label,
        "n_conflict": n,
        "acc_street": float(acc_s),
        "acc_remote": float(acc_r),
        "acc_crossview": float(acc_c),
        "overhead_advantage": float(overhead_adv),
        "street_contribution": float(street_contrib),
        "street_share": float(street_share),
        "remote_wins_n": int(remote_wins.sum()),
        "street_wins_n": int(street_wins.sum()),
        "both_wrong_n": int(both_wrong.sum()),
        "cross_acc_on_remote_wins": float(cross_acc_on_rw),
        "cross_acc_on_street_wins": float(cross_acc_on_sw),
    }


def make_decomposition_figure(results: list[dict], out_path: Path) -> None:
    if not HAS_MPL:
        return
    labels = [r["dataset"] for r in results]
    overhead = [r["overhead_advantage"] for r in results]
    street   = [r["street_contribution"] for r in results]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5, 4))
    b1 = ax.bar(x, overhead, color="#4A90C4", label="Overhead advantage (remote − street)")
    b2 = ax.bar(x, street,   bottom=overhead, color="#D95F42", label="Street contribution (cross − remote)")

    for i, (ov, st) in enumerate(zip(overhead, street)):
        ax.text(i, ov / 2,       f"+{ov:.3f}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        ax.text(i, ov + st / 2,  f"+{st:.3f}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in labels], fontsize=11)
    ax.set_ylabel("Conflict-subset accuracy gain over random (50%)", fontsize=9)
    ax.set_title("Cross-view gain decomposition on conflict subset", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, max(o + s for o, s in zip(overhead, street)) * 1.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"\nFigure saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gain decomposition + S→O analysis")
    for ds in ("wildfire", "hurricane"):
        parser.add_argument(f"--{ds}-street", required=True)
        parser.add_argument(f"--{ds}-remote", required=True)
        parser.add_argument(f"--{ds}-cross",  required=True)
        parser.add_argument(f"--{ds}-split",  required=True)
    parser.add_argument("--output-dir", default="outputs/analysis/gain_decomposition")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    wf_df  = load_merged(args.wildfire_street,  args.wildfire_remote,  args.wildfire_cross,  args.wildfire_split)
    ian_df = load_merged(args.hurricane_street, args.hurricane_remote, args.hurricane_cross, args.hurricane_split)

    results = [
        analyze(wf_df,  "wildfire"),
        analyze(ian_df, "hurricane"),
    ]

    with open(out / "gain_decomposition.json", "w") as f:
        json.dump(results, f, indent=2)

    make_decomposition_figure(results, out / "gain_decomposition.pdf")
    print(f"\nJSON saved to {out}/gain_decomposition.json")


if __name__ == "__main__":
    main()
