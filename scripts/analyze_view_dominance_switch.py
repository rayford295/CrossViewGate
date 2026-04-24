"""
Innovation A: View Dominance Switching analysis.

In conflict cases, which single-view model is more often correct?
And does crossview succeed specifically on the type of conflict
where the dominant view is being overridden?

Computes:
  S→O cases: street correct, overhead wrong  (street is the truth-teller)
  O→S cases: overhead correct, street wrong  (overhead is the truth-teller)
  Both-wrong: neither single-view is correct

For each conflict type, reports crossview accuracy.
Shows whether the dominant conflict type flips across datasets.

Usage:
    python scripts/analyze_view_dominance_switch.py \
        --wildfire-street  outputs/eaton_wildfire/triage_street_only_resnet18/test_predictions.csv \
        --wildfire-remote  outputs/eaton_wildfire/triage_remote_only_resnet18/test_predictions.csv \
        --wildfire-cross   outputs/eaton_wildfire/triage_crossview_resnet18/test_predictions.csv \
        --wildfire-split   data/splits/eaton_wildfire/test.csv \
        --hurricane-street outputs/ian_hurricane/triage_street_only_resnet18/test_predictions.csv \
        --hurricane-remote outputs/ian_hurricane/triage_remote_only_resnet18/test_predictions.csv \
        --hurricane-cross  outputs/ian_hurricane/triage_crossview_resnet18/test_predictions.csv \
        --hurricane-split  data/splits/ian_hurricane_minor_vs_severe/test.csv \
        --tau 0.1 \
        --output-dir outputs/analysis/view_dominance_switch

Requires prob_damaged column in prediction CSVs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load(path: str, pred_col: str = "pred_label", prob_col: str = "prob_damaged") -> pd.DataFrame:
    df = pd.read_csv(path)
    available_pred = next((c for c in [pred_col, "prediction"] if c in df.columns), None)
    available_prob = next((c for c in [prob_col, "probability", "prob"] if c in df.columns), None)
    cols = ["sample_id"]
    rename = {}
    if available_pred:
        cols.append(available_pred); rename[available_pred] = "pred"
    if available_prob:
        cols.append(available_prob); rename[available_prob] = "prob"
    out = df[cols].rename(columns=rename)
    out["sample_id"] = (
        out["sample_id"]
        .astype(str)
        .str.replace(r"^tensor\((.+)\)$", r"\1", regex=True)
    )
    return out


def analyze(
    street_df: pd.DataFrame,
    remote_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    split_df: pd.DataFrame,
    tau: float,
    label: str,
) -> dict:
    label_col = next(c for c in ["label", "binary_label", "target"] if c in split_df.columns)
    df = (
        split_df[["sample_id", label_col]]
        .rename(columns={label_col: "gt"})
        .assign(sample_id=lambda x: x["sample_id"].astype(str))
        .merge(street_df.rename(columns={"pred": "pred_s", "prob": "prob_s"}), on="sample_id")
        .merge(remote_df.rename(columns={"pred": "pred_r", "prob": "prob_r"}), on="sample_id")
        .merge(cross_df .rename(columns={"pred": "pred_c", "prob": "prob_c"}), on="sample_id")
    )

    # Use tau-based conflict definition if prob columns available
    if "prob_s" in df.columns and "prob_r" in df.columns:
        conflict_mask = (df["prob_s"] - df["prob_r"]).abs() > tau
    else:
        conflict_mask = df["pred_s"] != df["pred_r"]

    conflict = df[conflict_mask].copy()
    n_total  = len(df)
    n_conf   = len(conflict)

    # Conflict type taxonomy
    so_mask = (conflict["pred_s"] == conflict["gt"]) & (conflict["pred_r"] != conflict["gt"])
    os_mask = (conflict["pred_r"] == conflict["gt"]) & (conflict["pred_s"] != conflict["gt"])
    bw_mask = (conflict["pred_s"] != conflict["gt"]) & (conflict["pred_r"] != conflict["gt"])
    bc_mask = (conflict["pred_s"] == conflict["gt"]) & (conflict["pred_r"] == conflict["gt"])

    # Crossview accuracy on each type
    def cross_acc(mask):
        sub = conflict[mask]
        return (sub["pred_c"] == sub["gt"]).mean() if len(sub) > 0 else float("nan"), len(sub)

    so_cross, so_n = cross_acc(so_mask)
    os_cross, os_n = cross_acc(os_mask)
    bw_cross, bw_n = cross_acc(bw_mask)
    bc_cross, bc_n = cross_acc(bc_mask)

    overall_acc_s = (conflict["pred_s"] == conflict["gt"]).mean()
    overall_acc_r = (conflict["pred_r"] == conflict["gt"]).mean()
    overall_acc_c = (conflict["pred_c"] == conflict["gt"]).mean()

    dominant = "street" if so_mask.sum() > os_mask.sum() else "overhead"

    print(f"\n{'='*62}")
    print(f"  {label}   tau={tau}")
    print(f"  n_total={n_total}   n_conflict={n_conf} ({n_conf/n_total:.1%})")
    print(f"{'='*62}")
    print(f"  Single-view accuracy on conflict subset:")
    print(f"    street:    {overall_acc_s:.4f}")
    print(f"    remote:    {overall_acc_r:.4f}")
    print(f"    crossview: {overall_acc_c:.4f}")
    print(f"  → Dominant single-view arbitrator: {dominant.upper()}")
    print()
    print(f"  Conflict type breakdown:")
    print(f"    S→O (street corrects overhead): n={so_n:3d}  ({so_mask.mean():.1%})  "
          f"crossview acc={so_cross:.3f}")
    print(f"    O→S (overhead corrects street): n={os_n:3d}  ({os_mask.mean():.1%})  "
          f"crossview acc={os_cross:.3f}")
    print(f"    Both-correct:                   n={bc_n:3d}  ({bc_mask.mean():.1%})  "
          f"crossview acc={bc_cross:.3f}")
    print(f"    Both-wrong:                     n={bw_n:3d}  ({bw_mask.mean():.1%})  "
          f"crossview acc={bw_cross:.3f}")
    print()

    if dominant == "street":
        print(f"  VIEW DOMINANCE: street regime (wildfire pattern)")
        print(f"  → Ground view corrects overhead in {so_mask.mean():.1%} of conflicts")
    else:
        print(f"  VIEW DOMINANCE: overhead regime (hurricane pattern)")
        print(f"  → Overhead corrects ground view in {os_mask.mean():.1%} of conflicts")
    print(f"{'='*62}")

    return {
        "dataset": label,
        "tau": tau,
        "n_total": n_total,
        "n_conflict": n_conf,
        "conflict_rate": float(n_conf / n_total),
        "acc_street_conflict": float(overall_acc_s),
        "acc_remote_conflict": float(overall_acc_r),
        "acc_crossview_conflict": float(overall_acc_c),
        "dominant_view": dominant,
        "so_n": so_n,
        "os_n": os_n,
        "bc_n": bc_n,
        "bw_n": bw_n,
        "so_frac": float(so_mask.mean()),
        "os_frac": float(os_mask.mean()),
        "bc_frac": float(bc_mask.mean()),
        "bw_frac": float(bw_mask.mean()),
        "cross_acc_on_so": float(so_cross),
        "cross_acc_on_os": float(os_cross),
        "cross_acc_on_bc": float(bc_cross),
        "cross_acc_on_bw": float(bw_cross),
    }


def make_figure(results: list[dict], out_path: Path) -> None:
    if not HAS_MPL or len(results) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    colors = {
        "S→O": "#D95F42",
        "O→S": "#4A90C4",
        "Both-correct": "#6BAE75",
        "Both-wrong": "#B0AEA5",
    }

    for ax, r in zip(axes, results):
        fracs = [r["so_frac"], r["os_frac"], r["bc_frac"], r["bw_frac"]]
        labels_pie = [
            f"S→O\n{r['so_frac']:.1%}\n(cross={r['cross_acc_on_so']:.2f})",
            f"O→S\n{r['os_frac']:.1%}\n(cross={r['cross_acc_on_os']:.2f})",
            f"Both-correct\n{r['bc_frac']:.1%}\n(cross={r['cross_acc_on_bc']:.2f})",
            f"Both-wrong\n{r['bw_frac']:.1%}\n(cross={r['cross_acc_on_bw']:.2f})",
        ]
        ax.pie(
            fracs,
            labels=labels_pie,
            colors=list(colors.values()),
            startangle=90,
            textprops={"fontsize": 8},
        )
        dominant = r["dominant_view"].upper()
        ax.set_title(
            f"{r['dataset'].capitalize()} conflict types (τ={r['tau']})\n"
            f"Dominant: {dominant}  —  crossview={r['acc_crossview_conflict']:.3f}",
            fontsize=9,
        )

    plt.suptitle("View Dominance Switching across disaster regimes", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"\nFigure saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="View Dominance Switch analysis")
    for ds in ("wildfire", "hurricane"):
        parser.add_argument(f"--{ds}-street", required=True)
        parser.add_argument(f"--{ds}-remote", required=True)
        parser.add_argument(f"--{ds}-cross",  required=True)
        parser.add_argument(f"--{ds}-split",  required=True)
    parser.add_argument("--tau",        type=float, default=0.1)
    parser.add_argument("--output-dir", default="outputs/analysis/view_dominance_switch")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def load_all(ds: str):
        return (
            load(getattr(args, f"{ds}_street")),
            load(getattr(args, f"{ds}_remote")),
            load(getattr(args, f"{ds}_cross")),
            pd.read_csv(getattr(args, f"{ds}_split")),
        )

    results = [
        analyze(*load_all("wildfire"),  tau=args.tau, label="wildfire"),
        analyze(*load_all("hurricane"), tau=args.tau, label="hurricane"),
    ]

    with open(out / f"view_dominance_switch_tau{args.tau}.json", "w") as f:
        json.dump(results, f, indent=2)

    make_figure(results, out / f"view_dominance_switch_tau{args.tau}.pdf")

    # Summary verdict
    print("\n=== VIEW DOMINANCE SWITCH VERDICT ===")
    ds_dominant = {r["dataset"]: r["dominant_view"] for r in results}
    if len(set(ds_dominant.values())) > 1:
        print("✓ Switch confirmed: dominant view differs across datasets.")
        for ds, dom in ds_dominant.items():
            print(f"  {ds}: {dom.upper()} is the dominant arbitrator in conflicts")
    else:
        print("✗ No switch: both datasets show the same dominant view.")
        print("  Hypothesis not supported — revisit mechanism story.")

    print(f"\nResults saved to {out}/")


if __name__ == "__main__":
    main()
