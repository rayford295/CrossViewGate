"""
B1: Decompose conflict cases into confident-disagreement vs mutual-uncertainty.

Usage:
    python scripts/analyze_conflict_confidence.py \
        --wildfire-street  outputs/eaton_wildfire/triage_street_only_resnet18/test_predictions.csv \
        --wildfire-remote  outputs/eaton_wildfire/triage_remote_only_resnet18/test_predictions.csv \
        --wildfire-cross   outputs/eaton_wildfire/triage_crossview_resnet18/test_predictions.csv \
        --wildfire-split   data/splits/eaton_wildfire/test.csv \
        --hurricane-street outputs/ian_hurricane/triage_street_only_resnet18/test_predictions.csv \
        --hurricane-remote outputs/ian_hurricane/triage_remote_only_resnet18/test_predictions.csv \
        --hurricane-cross  outputs/ian_hurricane/triage_crossview_resnet18/test_predictions.csv \
        --hurricane-split  data/splits/ian_hurricane_minor_vs_severe/test.csv \
        --output-dir       outputs/analysis/conflict_confidence

Requires a `prob_damaged` column in each predictions CSV (raw sigmoid score).
If your eval script does not save this, add --save-probs to eval_triage.py first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


def load_preds(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sample_id", "pred_label", "prob_damaged"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns: {missing}. "
            "Re-run eval_triage.py with --save-probs to include sigmoid scores."
        )
    return df


def build_conflict_confidence(
    street_df: pd.DataFrame,
    remote_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    split_df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    merged = (
        street_df.rename(columns={"pred_label": "pred_s", "prob_damaged": "prob_s"})
        .merge(
            remote_df.rename(columns={"pred_label": "pred_r", "prob_damaged": "prob_r"}),
            on="sample_id",
        )
        .merge(
            cross_df.rename(columns={"pred_label": "pred_c", "prob_damaged": "prob_c"}),
            on="sample_id",
        )
        .merge(split_df[["sample_id", "label"]], on="sample_id")
    )

    conflict = merged[merged["pred_s"] != merged["pred_r"]].copy()
    conflict["conf_s"] = (conflict["prob_s"] - 0.5).abs()
    conflict["conf_r"] = (conflict["prob_r"] - 0.5).abs()
    conflict["min_conf"] = conflict[["conf_s", "conf_r"]].min(axis=1)
    conflict["max_conf"] = conflict[["conf_s", "conf_r"]].max(axis=1)
    conflict["crossview_correct"] = (conflict["pred_c"] == conflict["label"]).astype(int)
    conflict["dataset"] = label
    return conflict


def print_summary(df: pd.DataFrame, label: str) -> None:
    print(f"\n{'='*50}")
    print(f"  {label}  (n_conflict={len(df)})")
    print(f"{'='*50}")
    print(df[["min_conf", "max_conf"]].describe().round(3).to_string())

    n_type_a = (df["min_conf"] >= 0.20).sum()
    n_type_b = (df["min_conf"] < 0.20).sum()
    pct_a = 100 * n_type_a / len(df)
    print(f"\n  Type A (min_conf >= 0.20, confident disagreement): {n_type_a}  ({pct_a:.1f}%)")
    print(f"  Type B (min_conf <  0.20, mutual uncertainty):     {n_type_b}  ({100-pct_a:.1f}%)")

    r, p = spearmanr(df["min_conf"], df["crossview_correct"])
    print(f"\n  Spearman r(min_conf, crossview_correct) = {r:.3f}  p = {p:.4f}")

    acc_a = df[df["min_conf"] >= 0.20]["crossview_correct"].mean()
    acc_b = df[df["min_conf"] < 0.20]["crossview_correct"].mean()
    print(f"  Crossview accuracy on Type A: {acc_a:.3f}")
    print(f"  Crossview accuracy on Type B: {acc_b:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Conflict confidence decomposition (B1)")
    for ds in ("wildfire", "hurricane"):
        parser.add_argument(f"--{ds}-street", required=True)
        parser.add_argument(f"--{ds}-remote", required=True)
        parser.add_argument(f"--{ds}-cross", required=True)
        parser.add_argument(f"--{ds}-split", required=True)
    parser.add_argument("--output-dir", default="outputs/analysis/conflict_confidence")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    wf = build_conflict_confidence(
        load_preds(args.wildfire_street),
        load_preds(args.wildfire_remote),
        load_preds(args.wildfire_cross),
        pd.read_csv(args.wildfire_split),
        label="wildfire",
    )
    ian = build_conflict_confidence(
        load_preds(args.hurricane_street),
        load_preds(args.hurricane_remote),
        load_preds(args.hurricane_cross),
        pd.read_csv(args.hurricane_split),
        label="hurricane",
    )

    print_summary(wf, "Wildfire")
    print_summary(ian, "Hurricane")

    stat, p_mw = mannwhitneyu(wf["min_conf"], ian["min_conf"], alternative="greater")
    print(f"\n{'='*50}")
    print("  Between-dataset test: wildfire min_conf > hurricane min_conf")
    print(f"  Mann-Whitney U = {stat:.1f},  p = {p_mw:.4f}")
    if p_mw < 0.05:
        print("  -> Wildfire conflicts are significantly more confident. B1 hypothesis SUPPORTED.")
    else:
        print("  -> No significant difference. B1 hypothesis NOT supported at p<0.05.")
    print(f"{'='*50}\n")

    combined = pd.concat([wf, ian], ignore_index=True)
    combined.to_csv(out / "conflict_confidence_per_sample.csv", index=False)

    summary = {
        "wildfire": {
            "n_conflict": len(wf),
            "min_conf_mean": float(wf["min_conf"].mean()),
            "min_conf_median": float(wf["min_conf"].median()),
            "pct_type_a": float((wf["min_conf"] >= 0.20).mean()),
            "crossview_acc_type_a": float(wf[wf["min_conf"] >= 0.20]["crossview_correct"].mean()),
            "crossview_acc_type_b": float(wf[wf["min_conf"] < 0.20]["crossview_correct"].mean()),
        },
        "hurricane": {
            "n_conflict": len(ian),
            "min_conf_mean": float(ian["min_conf"].mean()),
            "min_conf_median": float(ian["min_conf"].median()),
            "pct_type_a": float((ian["min_conf"] >= 0.20).mean()),
            "crossview_acc_type_a": float(ian[ian["min_conf"] >= 0.20]["crossview_correct"].mean()),
            "crossview_acc_type_b": float(ian[ian["min_conf"] < 0.20]["crossview_correct"].mean()),
        },
        "mannwhitney_u": float(stat),
        "mannwhitney_p": float(p_mw),
        "b1_hypothesis_supported": bool(p_mw < 0.05),
    }
    with open(out / "conflict_confidence_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved per-sample CSV and summary JSON to {out}/")


if __name__ == "__main__":
    main()
