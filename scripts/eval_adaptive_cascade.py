"""
Innovation I4: Adaptive inference cascade evaluation.

Stage 1: run both single-view models. If |logit_s - logit_r| <= theta,
         return the averaged prediction (no crossview needed).
Stage 2: for conflicting cases (|logit_s - logit_r| > theta),
         use the crossview model's prediction instead.

Sweeps theta and reports the efficiency-accuracy tradeoff.

Usage:
    python scripts/eval_adaptive_cascade.py \
        --street-preds  outputs/eaton_wildfire/triage_street_only_resnet18/test_predictions.csv \
        --remote-preds  outputs/eaton_wildfire/triage_remote_only_resnet18/test_predictions.csv \
        --cross-preds   outputs/eaton_wildfire/triage_crossview_resnet18/test_predictions.csv \
        --split-csv     data/splits/eaton_wildfire/test.csv \
        --output-dir    outputs/analysis/adaptive_cascade \
        --dataset       wildfire

Requires prob_damaged column in each predictions CSV.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def _find_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"Missing {label} column. Tried: {candidates}")


def cascade_predict(
    prob_s: np.ndarray,
    prob_r: np.ndarray,
    prob_c: np.ndarray,
    theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (predictions, routed_to_stage2 mask).
    Stage 1: average single-view probs when |prob_s - prob_r| <= theta.
    Stage 2: use crossview prob when |prob_s - prob_r| > theta.
    """
    conflict_mask = np.abs(prob_s - prob_r) > theta
    avg_prob = (prob_s + prob_r) / 2.0
    cascade_prob = np.where(conflict_mask, prob_c, avg_prob)
    preds = (cascade_prob > 0.5).astype(int)
    return preds, conflict_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="I4: adaptive cascade evaluation")
    parser.add_argument("--street-preds", required=True)
    parser.add_argument("--remote-preds", required=True)
    parser.add_argument("--cross-preds",  required=True)
    parser.add_argument("--split-csv",    required=True)
    parser.add_argument("--output-dir",   default="outputs/analysis/adaptive_cascade")
    parser.add_argument("--dataset",      default="dataset")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def load(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        pred_col = _find_column(df, ["pred_label", "prediction"], "prediction")
        prob_col = _find_column(df, ["prob_damaged", "probability"], "probability")
        out_df = df[["sample_id", pred_col, prob_col]].copy()
        out_df["sample_id"] = out_df["sample_id"].astype(str).str.replace(r"^tensor\((.*)\)$", r"\1", regex=True)
        return out_df.rename(
            columns={pred_col: "pred_label", prob_col: "prob_damaged"}
        )

    street = load(args.street_preds).rename(columns={"pred_label": "pred_s", "prob_damaged": "prob_s"})
    remote = load(args.remote_preds).rename(columns={"pred_label": "pred_r", "prob_damaged": "prob_r"})
    cross  = load(args.cross_preds) .rename(columns={"pred_label": "pred_c", "prob_damaged": "prob_c"})
    split_df = pd.read_csv(args.split_csv)
    label_col = _find_column(split_df, ["label", "binary_label", "target"], "label")
    split = split_df[["sample_id", label_col]].rename(columns={label_col: "label"})
    split["sample_id"] = split["sample_id"].astype(str)

    df = split.merge(street, on="sample_id").merge(remote, on="sample_id").merge(cross, on="sample_id")
    labels = df["label"].values

    # Baselines
    f1_street = f1_score(labels, df["pred_s"].values)
    f1_remote = f1_score(labels, df["pred_r"].values)
    f1_cross  = f1_score(labels, df["pred_c"].values)

    print(f"\n{'='*60}")
    print(f"  Dataset: {args.dataset}   n={len(df)}")
    print(f"{'='*60}")
    print(f"  Baselines: street={f1_street:.4f}  remote={f1_remote:.4f}  crossview={f1_cross:.4f}")
    print()

    thetas = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    rows = []

    for theta in thetas:
        preds, mask = cascade_predict(
            df["prob_s"].values, df["prob_r"].values, df["prob_c"].values, theta
        )
        pct_stage2 = mask.mean() * 100
        f1_casc = f1_score(labels, preds)
        f1_drop = f1_cross - f1_casc
        conflict_recall = (mask & (df["pred_s"].values != df["pred_r"].values)).sum() / \
                          max(1, (df["pred_s"].values != df["pred_r"].values).sum())

        rows.append({
            "theta": theta,
            "pct_stage2": pct_stage2,
            "f1_cascade": f1_casc,
            "f1_drop_vs_crossview": f1_drop,
            "conflict_recall": conflict_recall,
        })
        print(
            f"  theta={theta:.2f}  stage2={pct_stage2:5.1f}%  "
            f"F1={f1_casc:.4f}  delta={f1_drop:+.4f}  "
            f"conflict_recall={conflict_recall:.3f}"
        )

    print(f"{'='*60}\n")
    results_df = pd.DataFrame(rows)
    results_df.to_csv(out / f"cascade_sweep_{args.dataset}.csv", index=False)

    summary = {
        "dataset": args.dataset,
        "n": len(df),
        "f1_street": float(f1_street),
        "f1_remote": float(f1_remote),
        "f1_crossview": float(f1_cross),
        "cascade_results": rows,
    }
    with open(out / f"cascade_summary_{args.dataset}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out}/")


if __name__ == "__main__":
    main()
