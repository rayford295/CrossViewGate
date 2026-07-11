#!/usr/bin/env python
"""RQ1 Study A: directional anatomy of cross-view disagreement on CVIAN.

Implements docs/results/disagreement_anatomy_protocol.md. Exploratory only:
reads gate_fit and risk_calibration role exports, never final_test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ALLOWED_ROLES = ("gate_fit", "risk_calibration")
VISIBILITY_FEATURES = [
    "building_ratio",
    "center_building_ratio",
    "center_minus_global",
    "centroid_distance_norm",
]
CONFIDENCE_FEATURES = ["street_confidence", "remote_confidence"]
BLOCK_COLUMN = "spatial_block_id"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_role_frames(predictions_dir: Path, seed: int) -> dict[str, pd.DataFrame]:
    """Load the allowed role exports for one seed, refusing final_test."""

    frames = {}
    for role in ALLOWED_ROLES:
        path = predictions_dir / f"ian_original_seed{seed}_{role}.csv"
        if "final_test" in path.name:
            raise ValueError(f"final_test input is forbidden by protocol: {path}")
        frame = pd.read_csv(path)
        bad_roles = set(frame["protocol_role"].unique()) - set(ALLOWED_ROLES)
        if bad_roles:
            raise ValueError(f"{path} contains forbidden roles: {sorted(bad_roles)}")
        frames[role] = frame
    return frames


def add_anatomy_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["disagree"] = frame["street_prediction"] != frame["remote_prediction"]
    frame["signed_direction"] = frame["remote_prediction"] - frame["street_prediction"]
    street_correct = frame["street_prediction"] == frame["target"]
    remote_correct = frame["remote_prediction"] == frame["target"]
    frame["decidable"] = frame["disagree"] & (street_correct ^ remote_correct)
    frame["street_correct"] = street_correct
    return frame


def cluster_bootstrap_ci(
    frame: pd.DataFrame,
    statistic,
    replicates: int,
    rng: np.random.Generator,
    block_column: str = BLOCK_COLUMN,
) -> tuple[float, float, float]:
    """Point estimate and 95% CI, resampling spatial blocks with replacement."""

    point = statistic(frame)
    blocks = frame[block_column].unique()
    groups = {block: group for block, group in frame.groupby(block_column)}
    samples = []
    for _ in range(replicates):
        chosen = rng.choice(blocks, size=len(blocks), replace=True)
        resampled = pd.concat([groups[block] for block in chosen], ignore_index=True)
        value = statistic(resampled)
        if value is not None and not np.isnan(value):
            samples.append(value)
    if not samples:
        return point, float("nan"), float("nan")
    low, high = np.percentile(samples, [2.5, 97.5])
    return point, float(low), float(high)


def positive_direction_share(frame: pd.DataFrame) -> float:
    disagreements = frame[frame["disagree"]]
    if disagreements.empty:
        return float("nan")
    return float((disagreements["signed_direction"] > 0).mean())


def visibility_median_difference(frame: pd.DataFrame) -> float:
    decidable = frame[frame["decidable"]]
    street_rows = decidable[decidable["street_correct"]]
    remote_rows = decidable[~decidable["street_correct"]]
    if street_rows.empty or remote_rows.empty:
        return float("nan")
    return float(
        street_rows["center_building_ratio"].median()
        - remote_rows["center_building_ratio"].median()
    )


def rank_biserial(frame: pd.DataFrame) -> float:
    """Rank-biserial effect size for center_building_ratio, street- vs remote-correct."""

    decidable = frame[frame["decidable"]]
    a = decidable.loc[decidable["street_correct"], "center_building_ratio"].to_numpy()
    b = decidable.loc[~decidable["street_correct"], "center_building_ratio"].to_numpy()
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    greater = sum((x > b).sum() for x in a)
    ties = sum((x == b).sum() for x in a)
    u = greater + 0.5 * ties
    return float(2.0 * u / (len(a) * len(b)) - 1.0)


def probe_auroc(
    fit_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    features: list[str],
) -> float:
    """Fit a logistic probe on gate_fit decidable rows, AUROC on risk_calibration."""

    fit_rows = fit_frame[fit_frame["decidable"]]
    eval_rows = eval_frame[eval_frame["decidable"]]
    if fit_rows.empty or eval_rows.empty or eval_rows["street_correct"].nunique() < 2:
        return float("nan")
    scaler = StandardScaler().fit(fit_rows[features])
    model = LogisticRegression(max_iter=1000)
    model.fit(scaler.transform(fit_rows[features]), fit_rows["street_correct"])
    scores = model.predict_proba(scaler.transform(eval_rows[features]))[:, 1]
    return float(roc_auc_score(eval_rows["street_correct"], scores))


def auroc_statistic(features: list[str], fit_frame: pd.DataFrame):
    """Bootstrap statistic: refit-free evaluation resampling of eval blocks."""

    def statistic(eval_frame: pd.DataFrame) -> float:
        return probe_auroc(fit_frame, eval_frame, features)

    return statistic


def analyze_seed(
    frames: dict[str, pd.DataFrame],
    replicates: int,
    rng: np.random.Generator,
) -> dict:
    fit_frame = add_anatomy_columns(frames["gate_fit"])
    eval_frame = add_anatomy_columns(frames["risk_calibration"])
    pooled = pd.concat([fit_frame, eval_frame], ignore_index=True)

    result = {
        "n_rows": int(len(pooled)),
        "n_disagreements": int(pooled["disagree"].sum()),
        "disagreement_rate": float(pooled["disagree"].mean()),
        "n_decidable": int(pooled["decidable"].sum()),
        "decidable_street_correct_share": float(
            pooled.loc[pooled["decidable"], "street_correct"].mean()
        )
        if pooled["decidable"].any()
        else float("nan"),
    }

    for name, statistic in [
        ("h_a1_positive_direction_share", positive_direction_share),
        ("h_a2_visibility_median_diff", visibility_median_difference),
    ]:
        point, low, high = cluster_bootstrap_ci(pooled, statistic, replicates, rng)
        result[name] = point
        result[f"{name}_ci_low"] = low
        result[f"{name}_ci_high"] = high
    result["h_a2_rank_biserial"] = rank_biserial(pooled)

    for label, features in [
        ("visibility_only", VISIBILITY_FEATURES),
        ("confidence_only", CONFIDENCE_FEATURES),
        ("combined", VISIBILITY_FEATURES + CONFIDENCE_FEATURES),
    ]:
        point, low, high = cluster_bootstrap_ci(
            eval_frame, auroc_statistic(features, fit_frame), replicates, rng
        )
        result[f"h_a3_auroc_{label}"] = point
        result[f"h_a3_auroc_{label}_ci_low"] = low
        result[f"h_a3_auroc_{label}_ci_high"] = high
    return result


def development_verdicts(per_seed: pd.DataFrame) -> dict[str, bool]:
    """Apply the pre-specified >=4/5 (>=3/5 for H-A3) development criteria."""

    h_a1 = (
        (
            (per_seed["h_a1_positive_direction_share_ci_low"] > 0.5)
            | (per_seed["h_a1_positive_direction_share_ci_high"] < 0.5)
        ).sum()
        >= 4
        and (per_seed["h_a1_positive_direction_share"] > 0.5).nunique() == 1
    )
    h_a2 = (per_seed["h_a2_visibility_median_diff_ci_low"] > 0).sum() >= 4
    h_a3 = (per_seed["h_a3_auroc_visibility_only_ci_low"] > 0.5).sum() >= 3
    return {
        "h_a1_directional_asymmetry": bool(h_a1),
        "h_a2_visibility_conditions_correctness": bool(h_a2),
        "h_a3_visibility_explains_held_out": bool(h_a3),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="42,123,456,789,1011")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    seeds = [int(seed) for seed in args.seeds.split(",")]
    rng = np.random.default_rng(args.bootstrap_seed)

    fingerprints = {}
    rows = []
    for seed in seeds:
        frames = load_role_frames(args.predictions_dir, seed)
        for role in ALLOWED_ROLES:
            path = args.predictions_dir / f"ian_original_seed{seed}_{role}.csv"
            fingerprints[path.name] = sha256_file(path)
        row = {"seed": seed}
        row.update(analyze_seed(frames, args.bootstrap_replicates, rng))
        rows.append(row)

    per_seed = pd.DataFrame(rows)
    verdicts = development_verdicts(per_seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.output_dir / "per_seed_statistics.csv", index=False)
    summary = {
        "protocol": "docs/results/disagreement_anatomy_protocol.md",
        "claim_scope": "exploratory/development; final_test untouched",
        "roles_used": list(ALLOWED_ROLES),
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
            "cluster_column": BLOCK_COLUMN,
        },
        "input_fingerprints_sha256": fingerprints,
        "development_verdicts": verdicts,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"verdicts": verdicts, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
