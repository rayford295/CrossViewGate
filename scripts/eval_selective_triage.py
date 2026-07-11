"""Calibrate and evaluate a finite-sample selective-triage policy.

This command consumes three distinct per-sample CSVs:

1. gate-fit (used only for role/disjointness auditing here),
2. risk-calibration (the only split used to select a threshold), and
3. final-test (evaluated once at the locked threshold).

It accepts both the compact ``*_predictions.csv`` files and the richer
reliability-gate evidence CSVs produced by this repository.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.decision.risk_control import (
    DEFAULT_REVIEW_BUDGETS,
    assert_disjoint_protocol_splits,
    bounded_classification_loss,
    build_per_sample_decisions,
    calibrate_threshold,
    critical_target_mask,
    evaluate_locked_threshold,
    recall_at_review_budgets,
    risk_coverage_curve,
)


SCHEMA_VERSION = "p0.7-v1"
PREDICTION_CANDIDATES = (
    "gate3_selected_prediction",
    "gate3_linear_prediction",
    "gate2_selected_prediction",
    "gate_linear_prediction",
    "prediction",
    "crossview_prediction",
    "remote_prediction",
    "overhead_prediction",
    "street_prediction",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Risk-calibrate a selective triage threshold and evaluate it once on a "
            "strictly separate final-test CSV."
        )
    )
    parser.add_argument("--gate-fit-csv", required=True)
    parser.add_argument(
        "--risk-calibration-csv", "--calibration-csv", dest="risk_calibration_csv", required=True
    )
    parser.add_argument("--final-test-csv", "--test-csv", dest="final_test_csv", required=True)
    parser.add_argument("--output-dir", default="outputs/analysis/selective_triage")
    parser.add_argument(
        "--id-columns",
        default="auto",
        help=(
            "Comma-separated stable identifiers used for split-disjointness audit. "
            "Default: dataset+sample_id when available. Use dataset,objectid for a "
            "property-level audit."
        ),
    )
    parser.add_argument(
        "--group-columns",
        default="auto",
        help=(
            "Comma-separated property/sequence/spatial group columns. Auto audits every "
            "shared candidate among sequence_id, spatial_block_id, tile_id, group_id, and objectid."
        ),
    )
    parser.add_argument(
        "--spatial-separation-m",
        type=float,
        help="Optionally require every cross-role coordinate pair to exceed this distance.",
    )
    parser.add_argument(
        "--risk-group-col",
        default="auto",
        help=(
            "Independent sampling/group unit for finite-sample bounds. Auto prefers "
            "spatial_block_id, tile_id, group_id, then objectid. Group losses are "
            "macro-averaged and bounded with Hoeffding."
        ),
    )
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--prediction-col", default="auto")
    parser.add_argument(
        "--risk-score-col",
        default="auto",
        help=(
            "Column where larger means riskier. Auto supports risk_score, confidence, "
            "gate3/gate2 probability columns, entropy, and JS divergence."
        ),
    )
    parser.add_argument(
        "--score-transform",
        choices=("auto", "identity", "one-minus", "negate"),
        default="auto",
        help=(
            "Transform applied to an explicit score column; auto turns confidence "
            "into 1-confidence."
        ),
    )
    parser.add_argument(
        "--loss",
        choices=("severe_miss", "extreme_error", "cost_weighted"),
        default="severe_miss",
    )
    parser.add_argument(
        "--severe-labels",
        default="",
        help="Comma-separated pre-registered severe/destroyed labels (required for severe_miss).",
    )
    parser.add_argument(
        "--label-order",
        default="",
        help="Comma-separated ordinal labels, from least to most severe.",
    )
    parser.add_argument("--extreme-distance", type=int, default=2)
    parser.add_argument(
        "--cost-matrix-json",
        help="Path to, or inline JSON for, a complete square/nested cost matrix.",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        help="Pre-registered upper bound used to normalize cost-weighted loss to [0,1].",
    )
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument(
        "--threshold-grid",
        default="0:1:0.01",
        help="Pre-registered comma list or inclusive START:STOP:STEP grid.",
    )
    parser.add_argument(
        "--bound-method",
        choices=("auto", "clopper_pearson", "hoeffding"),
        default="auto",
    )
    parser.add_argument(
        "--review-budgets",
        default=",".join(str(int(value)) for value in DEFAULT_REVIEW_BUDGETS),
        help="Comma-separated review percentages, default 5,10,20,30,50.",
    )
    parser.add_argument(
        "--recall-target",
        choices=("auto", "severe_target", "loss_event", "classification_error"),
        default="auto",
    )
    parser.add_argument(
        "--claim-scope",
        choices=("in_event", "cross_event_stress_test"),
        default="in_event",
    )
    parser.add_argument(
        "--decision-source",
        choices=("auto", "street", "overhead", "gated_mixture"),
        default="auto",
    )
    parser.add_argument(
        "--review-disposition",
        choices=("defer_human", "acquire_view"),
        default="defer_human",
    )
    parser.add_argument("--acquisition-target")
    return parser.parse_args(argv)


def _comma_values(specification: str) -> list[str]:
    return [value.strip() for value in specification.split(",") if value.strip()]


def _parse_threshold_grid(specification: str) -> np.ndarray:
    text = specification.strip()
    if not text:
        raise ValueError("threshold_grid must not be empty")
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 3:
            raise ValueError("A threshold range must use START:STOP:STEP")
        start, stop, step = (float(value) for value in parts)
        if not np.isfinite([start, stop, step]).all() or step <= 0.0 or stop < start:
            raise ValueError("Threshold START/STOP/STEP must be finite with STOP>=START and STEP>0")
        count = int(np.floor((stop - start) / step + 1e-10)) + 1
        values = start + step * np.arange(count, dtype=np.float64)
        if values[-1] < stop - 1e-10:
            values = np.append(values, stop)
        elif abs(values[-1] - stop) <= 1e-10:
            values[-1] = stop
        return values
    try:
        return np.asarray([float(value) for value in _comma_values(text)], dtype=np.float64)
    except ValueError as error:
        raise ValueError("threshold_grid contains a non-numeric value") from error


def _parse_budgets(specification: str) -> list[float]:
    try:
        budgets = [float(value) for value in _comma_values(specification)]
    except ValueError as error:
        raise ValueError("review_budgets contains a non-numeric value") from error
    if not budgets:
        raise ValueError("review_budgets must not be empty")
    return budgets


def _load_json_argument(specification: str | None) -> object | None:
    if specification is None:
        return None
    stripped = specification.strip()
    if stripped.startswith(("{", "[")):
        text = stripped
    else:
        candidate = Path(specification)
        if not candidate.is_file():
            raise FileNotFoundError(f"Cost-matrix JSON file not found: {candidate}")
        text = candidate.read_text(encoding="utf-8")
    return json.loads(text)


def _read_nonempty_csv(path: Path, role: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{role} CSV not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{role} CSV is empty: {path}")
    return frame


def _common_column(
    frames: Sequence[pd.DataFrame], requested: str, candidates: Sequence[str]
) -> str:
    if requested != "auto":
        missing = [index for index, frame in enumerate(frames) if requested not in frame.columns]
        if missing:
            raise ValueError(f"Column {requested!r} is missing from policy frames {missing}")
        return requested
    for candidate in candidates:
        if all(candidate in frame.columns for frame in frames):
            return candidate
    raise ValueError(f"No common column found among candidates: {list(candidates)}")


def _numbered_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    pairs: list[tuple[int, str]] = []
    for column in frame.columns:
        match = pattern.match(str(column))
        if match:
            pairs.append((int(match.group(1)), str(column)))
    return [column for _, column in sorted(pairs)]


def _resolve_score_spec(
    frames: Sequence[pd.DataFrame], requested: str, transform: str
) -> dict[str, object]:
    if requested != "auto":
        if any(requested not in frame.columns for frame in frames):
            raise ValueError(f"risk score column {requested!r} must exist in calibration and test")
        resolved_transform = transform
        if resolved_transform == "auto":
            resolved_transform = "one-minus" if "confidence" in requested.lower() else "identity"
        return {
            "kind": "column",
            "column": requested,
            "transform": resolved_transform,
            "source": f"{resolved_transform}({requested})",
        }

    if all("risk_score" in frame.columns for frame in frames):
        return {
            "kind": "column",
            "column": "risk_score",
            "transform": "identity",
            "source": "risk_score",
        }
    for confidence_column in ("gate3_confidence", "gate2_confidence", "confidence"):
        if all(confidence_column in frame.columns for frame in frames):
            return {
                "kind": "column",
                "column": confidence_column,
                "transform": "one-minus",
                "source": f"1-{confidence_column}",
            }
    for prefix in ("gate3_probability", "gate2_probability", "crossview_probability"):
        numbered = [_numbered_columns(frame, prefix) for frame in frames]
        if numbered[0] and all(columns == numbered[0] for columns in numbered[1:]):
            return {
                "kind": "max_columns",
                "columns": numbered[0],
                "transform": "one-minus",
                "source": f"1-max({prefix}_*)",
            }
    for risk_column in ("uncertainty", "crossview_entropy", "js_divergence"):
        if all(risk_column in frame.columns for frame in frames):
            return {
                "kind": "column",
                "column": risk_column,
                "transform": "identity",
                "source": risk_column,
            }
    raise ValueError(
        "Could not infer a common risk score. Supply --risk-score-col and, if needed, "
        "--score-transform."
    )


def _extract_scores(frame: pd.DataFrame, specification: dict[str, object]) -> np.ndarray:
    if specification["kind"] == "max_columns":
        columns = list(specification["columns"])
        values = frame[columns].to_numpy(dtype=np.float64).max(axis=1)
    else:
        values = pd.to_numeric(frame[str(specification["column"])], errors="raise").to_numpy(
            dtype=np.float64
        )
    transform = specification["transform"]
    if transform == "one-minus":
        values = 1.0 - values
    elif transform == "negate":
        values = -values
    elif transform != "identity":
        raise ValueError(f"Unsupported score transform: {transform}")
    if not np.isfinite(values).all():
        raise ValueError("Resolved risk scores contain non-finite values")
    return values


def _label_key(value: object) -> str:
    if pd.isna(value):
        raise ValueError("Labels may not be missing")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    integer_like = re.fullmatch(r"([+-]?\d+)(?:\.0+)?", text)
    if integer_like:
        return str(int(integer_like.group(1)))
    return text


def _maximum_registered_cost(cost_matrix: object | None) -> float | None:
    if cost_matrix is None:
        return None
    if isinstance(cost_matrix, dict):
        values = [float(cost) for row in cost_matrix.values() for cost in row.values()]
        return max(values) if values else None
    matrix = np.asarray(cost_matrix, dtype=np.float64)
    return float(matrix.max()) if matrix.size else None


def _classification_error(targets: pd.Series, predictions: pd.Series) -> np.ndarray:
    return np.asarray(
        [_label_key(target) != _label_key(prediction)
         for target, prediction in zip(targets, predictions)],
        dtype=bool,
    )


def _infer_decision_source(prediction_column: str) -> str:
    lowered = prediction_column.lower()
    if "street" in lowered:
        return "street"
    if "remote" in lowered or "overhead" in lowered:
        return "overhead"
    return "gated_mixture"


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(payload: dict[str, object], path: Path) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _audit_event_identity(frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Infer whether protocol roles belong to one event or a transfer stress test."""
    candidates = ("event_id", "event", "source_dataset", "dataset")
    column = next(
        (candidate for candidate in candidates if all(candidate in frame for frame in frames.values())),
        None,
    )
    if column is None:
        return {
            "audit_complete": False,
            "event_column": None,
            "role_values": {},
            "observed_cross_event": False,
            "reason": "no_common_event_or_dataset_column",
        }
    role_values: dict[str, list[str]] = {}
    for role, frame in frames.items():
        if frame[column].isna().any():
            raise ValueError(f"{role}.{column} contains missing event identifiers")
        values = sorted({_normalized_event_value(value) for value in frame[column]})
        if len(values) != 1:
            raise ValueError(
                f"{role}.{column} must identify exactly one event; observed {values[:5]}"
            )
        role_values[role] = values
    observed_cross_event = len({values[0] for values in role_values.values()}) > 1
    return {
        "audit_complete": True,
        "event_column": column,
        "role_values": role_values,
        "observed_cross_event": observed_cross_event,
        "reason": "event_ids_differ" if observed_cross_event else "single_event_verified",
    }


def _normalized_event_value(value: object) -> str:
    if pd.isna(value):
        raise ValueError("Event identifiers may not be missing")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("Event identifiers may not be blank")
    return normalized


def _audit_policy_identity(frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    column = "gate_artifact_id"
    if not all(column in frame for frame in frames.values()):
        return {
            "audit_complete": False,
            "identity_column": None,
            "role_values": {},
            "reason": "gate_artifact_id_missing_from_one_or_more_roles",
        }
    role_values: dict[str, list[str]] = {}
    for role, frame in frames.items():
        normalized = frame[column].map(
            lambda value: "" if pd.isna(value) else str(value).strip()
        )
        if (normalized == "").any():
            raise ValueError(f"{role}.{column} contains missing or blank gate identities")
        values = sorted(set(normalized))
        if len(values) != 1:
            raise ValueError(
                f"{role}.{column} must contain one fixed gate identity; observed {values[:5]}"
            )
        role_values[role] = values
    identities = {values[0] for values in role_values.values()}
    if len(identities) != 1:
        raise ValueError(f"Protocol roles were produced by different gate artifacts: {role_values}")
    return {
        "audit_complete": True,
        "identity_column": column,
        "role_values": role_values,
        "reason": "one_fixed_gate_artifact_verified",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    gate_path = Path(args.gate_fit_csv)
    calibration_path = Path(args.risk_calibration_csv)
    test_path = Path(args.final_test_csv)
    resolved_paths = [path.resolve() for path in (gate_path, calibration_path, test_path)]
    if len(set(resolved_paths)) != 3:
        raise ValueError("gate-fit, risk-calibration, and final-test must be different CSV files")

    gate_fit = _read_nonempty_csv(gate_path, "gate-fit")
    calibration = _read_nonempty_csv(calibration_path, "risk-calibration")
    final_test = _read_nonempty_csv(test_path, "final-test")
    id_columns = None if args.id_columns == "auto" else _comma_values(args.id_columns)
    group_columns = (
        None if args.group_columns == "auto" else _comma_values(args.group_columns)
    )
    split_audit = assert_disjoint_protocol_splits(
        gate_fit,
        calibration,
        final_test,
        id_columns=id_columns,
        group_columns=group_columns,
        spatial_threshold_m=args.spatial_separation_m,
    )
    role_frames = {
        "gate_fit": gate_fit,
        "risk_calibration": calibration,
        "final_test": final_test,
    }
    event_audit = _audit_event_identity(role_frames)
    policy_identity_audit = _audit_policy_identity(role_frames)
    if args.risk_group_col == "auto":
        risk_group_column = next(
            (
                column
                for column in ("spatial_block_id", "tile_id", "group_id", "objectid")
                if all(column in frame for frame in role_frames.values())
            ),
            None,
        )
    else:
        risk_group_column = args.risk_group_col
        if any(risk_group_column not in frame for frame in role_frames.values()):
            raise ValueError(
                f"--risk-group-col {risk_group_column!r} must exist in every protocol role"
            )
    if risk_group_column is not None and risk_group_column not in split_audit["group_columns"]:
        raise ValueError(
            f"Risk group column {risk_group_column!r} must also be registered in --group-columns"
        )
    effective_claim_scope = args.claim_scope
    if event_audit["observed_cross_event"]:
        effective_claim_scope = "cross_event_stress_test"
    protocol_audit_complete = bool(
        split_audit["audit_complete"]
        and event_audit["audit_complete"]
        and policy_identity_audit["audit_complete"]
        and risk_group_column is not None
    )

    policy_frames = (calibration, final_test)
    prediction_column = _common_column(
        policy_frames, args.prediction_col, PREDICTION_CANDIDATES
    )
    if any(args.target_col not in frame.columns for frame in policy_frames):
        raise ValueError(f"target column {args.target_col!r} must exist in calibration and test")
    score_spec = _resolve_score_spec(policy_frames, args.risk_score_col, args.score_transform)
    calibration_scores = _extract_scores(calibration, score_spec)
    test_scores = _extract_scores(final_test, score_spec)

    severe_labels = _comma_values(args.severe_labels)
    label_order = _comma_values(args.label_order) or None
    if args.loss == "severe_miss" and not severe_labels:
        raise ValueError("--severe-labels is required for severe_miss")
    cost_matrix = _load_json_argument(args.cost_matrix_json)
    if args.loss == "cost_weighted" and cost_matrix is None:
        raise ValueError("--cost-matrix-json is required for cost_weighted")
    registered_matrix_max = _maximum_registered_cost(cost_matrix)
    effective_max_cost = args.max_cost if args.max_cost is not None else registered_matrix_max
    loss_kwargs = {
        "loss_name": args.loss,
        "severe_classes": severe_labels or None,
        "extreme_distance": args.extreme_distance,
        "label_order": label_order,
        "cost_matrix": cost_matrix,
        "max_cost": args.max_cost,
    }
    calibration_losses = bounded_classification_loss(
        calibration[args.target_col].to_numpy(),
        calibration[prediction_column].to_numpy(),
        **loss_kwargs,
    )
    test_losses = bounded_classification_loss(
        final_test[args.target_col].to_numpy(),
        final_test[prediction_column].to_numpy(),
        **loss_kwargs,
    )
    if args.loss == "severe_miss":
        # The primary severe-miss risk is the false-negative rate conditional
        # on a severe/destroyed target, not prevalence-diluted population
        # incidence.  Non-severe accepted samples still count toward coverage
        # but not toward the risk denominator.
        calibration_risk_denominator = critical_target_mask(
            calibration[args.target_col].to_numpy(), severe_labels
        )
        test_risk_denominator = critical_target_mask(
            final_test[args.target_col].to_numpy(), severe_labels
        )
        risk_denominator_name = "accepted_severe_targets"
    else:
        calibration_risk_denominator = None
        test_risk_denominator = None
        risk_denominator_name = "all_accepted_samples"
    loss_is_binary = args.loss in {"severe_miss", "extreme_error"}
    thresholds = _parse_threshold_grid(args.threshold_grid)
    calibration_result = calibrate_threshold(
        calibration_scores,
        calibration_losses,
        threshold_grid=thresholds,
        alpha=args.alpha,
        delta=args.delta,
        bound_method=args.bound_method,
        loss_is_binary=loss_is_binary,
        risk_denominator_mask=calibration_risk_denominator,
        risk_group_ids=(
            calibration[risk_group_column].to_numpy()
            if risk_group_column is not None
            else None
        ),
    )
    test_result = evaluate_locked_threshold(
        calibration_result,
        test_scores,
        test_losses,
        claim_scope=effective_claim_scope,
        protocol_audit_complete=protocol_audit_complete,
        risk_denominator_mask=test_risk_denominator,
        risk_group_ids=(
            final_test[risk_group_column].to_numpy()
            if risk_group_column is not None
            else None
        ),
    )

    risk_coverage, aurc = risk_coverage_curve(
        test_scores,
        test_losses,
        risk_denominator_mask=test_risk_denominator,
    )
    risk_coverage["loss_name"] = args.loss
    risk_coverage["protocol_role"] = "final_test"
    risk_coverage["locked_threshold"] = calibration_result.selected_threshold

    recall_target = args.recall_target
    if recall_target == "auto":
        recall_target = "severe_target" if severe_labels else "loss_event"
    if recall_target == "severe_target":
        if not severe_labels:
            raise ValueError("severe_target recall requires --severe-labels")
        relevant = critical_target_mask(final_test[args.target_col].to_numpy(), severe_labels)
    elif recall_target == "classification_error":
        relevant = _classification_error(
            final_test[args.target_col], final_test[prediction_column]
        )
    else:
        relevant = test_losses > 0.0
    review_budgets = _parse_budgets(args.review_budgets)
    recall = recall_at_review_budgets(test_scores, relevant, budgets=review_budgets)
    recall["recall_target"] = recall_target
    recall["protocol_role"] = "final_test"

    decision_source = (
        _infer_decision_source(prediction_column)
        if args.decision_source == "auto"
        else args.decision_source
    )
    decisions = build_per_sample_decisions(
        final_test,
        test_scores,
        test_result,
        losses=test_losses,
        decision_source=decision_source,
        review_disposition=args.review_disposition,
        acquisition_target=args.acquisition_target,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_result.threshold_table.to_csv(
        output_dir / "calibration_threshold_grid.csv", index=False
    )
    risk_coverage.to_csv(output_dir / "risk_coverage.csv", index=False)
    recall.to_csv(output_dir / "recall_at_review_budgets.csv", index=False)
    decisions.to_csv(output_dir / "per_sample_decision.csv", index=False)
    _write_json(split_audit, output_dir / "split_audit.json")

    recall_summary = {
        str(int(row["budget_percent"]) if float(row["budget_percent"]).is_integer()
            else row["budget_percent"]): row["recall"]
        for _, row in recall.iterrows()
    }
    calibration_summary = calibration_result.summary()
    test_summary = test_result.summary()
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": test_result.status,
        "status_reason": test_result.status_reason,
        "selected_threshold": calibration_result.selected_threshold,
        "selected_count": calibration_result.selected_count,
        "selected_upper_bound": calibration_result.upper_bound,
        "protocol_roles": {
            "gate_fit": {"path": str(gate_path), "count": len(gate_fit)},
            "risk_calibration": {"path": str(calibration_path), "count": len(calibration)},
            "final_test": {"path": str(test_path), "count": len(final_test)},
        },
        "split_audit": split_audit,
        "event_audit": event_audit,
        "policy_identity_audit": policy_identity_audit,
        "policy": {
            "prediction_column": prediction_column,
            "risk_score_source": score_spec["source"],
            "score_convention": "higher_is_reviewed; accept_if_score_lte_threshold",
            "threshold_grid": thresholds.tolist(),
            "alpha": args.alpha,
            "delta": args.delta,
            "bound_method_requested": args.bound_method,
        },
        "loss": {
            "name": args.loss,
            "risk_denominator": risk_denominator_name,
            "risk_group_column": risk_group_column,
            "risk_estimand": (
                "macro_mean_loss_across_accepted_risk_groups"
                if risk_group_column is not None
                else "row_mean_loss_diagnostic_only"
            ),
            "bounded_range": [0.0, 1.0],
            "severe_labels": severe_labels,
            "label_order": label_order,
            "extreme_distance": args.extreme_distance,
            "max_cost": effective_max_cost,
            "cost_matrix_registered": cost_matrix is not None,
            "cost_matrix": cost_matrix,
        },
        "calibration": calibration_summary,
        "final_test": test_summary,
        "descriptive_metrics": {
            "aurc": aurc,
            "aurc_definition": (
                "tie-grouped right-continuous selective-risk integral weighted by "
                "coverage increments"
            ),
            "recall_target": recall_target,
            "recall_at_review_budget_percent": recall_summary,
        },
        "claim": {
            "scope_requested": args.claim_scope,
            "scope": effective_claim_scope,
            "protocol_audit_complete": protocol_audit_complete,
            "status": test_result.status,
            "reason": test_result.status_reason,
            "statement": (
                "Finite-grid calibration claim under i.i.d./exchangeable sampling, a fixed "
                "score/loss/grid, and no test-time policy tuning. Cross-event runs are stress "
                "tests and are never labeled risk-controlled by this script."
            ),
        },
        "artifacts": {
            "calibration_threshold_grid": "calibration_threshold_grid.csv",
            "risk_coverage": "risk_coverage.csv",
            "recall_at_review_budgets": "recall_at_review_budgets.csv",
            "per_sample_decision": "per_sample_decision.csv",
            "split_audit": "split_audit.json",
        },
    }
    _write_json(summary, output_dir / "summary.json")
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
