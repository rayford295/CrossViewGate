"""Finite-sample risk control for selective disaster triage.

The policy convention in this module is deliberately simple: larger risk
scores receive review and a sample is accepted when ``risk_score <=
threshold``.  A threshold is selected only from a pre-specified grid on the
risk-calibration split.  The locked threshold can then be passed to
``evaluate_locked_threshold``; that function has no threshold-search input.

For binary losses, calibration uses one-sided Clopper--Pearson bounds.  For a
general loss bounded in ``[0, 1]``, it uses a one-sided Hoeffding bound.  The
calibration bounds use ``delta / number_of_thresholds`` (Bonferroni), yielding
simultaneous finite-grid coverage under the assumptions documented in
``docs/results/selective_triage_protocol.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import ceil, exp, isfinite, lgamma, log, log1p, sqrt
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_REVIEW_BUDGETS = (5.0, 10.0, 20.0, 30.0, 50.0)
SUPPORTED_LOSSES = ("severe_miss", "extreme_error", "cost_weighted")
SUPPORTED_BOUNDS = ("auto", "clopper_pearson", "hoeffding")
SUPPORTED_CLAIM_SCOPES = ("in_event", "cross_event_stress_test")


def _as_one_dimensional(values: Sequence[object] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return array


def _validated_scores_and_losses(
    risk_scores: Sequence[float] | np.ndarray,
    losses: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = _as_one_dimensional(risk_scores, "risk_scores").astype(np.float64)
    bounded_losses = _as_one_dimensional(losses, "losses").astype(np.float64)
    if len(scores) != len(bounded_losses):
        raise ValueError("risk_scores and losses must have the same length")
    if not np.isfinite(scores).all():
        raise ValueError("risk_scores must all be finite")
    if not np.isfinite(bounded_losses).all():
        raise ValueError("losses must all be finite")
    if ((bounded_losses < 0.0) | (bounded_losses > 1.0)).any():
        raise ValueError("losses must be bounded in [0, 1]")
    return scores, bounded_losses


def _label_key(value: object) -> str:
    """Create stable keys across CSV integer/string type inference."""
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


def _label_keys(values: Sequence[object] | np.ndarray, name: str) -> np.ndarray:
    array = _as_one_dimensional(values, name)
    return np.asarray([_label_key(value) for value in array], dtype=object)


def _ordinal_positions(
    values: np.ndarray,
    *,
    label_order: Sequence[object] | None,
    name: str,
) -> np.ndarray:
    if label_order is not None:
        keys = [_label_key(value) for value in label_order]
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("label_order must contain unique labels")
        mapping = {key: index for index, key in enumerate(keys)}
        missing = sorted(set(values) - set(mapping))
        if missing:
            raise ValueError(f"{name} contains labels absent from label_order: {missing[:5]}")
        return np.asarray([mapping[value] for value in values], dtype=np.int64)
    try:
        numeric = values.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Non-numeric ordinal labels require label_order") from error
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain finite integer class indices")
    return numeric.astype(np.int64)


def critical_target_mask(
    targets: Sequence[object] | np.ndarray,
    severe_classes: Sequence[object],
) -> np.ndarray:
    """Return the pre-registered severe/destroyed target indicator."""
    severe = {_label_key(value) for value in severe_classes}
    if not severe:
        raise ValueError("severe_classes must not be empty")
    target_keys = _label_keys(targets, "targets")
    return np.asarray([value in severe for value in target_keys], dtype=bool)


def _cost_from_mapping(
    target_keys: np.ndarray,
    prediction_keys: np.ndarray,
    cost_matrix: Mapping[object, Mapping[object, float]],
) -> tuple[np.ndarray, float]:
    normalized: dict[str, dict[str, float]] = {}
    for target, prediction_costs in cost_matrix.items():
        if not isinstance(prediction_costs, Mapping):
            raise ValueError("A mapping cost_matrix must map targets to prediction-cost mappings")
        normalized[_label_key(target)] = {
            _label_key(prediction): float(cost)
            for prediction, cost in prediction_costs.items()
        }
    labels = set(normalized)
    if not labels or any(set(row) != labels for row in normalized.values()):
        raise ValueError(
            "A mapping cost_matrix must be complete and square over the same label set"
        )
    all_costs = np.asarray(
        [cost for row in normalized.values() for cost in row.values()], dtype=np.float64
    )
    if not np.isfinite(all_costs).all() or (all_costs < 0.0).any():
        raise ValueError("cost_matrix entries must be finite and non-negative")
    costs: list[float] = []
    for target, prediction in zip(target_keys, prediction_keys):
        if target not in normalized or prediction not in normalized[target]:
            raise ValueError(
                f"cost_matrix has no entry for target={target}, prediction={prediction}"
            )
        costs.append(normalized[target][prediction])
    return np.asarray(costs, dtype=np.float64), float(all_costs.max())


def _cost_from_array(
    target_keys: np.ndarray,
    prediction_keys: np.ndarray,
    cost_matrix: Sequence[Sequence[float]] | np.ndarray,
    label_order: Sequence[object] | None,
) -> tuple[np.ndarray, float]:
    matrix = np.asarray(cost_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("cost_matrix must be a non-empty square matrix")
    if not np.isfinite(matrix).all() or (matrix < 0.0).any():
        raise ValueError("cost_matrix entries must be finite and non-negative")
    if label_order is None:
        order: Sequence[object] = list(range(matrix.shape[0]))
    else:
        order = label_order
    keys = [_label_key(value) for value in order]
    if len(keys) != matrix.shape[0] or len(set(keys)) != len(keys):
        raise ValueError("label_order must uniquely name every cost_matrix row/column")
    mapping = {key: index for index, key in enumerate(keys)}
    missing = sorted((set(target_keys) | set(prediction_keys)) - set(mapping))
    if missing:
        raise ValueError(f"Labels absent from cost_matrix label_order: {missing[:5]}")
    target_indices = np.asarray([mapping[value] for value in target_keys], dtype=np.int64)
    prediction_indices = np.asarray([mapping[value] for value in prediction_keys], dtype=np.int64)
    return matrix[target_indices, prediction_indices], float(matrix.max())


def bounded_classification_loss(
    targets: Sequence[object] | np.ndarray,
    predictions: Sequence[object] | np.ndarray,
    *,
    loss_name: str,
    severe_classes: Sequence[object] | None = None,
    extreme_distance: int = 2,
    label_order: Sequence[object] | None = None,
    cost_matrix: Mapping[object, Mapping[object, float]]
    | Sequence[Sequence[float]]
    | np.ndarray
    | None = None,
    max_cost: float | None = None,
) -> np.ndarray:
    """Compute a pre-specified classification loss bounded in ``[0, 1]``.

    ``severe_miss`` is one when a severe/destroyed target is predicted outside
    the registered severe set. ``extreme_error`` is one when ordinal class
    distance is at least ``extreme_distance``. ``cost_weighted`` looks up a
    registered cost and divides it by ``max_cost`` (or by the maximum of the
    complete supplied matrix). No clipping is performed: an invalid bound is
    rejected so the finite-sample claim cannot silently become optimistic.
    """
    if loss_name not in SUPPORTED_LOSSES:
        raise ValueError(f"loss_name must be one of {SUPPORTED_LOSSES}")
    target_keys = _label_keys(targets, "targets")
    prediction_keys = _label_keys(predictions, "predictions")
    if len(target_keys) != len(prediction_keys):
        raise ValueError("targets and predictions must have the same length")

    if loss_name == "severe_miss":
        if not severe_classes:
            raise ValueError("severe_miss requires severe_classes")
        severe = {_label_key(value) for value in severe_classes}
        return np.asarray(
            [target in severe and prediction not in severe
             for target, prediction in zip(target_keys, prediction_keys)],
            dtype=np.float64,
        )

    if loss_name == "extreme_error":
        if not isinstance(extreme_distance, (int, np.integer)) or extreme_distance < 1:
            raise ValueError("extreme_distance must be a positive integer")
        target_positions = _ordinal_positions(
            target_keys, label_order=label_order, name="targets"
        )
        prediction_positions = _ordinal_positions(
            prediction_keys, label_order=label_order, name="predictions"
        )
        return (np.abs(target_positions - prediction_positions) >= extreme_distance).astype(
            np.float64
        )

    if cost_matrix is None:
        raise ValueError("cost_weighted requires a complete cost_matrix")
    if isinstance(cost_matrix, Mapping):
        raw_costs, matrix_max = _cost_from_mapping(
            target_keys, prediction_keys, cost_matrix
        )
    else:
        raw_costs, matrix_max = _cost_from_array(
            target_keys, prediction_keys, cost_matrix, label_order
        )
    if not np.isfinite(raw_costs).all() or (raw_costs < 0.0).any():
        raise ValueError("cost_matrix entries must be finite and non-negative")
    bound = float(max_cost) if max_cost is not None else matrix_max
    if not isfinite(bound) or bound <= 0.0:
        raise ValueError("max_cost must be finite and positive")
    if matrix_max > bound + 1e-12 or (raw_costs > bound + 1e-12).any():
        raise ValueError("max_cost is smaller than an entry in cost_matrix")
    return raw_costs / bound


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    """Numerically evaluate P[Binomial(n, probability) <= k]."""
    if k < 0:
        return 0.0
    if k >= n or probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    log_term = (
        lgamma(n + 1)
        - lgamma(k + 1)
        - lgamma(n - k + 1)
        + k * log(probability)
        + (n - k) * log1p(-probability)
    )
    term = exp(log_term) if log_term > -745.0 else 0.0
    total = term
    odds_inverse = (1.0 - probability) / probability
    for index in range(k, 0, -1):
        term *= (index / (n - index + 1.0)) * odds_inverse
        total += term
        if term == 0.0:
            break
    return float(min(1.0, total))


def _clopper_pearson_upper(successes: int, count: int, delta: float) -> float:
    if count <= 0:
        raise ValueError("At least one observation is required")
    if successes < 0 or successes > count:
        raise ValueError("successes must lie between zero and count")
    if successes == count:
        return 1.0
    if successes == 0:
        return float(1.0 - delta ** (1.0 / count))
    lower = successes / count
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if _binomial_cdf(successes, count, midpoint) > delta:
            lower = midpoint
        else:
            upper = midpoint
    return float(upper)


def _resolved_bound_method(method: str, loss_is_binary: bool) -> str:
    if method not in SUPPORTED_BOUNDS:
        raise ValueError(f"method must be one of {SUPPORTED_BOUNDS}")
    if method == "auto":
        return "clopper_pearson" if loss_is_binary else "hoeffding"
    if method == "clopper_pearson" and not loss_is_binary:
        raise ValueError("Clopper-Pearson is valid only for a pre-specified binary loss")
    return method


def one_sided_risk_upper_bound(
    losses: Sequence[float] | np.ndarray,
    *,
    delta: float,
    method: str = "hoeffding",
    loss_is_binary: bool = False,
) -> float:
    """Return a finite-sample one-sided upper confidence bound on mean loss.

    The caller must pass an already multiplicity-adjusted ``delta`` when this
    function is used during threshold search. ``calibrate_threshold`` performs
    that adjustment automatically.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    bounded_losses = _as_one_dimensional(losses, "losses").astype(np.float64)
    if len(bounded_losses) == 0:
        raise ValueError("At least one loss is required for a risk bound")
    if not np.isfinite(bounded_losses).all() or (
        (bounded_losses < 0.0) | (bounded_losses > 1.0)
    ).any():
        raise ValueError("losses must be finite and bounded in [0, 1]")
    resolved = _resolved_bound_method(method, loss_is_binary)
    empirical_risk = float(bounded_losses.mean())
    if resolved == "clopper_pearson":
        if not np.isin(bounded_losses, (0.0, 1.0)).all():
            raise ValueError("Clopper-Pearson losses must be exactly binary")
        return _clopper_pearson_upper(int(bounded_losses.sum()), len(bounded_losses), delta)
    radius = sqrt(log(1.0 / delta) / (2.0 * len(bounded_losses)))
    return float(min(1.0, empirical_risk + radius))


@dataclass(frozen=True)
class CalibrationResult:
    """A threshold selected exclusively on the risk-calibration split."""

    selected_threshold: float | None
    selected_count: int
    risk_denominator_count: int
    risk_group_count: int
    calibration_count: int
    empirical_risk: float | None
    row_empirical_risk: float | None
    upper_bound: float | None
    alpha: float
    delta: float
    effective_delta: float
    grid_size: int
    bound_method: str
    loss_is_binary: bool
    grouping_required: bool
    status: str
    status_reason: str
    threshold_table: pd.DataFrame

    @property
    def coverage(self) -> float:
        if self.calibration_count == 0:
            return 0.0
        return self.selected_count / self.calibration_count

    def summary(self) -> dict[str, object]:
        return {
            "selected_threshold": self.selected_threshold,
            "selected_count": self.selected_count,
            "risk_denominator_count": self.risk_denominator_count,
            "risk_group_count": self.risk_group_count,
            "calibration_count": self.calibration_count,
            "coverage": self.coverage,
            "empirical_risk": self.empirical_risk,
            "row_empirical_risk": self.row_empirical_risk,
            "upper_bound": self.upper_bound,
            "alpha": self.alpha,
            "delta": self.delta,
            "effective_delta": self.effective_delta,
            "grid_size": self.grid_size,
            "bound_method": self.bound_method,
            "loss_is_binary": self.loss_is_binary,
            "grouping_required": self.grouping_required,
            "status": self.status,
            "status_reason": self.status_reason,
        }


def calibrate_threshold(
    risk_scores: Sequence[float] | np.ndarray,
    losses: Sequence[float] | np.ndarray,
    *,
    threshold_grid: Sequence[float] | np.ndarray,
    alpha: float,
    delta: float,
    bound_method: str = "auto",
    loss_is_binary: bool = False,
    risk_denominator_mask: Sequence[bool] | np.ndarray | None = None,
    risk_group_ids: Sequence[object] | np.ndarray | None = None,
) -> CalibrationResult:
    """Select maximum accepted coverage using calibration labels only.

    Every threshold receives a bound at ``delta / m`` for ``m`` registered
    thresholds.  The selected threshold has maximum accepted count among those
    whose simultaneous upper bound is at most ``alpha``.  A tie in count is
    resolved toward the smaller threshold.  If no non-empty threshold passes,
    no automated decisions are accepted and the result is ``risk-aware``.
    """
    scores, bounded_losses = _validated_scores_and_losses(risk_scores, losses)
    if risk_denominator_mask is None:
        denominator = np.ones(len(scores), dtype=bool)
    else:
        denominator = _as_one_dimensional(
            risk_denominator_mask, "risk_denominator_mask"
        ).astype(bool)
        if len(denominator) != len(scores):
            raise ValueError("risk_denominator_mask must match risk_scores")
    if risk_group_ids is None:
        groups = None
    else:
        raw_groups = _as_one_dimensional(risk_group_ids, "risk_group_ids")
        if len(raw_groups) != len(scores):
            raise ValueError("risk_group_ids must match risk_scores")
        groups = np.asarray(
            [_normalized_identifier(value) for value in raw_groups], dtype=object
        )
    if len(scores) == 0:
        raise ValueError("The risk-calibration split must not be empty")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    thresholds = _as_one_dimensional(threshold_grid, "threshold_grid").astype(np.float64)
    if len(thresholds) == 0 or not np.isfinite(thresholds).all():
        raise ValueError("threshold_grid must contain finite values")
    if len(np.unique(thresholds)) != len(thresholds):
        raise ValueError("threshold_grid values must be unique")
    thresholds = np.sort(thresholds)
    effective_binary = loss_is_binary and groups is None
    resolved_method = _resolved_bound_method(bound_method, effective_binary)
    effective_delta = delta / len(thresholds)
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        accepted = scores <= threshold
        accepted_count = int(accepted.sum())
        controlled = accepted & denominator
        risk_denominator_count = int(controlled.sum())
        if risk_denominator_count:
            selected_losses = bounded_losses[controlled]
            if groups is not None:
                grouped_losses = pd.DataFrame(
                    {"group": groups[controlled], "loss": selected_losses}
                ).groupby("group", sort=True)["loss"].mean().to_numpy(dtype=np.float64)
            else:
                grouped_losses = selected_losses
            risk_group_count = len(grouped_losses)
            row_empirical_risk = float(selected_losses.mean())
            empirical_risk = float(grouped_losses.mean())
            upper_bound = one_sided_risk_upper_bound(
                grouped_losses,
                delta=effective_delta,
                method=resolved_method,
                loss_is_binary=effective_binary,
            )
            passes = upper_bound <= alpha
        else:
            risk_group_count = 0
            empirical_risk = None
            row_empirical_risk = None
            upper_bound = None
            passes = False
        rows.append(
            {
                "threshold": float(threshold),
                "selected_count": accepted_count,
                "risk_denominator_count": risk_denominator_count,
                "risk_group_count": risk_group_count,
                "coverage": accepted_count / len(scores),
                "empirical_risk": empirical_risk,
                "row_empirical_risk": row_empirical_risk,
                "upper_bound": upper_bound,
                "alpha": float(alpha),
                "passes_alpha": bool(passes),
                "effective_delta": effective_delta,
                "bound_method": resolved_method,
            }
        )
    table = pd.DataFrame(rows)
    passing = table[table["passes_alpha"]]
    if passing.empty:
        table["selected"] = False
        return CalibrationResult(
            selected_threshold=None,
            selected_count=0,
            risk_denominator_count=0,
            risk_group_count=0,
            calibration_count=len(scores),
            empirical_risk=None,
            row_empirical_risk=None,
            upper_bound=None,
            alpha=float(alpha),
            delta=float(delta),
            effective_delta=effective_delta,
            grid_size=len(thresholds),
            bound_method=resolved_method,
            loss_is_binary=effective_binary,
            grouping_required=groups is not None,
            status="risk-aware",
            status_reason="no_nonempty_threshold_passed_the_simultaneous_calibration_bound",
            threshold_table=table,
        )
    best = passing.sort_values(
        ["selected_count", "threshold"], ascending=[False, True]
    ).iloc[0]
    selected_threshold = float(best["threshold"])
    table["selected"] = table["threshold"] == selected_threshold
    return CalibrationResult(
        selected_threshold=selected_threshold,
        selected_count=int(best["selected_count"]),
        risk_denominator_count=int(best["risk_denominator_count"]),
        risk_group_count=int(best["risk_group_count"]),
        calibration_count=len(scores),
        empirical_risk=float(best["empirical_risk"]),
        row_empirical_risk=float(best["row_empirical_risk"]),
        upper_bound=float(best["upper_bound"]),
        alpha=float(alpha),
        delta=float(delta),
        effective_delta=effective_delta,
        grid_size=len(thresholds),
        bound_method=resolved_method,
        loss_is_binary=effective_binary,
        grouping_required=groups is not None,
        status="risk-controlled",
        status_reason="calibration_upper_bound_at_or_below_alpha",
        threshold_table=table,
    )


@dataclass(frozen=True)
class LockedTestResult:
    """Held-out evaluation of one calibration-locked threshold."""

    selected_threshold: float | None
    selected_count: int
    risk_denominator_count: int
    risk_group_count: int
    test_count: int
    empirical_risk: float | None
    row_empirical_risk: float | None
    upper_bound: float | None
    alpha: float
    delta: float
    bound_method: str
    claim_scope: str
    protocol_audit_complete: bool
    status: str
    status_reason: str

    @property
    def coverage(self) -> float:
        if self.test_count == 0:
            return 0.0
        return self.selected_count / self.test_count

    def summary(self) -> dict[str, object]:
        return {
            "selected_threshold": self.selected_threshold,
            "selected_count": self.selected_count,
            "risk_denominator_count": self.risk_denominator_count,
            "risk_group_count": self.risk_group_count,
            "test_count": self.test_count,
            "coverage": self.coverage,
            "empirical_risk": self.empirical_risk,
            "row_empirical_risk": self.row_empirical_risk,
            "upper_bound": self.upper_bound,
            "alpha": self.alpha,
            "delta": self.delta,
            "bound_method": self.bound_method,
            "claim_scope": self.claim_scope,
            "protocol_audit_complete": self.protocol_audit_complete,
            "status": self.status,
            "status_reason": self.status_reason,
        }


def evaluate_locked_threshold(
    calibration: CalibrationResult,
    risk_scores: Sequence[float] | np.ndarray,
    losses: Sequence[float] | np.ndarray,
    *,
    protocol_audit_complete: bool,
    claim_scope: str = "in_event",
    risk_denominator_mask: Sequence[bool] | np.ndarray | None = None,
    risk_group_ids: Sequence[object] | np.ndarray | None = None,
) -> LockedTestResult:
    """Evaluate the calibration-locked policy once on final-test labels.

    There is intentionally no threshold or grid argument here.  The held-out
    upper bound is a diagnostic at one locked threshold and is never used to
    choose another policy.  Per the project claim rule, a failing diagnostic or
    a cross-event stress test is reported as ``risk-aware``.
    """
    if claim_scope not in SUPPORTED_CLAIM_SCOPES:
        raise ValueError(f"claim_scope must be one of {SUPPORTED_CLAIM_SCOPES}")
    scores, bounded_losses = _validated_scores_and_losses(risk_scores, losses)
    if risk_denominator_mask is None:
        denominator = np.ones(len(scores), dtype=bool)
    else:
        denominator = _as_one_dimensional(
            risk_denominator_mask, "risk_denominator_mask"
        ).astype(bool)
        if len(denominator) != len(scores):
            raise ValueError("risk_denominator_mask must match risk_scores")
    if risk_group_ids is None:
        groups = None
    else:
        raw_groups = _as_one_dimensional(risk_group_ids, "risk_group_ids")
        if len(raw_groups) != len(scores):
            raise ValueError("risk_group_ids must match risk_scores")
        groups = np.asarray(
            [_normalized_identifier(value) for value in raw_groups], dtype=object
        )
    if calibration.grouping_required != (groups is not None):
        requirement = "requires" if calibration.grouping_required else "forbids"
        raise ValueError(
            f"Locked calibration {requirement} risk_group_ids; calibration/test "
            "risk estimands must use the same grouping rule"
        )
    if len(scores) == 0:
        raise ValueError("The final-test split must not be empty")
    if calibration.selected_threshold is None:
        accepted = np.zeros(len(scores), dtype=bool)
    else:
        accepted = scores <= calibration.selected_threshold
    selected_count = int(accepted.sum())
    controlled = accepted & denominator
    risk_denominator_count = int(controlled.sum())
    if risk_denominator_count:
        selected_losses = bounded_losses[controlled]
        if groups is not None:
            grouped_losses = pd.DataFrame(
                {"group": groups[controlled], "loss": selected_losses}
            ).groupby("group", sort=True)["loss"].mean().to_numpy(dtype=np.float64)
        else:
            grouped_losses = selected_losses
        risk_group_count = len(grouped_losses)
        row_empirical_risk: float | None = float(selected_losses.mean())
        empirical_risk: float | None = float(grouped_losses.mean())
        upper_bound: float | None = one_sided_risk_upper_bound(
            grouped_losses,
            delta=calibration.delta,
            method=calibration.bound_method,
            loss_is_binary=calibration.loss_is_binary,
        )
    else:
        risk_group_count = 0
        empirical_risk = None
        row_empirical_risk = None
        upper_bound = None

    if claim_scope == "cross_event_stress_test":
        status = "risk-aware"
        reason = "cross_event_stress_test_is_outside_the_in_event_exchangeability_claim"
    elif not protocol_audit_complete:
        status = "risk-aware"
        reason = "group_or_event_protocol_audit_is_incomplete"
    elif calibration.status != "risk-controlled":
        status = "risk-aware"
        reason = "calibration_did_not_certify_a_threshold"
    elif selected_count == 0:
        status = "risk-aware"
        reason = "no_final_test_samples_were_accepted"
    elif risk_denominator_count == 0:
        status = "risk-aware"
        reason = "no_final_test_risk_denominator_samples_were_accepted"
    elif upper_bound is None or upper_bound > calibration.alpha:
        status = "risk-aware"
        reason = "held_out_upper_bound_exceeds_alpha"
    else:
        status = "risk-controlled"
        reason = "calibration_certified_and_held_out_upper_bound_at_or_below_alpha"
    return LockedTestResult(
        selected_threshold=calibration.selected_threshold,
        selected_count=selected_count,
        risk_denominator_count=risk_denominator_count,
        risk_group_count=risk_group_count,
        test_count=len(scores),
        empirical_risk=empirical_risk,
        row_empirical_risk=row_empirical_risk,
        upper_bound=upper_bound,
        alpha=calibration.alpha,
        delta=calibration.delta,
        bound_method=calibration.bound_method,
        claim_scope=claim_scope,
        protocol_audit_complete=bool(protocol_audit_complete),
        status=status,
        status_reason=reason,
    )


def risk_coverage_curve(
    risk_scores: Sequence[float] | np.ndarray,
    losses: Sequence[float] | np.ndarray,
    *,
    risk_denominator_mask: Sequence[bool] | np.ndarray | None = None,
) -> tuple[pd.DataFrame, float]:
    """Return a threshold-achievable, tie-invariant risk-coverage curve.

    Samples with an identical score enter coverage together.  The reported
    AURC is a right-continuous discrete integral weighted by each score group's
    coverage increment.  It is descriptive and does not participate in
    threshold calibration.
    """
    scores, bounded_losses = _validated_scores_and_losses(risk_scores, losses)
    if len(scores) == 0:
        raise ValueError("At least one sample is required")
    if risk_denominator_mask is None:
        denominator = np.ones(len(scores), dtype=bool)
    else:
        denominator = _as_one_dimensional(
            risk_denominator_mask, "risk_denominator_mask"
        ).astype(bool)
        if len(denominator) != len(scores):
            raise ValueError("risk_denominator_mask must match risk_scores")
    order = np.argsort(scores, kind="mergesort")
    ordered_scores = scores[order]
    cumulative_loss = np.cumsum(bounded_losses[order])
    cumulative_denominator = np.cumsum(denominator[order])
    group_ends = np.flatnonzero(
        np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    )
    accepted_count = group_ends + 1
    denominator_count = cumulative_denominator[group_ends]
    selective_risk = np.divide(
        cumulative_loss[group_ends],
        denominator_count,
        out=np.full(len(group_ends), np.nan, dtype=np.float64),
        where=denominator_count > 0,
    )
    group_sizes = np.diff(np.r_[0, accepted_count])
    curve = pd.DataFrame(
        {
            "accepted_count": accepted_count,
            "review_count": len(scores) - accepted_count,
            "coverage": accepted_count / len(scores),
            "empirical_risk": selective_risk,
            "risk_denominator_count": denominator_count,
            "risk_score_threshold": ordered_scores[group_ends],
            "score_tie_group_size": group_sizes,
        }
    )
    finite = np.isfinite(selective_risk)
    aurc = (
        float(np.sum(selective_risk[finite] * group_sizes[finite]) / group_sizes[finite].sum())
        if finite.any()
        else float("nan")
    )
    return curve, aurc


def recall_at_review_budgets(
    risk_scores: Sequence[float] | np.ndarray,
    relevant_mask: Sequence[bool] | np.ndarray,
    *,
    budgets: Iterable[float] = DEFAULT_REVIEW_BUDGETS,
) -> pd.DataFrame:
    """Measure discovery recall when reviewing the highest-risk samples."""
    scores = _as_one_dimensional(risk_scores, "risk_scores").astype(np.float64)
    relevant = _as_one_dimensional(relevant_mask, "relevant_mask").astype(bool)
    if len(scores) != len(relevant) or len(scores) == 0:
        raise ValueError("risk_scores and relevant_mask must have the same non-zero length")
    if not np.isfinite(scores).all():
        raise ValueError("risk_scores must all be finite")
    budget_values = [float(value) for value in budgets]
    if not budget_values or any(not 0.0 < value <= 100.0 for value in budget_values):
        raise ValueError("review budgets must lie in (0, 100]")
    order = np.argsort(-scores, kind="mergesort")
    relevant_total = int(relevant.sum())
    rows: list[dict[str, object]] = []
    for budget in budget_values:
        requested_review_count = min(
            len(scores), int(ceil(len(scores) * budget / 100.0))
        )
        boundary_score = scores[order[requested_review_count - 1]]
        # A score-threshold policy cannot split a tie group.  Include the full
        # boundary group and report the realized budget explicitly.
        reviewed = scores >= boundary_score
        review_count = int(reviewed.sum())
        relevant_reviewed = int(relevant[reviewed].sum())
        rows.append(
            {
                "budget_percent": budget,
                "requested_review_count": requested_review_count,
                "review_count": review_count,
                "realized_budget_percent": 100.0 * review_count / len(scores),
                "relevant_total": relevant_total,
                "relevant_reviewed": relevant_reviewed,
                "recall": (
                    relevant_reviewed / relevant_total if relevant_total else np.nan
                ),
                "precision": relevant_reviewed / review_count if review_count else np.nan,
                "risk_score_boundary": float(boundary_score),
            }
        )
    return pd.DataFrame(rows)


def _normalized_identifier(value: object) -> str:
    if pd.isna(value):
        raise ValueError("Protocol identifiers may not be missing")
    return _label_key(value)


def _minimum_cross_distance_m(left: pd.DataFrame, right: pd.DataFrame) -> float:
    """Return the minimum WGS84 great-circle distance for two non-empty frames."""
    left_coordinates = np.radians(
        left[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    )
    right_coordinates = np.radians(
        right[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    )
    minimum = np.inf
    for start in range(0, len(left_coordinates), 512):
        block = left_coordinates[start : start + 512]
        left_lat = block[:, 0][:, None]
        left_lon = block[:, 1][:, None]
        right_lat = right_coordinates[:, 0][None, :]
        right_lon = right_coordinates[:, 1][None, :]
        delta_lat = right_lat - left_lat
        delta_lon = right_lon - left_lon
        haversine = (
            np.sin(delta_lat / 2.0) ** 2
            + np.cos(left_lat) * np.cos(right_lat) * np.sin(delta_lon / 2.0) ** 2
        )
        distances = 2.0 * 6_371_008.8 * np.arcsin(
            np.sqrt(np.clip(haversine, 0.0, 1.0))
        )
        minimum = min(minimum, float(distances.min()))
    return minimum


def assert_disjoint_protocol_splits(
    gate_fit: pd.DataFrame,
    risk_calibration: pd.DataFrame,
    final_test: pd.DataFrame,
    *,
    id_columns: Sequence[str] | None = None,
    group_columns: Sequence[str] | None = None,
    spatial_threshold_m: float | None = None,
) -> dict[str, object]:
    """Audit record, group, and optional coordinate isolation across roles.

    Record identifiers must be unique within each role.  Group identifiers may
    repeat within a role (multiple samples per property/sequence/tile) but may
    never cross roles.  When group columns are inferred and none are present,
    the returned audit is explicitly incomplete so callers cannot treat mere
    row-id separation as a full protocol audit.
    """
    frames = {
        "gate_fit": gate_fit,
        "risk_calibration": risk_calibration,
        "final_test": final_test,
    }
    if any(frame.empty for frame in frames.values()):
        empty = [name for name, frame in frames.items() if frame.empty]
        raise ValueError(f"Protocol splits must be non-empty; empty roles: {empty}")
    if id_columns is None:
        if all("sample_id" in frame.columns for frame in frames.values()):
            columns = ["sample_id"]
            if all("dataset" in frame.columns for frame in frames.values()):
                columns.insert(0, "dataset")
        elif all("objectid" in frame.columns for frame in frames.values()):
            columns = ["objectid"]
            if all("dataset" in frame.columns for frame in frames.values()):
                columns.insert(0, "dataset")
        else:
            raise ValueError(
                "Could not infer protocol identifiers; provide id_columns shared by all splits"
            )
    else:
        columns = [str(column).strip() for column in id_columns if str(column).strip()]
        if not columns:
            raise ValueError("id_columns must not be empty")
    forbidden = {"split", "role", "protocol_role", "subset"}
    if any(column.lower() in forbidden for column in columns):
        raise ValueError("Split/role columns cannot serve as disjointness identifiers")

    key_sets: dict[str, set[tuple[str, ...]]] = {}
    for role, frame in frames.items():
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{role} is missing identifier columns: {missing}")
        keys = [
            tuple(_normalized_identifier(value) for value in row)
            for row in frame[columns].itertuples(index=False, name=None)
        ]
        if len(set(keys)) != len(keys):
            duplicates = pd.Series(keys).value_counts()
            examples = [list(key) for key in duplicates[duplicates > 1].index[:5]]
            raise ValueError(f"{role} has duplicate protocol identifiers: {examples}")
        key_sets[role] = set(keys)

    overlaps: dict[str, int] = {}
    role_pairs = (
        ("gate_fit", "risk_calibration"),
        ("gate_fit", "final_test"),
        ("risk_calibration", "final_test"),
    )
    overlap_examples: dict[str, list[list[str]]] = {}
    for left, right in role_pairs:
        name = f"{left}__{right}"
        shared = key_sets[left] & key_sets[right]
        overlaps[name] = len(shared)
        if shared:
            overlap_examples[name] = [list(key) for key in sorted(shared)[:5]]
    if any(overlaps.values()):
        raise ValueError(
            "Protocol roles overlap on registered identifiers: "
            + ", ".join(f"{name}={count}" for name, count in overlaps.items() if count)
        )

    if group_columns is None:
        group_candidates = (
            "sequence_id",
            "spatial_block_id",
            "tile_id",
            "group_id",
            "objectid",
        )
        resolved_group_columns = [
            column
            for column in group_candidates
            if all(column in frame.columns for frame in frames.values())
        ]
    else:
        resolved_group_columns = [
            str(column).strip() for column in group_columns if str(column).strip()
        ]
        if not resolved_group_columns:
            raise ValueError("group_columns must not be empty when explicitly supplied")

    group_overlap_counts: dict[str, dict[str, int]] = {}
    group_overlap_examples: dict[str, dict[str, list[str]]] = {}
    for column in resolved_group_columns:
        missing = [name for name, frame in frames.items() if column not in frame.columns]
        if missing:
            raise ValueError(f"Group column {column!r} is missing from roles: {missing}")
        value_sets: dict[str, set[str]] = {}
        for role, frame in frames.items():
            if frame[column].isna().any():
                raise ValueError(f"{role}.{column} contains missing group identifiers")
            value_sets[role] = {
                _normalized_identifier(value) for value in frame[column].to_numpy()
            }
        column_counts: dict[str, int] = {}
        column_examples: dict[str, list[str]] = {}
        for left, right in role_pairs:
            name = f"{left}__{right}"
            shared = value_sets[left] & value_sets[right]
            column_counts[name] = len(shared)
            if shared:
                column_examples[name] = sorted(shared)[:5]
        if any(column_counts.values()):
            raise ValueError(
                f"Protocol roles overlap on group column {column!r}: "
                + ", ".join(
                    f"{name}={count}" for name, count in column_counts.items() if count
                )
            )
        group_overlap_counts[column] = column_counts
        group_overlap_examples[column] = column_examples

    spatial_audit: dict[str, object] = {
        "requested": spatial_threshold_m is not None,
        "threshold_m": spatial_threshold_m,
        "pair_minimum_distance_m": {},
    }
    if spatial_threshold_m is not None:
        if not isfinite(spatial_threshold_m) or spatial_threshold_m < 0.0:
            raise ValueError("spatial_threshold_m must be finite and non-negative")
        for role, frame in frames.items():
            missing = [column for column in ("latitude", "longitude") if column not in frame]
            if missing:
                raise ValueError(f"{role} is missing spatial audit columns: {missing}")
            coordinates = frame[["latitude", "longitude"]].apply(
                pd.to_numeric, errors="raise"
            ).to_numpy(dtype=np.float64)
            if not np.isfinite(coordinates).all():
                raise ValueError(f"{role} contains non-finite spatial coordinates")
            if ((coordinates[:, 0] < -90.0) | (coordinates[:, 0] > 90.0)).any() or (
                (coordinates[:, 1] < -180.0) | (coordinates[:, 1] > 180.0)
            ).any():
                raise ValueError(f"{role} contains out-of-range WGS84 coordinates")
        minimum_distances: dict[str, float] = {}
        for left, right in role_pairs:
            name = f"{left}__{right}"
            distance = _minimum_cross_distance_m(frames[left], frames[right])
            minimum_distances[name] = distance
            if distance <= spatial_threshold_m:
                raise ValueError(
                    f"Spatial protocol leakage between {left} and {right}: "
                    f"minimum {distance:.3f} m <= {spatial_threshold_m:.3f} m"
                )
        spatial_audit["pair_minimum_distance_m"] = minimum_distances

    group_audit_complete = bool(resolved_group_columns)
    spatial_audit_complete = spatial_threshold_m is not None
    return {
        "audit_complete": group_audit_complete and spatial_audit_complete,
        "disjoint": True,
        "id_columns": columns,
        "record_id_audit_complete": True,
        "group_audit_complete": group_audit_complete,
        "group_columns": resolved_group_columns,
        "spatial_audit_complete": spatial_audit_complete,
        "row_counts": {name: len(frame) for name, frame in frames.items()},
        "overlap_counts": overlaps,
        "overlap_examples": overlap_examples,
        "group_overlap_counts": group_overlap_counts,
        "group_overlap_examples": group_overlap_examples,
        "spatial_audit": spatial_audit,
    }


def build_per_sample_decisions(
    frame: pd.DataFrame,
    risk_scores: Sequence[float] | np.ndarray,
    test_result: LockedTestResult,
    *,
    losses: Sequence[float] | np.ndarray | None = None,
    decision_source: str = "gated_mixture",
    review_disposition: str = "defer_human",
    acquisition_target: str | None = None,
) -> pd.DataFrame:
    """Add the P0.7 decision skeleton to a final-test evidence table.

    Operational acceptance is deliberately fail-closed.  A sample can only be
    emitted as ``accept`` when it is below the locked threshold *and* the
    held-out result retains ``risk-controlled`` status.  Threshold eligibility
    is preserved as a separate diagnostic column so a failed held-out bound or
    cross-event stress test remains auditable without authorizing automation.
    """
    if decision_source not in {"street", "overhead", "gated_mixture"}:
        raise ValueError("decision_source must be street, overhead, or gated_mixture")
    if review_disposition not in {"defer_human", "acquire_view"}:
        raise ValueError("review_disposition must be defer_human or acquire_view")
    if review_disposition == "acquire_view" and not acquisition_target:
        raise ValueError("acquire_view decisions require acquisition_target")
    scores = _as_one_dimensional(risk_scores, "risk_scores").astype(np.float64)
    if len(scores) != len(frame) or not np.isfinite(scores).all():
        raise ValueError("One finite risk score is required for every decision row")
    output = frame.copy()
    if test_result.selected_threshold is None:
        threshold_eligible = np.zeros(len(frame), dtype=bool)
        accepted_reason = "no_calibration_certified_threshold"
        review_reason = "no_calibration_certified_threshold"
    else:
        threshold_eligible = scores <= test_result.selected_threshold
        accepted_reason = "risk_score_at_or_below_locked_threshold"
        review_reason = "risk_score_above_locked_threshold"
    policy_authorized = test_result.status == "risk-controlled"
    accepted = threshold_eligible & policy_authorized
    if not policy_authorized:
        accepted_reason = f"policy_not_risk_controlled:{test_result.status_reason}"
        review_reason = accepted_reason
    output["protocol_role"] = "final_test"
    output["risk_score"] = scores
    output["locked_threshold"] = test_result.selected_threshold
    output["threshold_eligible"] = threshold_eligible.astype(np.int64)
    output["policy_authorized"] = int(policy_authorized)
    output["accepted"] = accepted.astype(np.int64)
    output["decision_source"] = decision_source
    output["disposition"] = np.where(accepted, "accept", review_disposition)
    output["acquisition_target"] = pd.NA
    if review_disposition == "acquire_view":
        output.loc[~accepted, "acquisition_target"] = acquisition_target
    if policy_authorized:
        output["reason"] = np.where(accepted, accepted_reason, review_reason)
    else:
        output["reason"] = review_reason
    output["risk_control_status"] = test_result.status
    output["risk_control_status_reason"] = test_result.status_reason
    if losses is None:
        output["bounded_loss"] = pd.NA
    else:
        bounded_losses = _as_one_dimensional(losses, "losses").astype(np.float64)
        if len(bounded_losses) != len(frame):
            raise ValueError("One loss is required for every decision row")
        if not np.isfinite(bounded_losses).all() or (
            (bounded_losses < 0.0) | (bounded_losses > 1.0)
        ).any():
            raise ValueError("losses must be finite and bounded in [0, 1]")
        output["bounded_loss"] = bounded_losses
    required = [
        "protocol_role",
        "risk_score",
        "locked_threshold",
        "threshold_eligible",
        "policy_authorized",
        "accepted",
        "decision_source",
        "disposition",
        "acquisition_target",
        "reason",
        "risk_control_status",
        "risk_control_status_reason",
        "bounded_loss",
    ]
    identifier_columns = [
        column
        for column in ("schema_version", "dataset", "seed", "sample_id", "objectid")
        if column in output.columns
    ]
    ordered = identifier_columns + required
    ordered.extend(column for column in output.columns if column not in ordered)
    return output[ordered]
