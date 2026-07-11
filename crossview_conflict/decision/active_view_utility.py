from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .active_view import (
    DEFAULT_OPERATIONAL_COST,
    build_selector_features,
    expected_operational_cost,
    validate_revealed_mask,
)


@dataclass(frozen=True)
class AdaptiveActionBatch:
    action: np.ndarray
    sector_id: np.ndarray
    predicted_net_utility: np.ndarray
    utility_was_clipped: np.ndarray
    bayes_action: np.ndarray
    bayes_risk: np.ndarray
    stop_q: np.ndarray
    acquire_q: np.ndarray
    defer_q: np.ndarray


def target_conditioned_soft_loss(
    probabilities: np.ndarray,
    targets: np.ndarray,
    cost_matrix: np.ndarray = DEFAULT_OPERATIONAL_COST,
) -> np.ndarray:
    """Supervised soft loss ``sum_j p(j|x) C[y,j]`` used only for targets."""
    return expected_operational_cost(probabilities, targets, cost_matrix)


def build_candidate_geometry_features(
    revealed_mask: np.ndarray,
    candidate_sector: np.ndarray,
    relative_azimuth_deg: np.ndarray,
    *,
    compass_angle_deg: np.ndarray,
    compass_available: np.ndarray,
) -> np.ndarray:
    """Encode a candidate using only canonical geometry and the revealed set."""
    azimuth = np.asarray(relative_azimuth_deg, dtype=np.float64).reshape(-1)
    mask = validate_revealed_mask(revealed_mask, sector_count=len(azimuth))
    candidate = np.asarray(candidate_sector, dtype=np.int64).reshape(-1)
    if len(candidate) != len(mask):
        raise ValueError("candidate_sector length does not match revealed_mask")
    if ((candidate < 0) | (candidate >= len(azimuth))).any():
        raise ValueError("candidate_sector contains an invalid sector id")
    if np.any(mask[np.arange(len(mask)), candidate]):
        raise ValueError("candidate_sector must be unrevealed")
    compass = np.asarray(compass_angle_deg, dtype=np.float64).reshape(-1)
    availability = np.asarray(compass_available, dtype=bool).reshape(-1)
    if len(compass) != len(mask) or len(availability) != len(mask):
        raise ValueError("compass metadata must match revealed_mask")
    if not np.isfinite(compass[availability]).all():
        raise ValueError("available compass angles must be finite")
    safe_compass = np.where(availability, compass, 0.0)
    one_hot = np.eye(len(azimuth), dtype=np.float32)[candidate]
    relative_radians = np.radians(azimuth[candidate])
    absolute_radians = np.radians((safe_compass + azimuth[candidate]) % 360.0)
    nearest_distance = np.empty(len(mask), dtype=np.float32)
    for row, sector_id in enumerate(candidate):
        revealed = np.flatnonzero(mask[row])
        if len(revealed) == 0:
            nearest_distance[row] = 1.0
            continue
        difference = np.abs(
            (azimuth[revealed] - azimuth[sector_id] + 180.0) % 360.0 - 180.0
        )
        nearest_distance[row] = float(difference.min() / 180.0)
    return np.concatenate(
        [
            one_hot,
            np.sin(relative_radians).astype(np.float32)[:, None],
            np.cos(relative_radians).astype(np.float32)[:, None],
            (
                np.sin(absolute_radians) * availability
            ).astype(np.float32)[:, None],
            (
                np.cos(absolute_radians) * availability
            ).astype(np.float32)[:, None],
            availability.astype(np.float32)[:, None],
            nearest_distance[:, None],
        ],
        axis=1,
    ).astype(np.float32)


def build_candidate_utility_features(
    state_features: np.ndarray,
    probabilities: np.ndarray,
    revealed_mask: np.ndarray,
    candidate_sector: np.ndarray,
    relative_azimuth_deg: np.ndarray,
    *,
    compass_angle_deg: np.ndarray,
    compass_available: np.ndarray,
) -> np.ndarray:
    """Build features without reading the hidden candidate image or embedding."""
    state = np.asarray(state_features, dtype=np.float32)
    geometry = build_candidate_geometry_features(
        revealed_mask,
        candidate_sector,
        relative_azimuth_deg,
        compass_angle_deg=compass_angle_deg,
        compass_available=compass_available,
    )
    selector = build_selector_features(state, probabilities)
    if len(selector) != len(geometry):
        raise ValueError("candidate geometry and state features have inconsistent lengths")
    return np.concatenate([selector, geometry], axis=1).astype(np.float32)


def bayes_operational_action_and_risk(
    probabilities: np.ndarray,
    cost_matrix: np.ndarray = DEFAULT_OPERATIONAL_COST,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the minimum-cost action and its model-implied operational risk."""
    probs = np.asarray(probabilities, dtype=np.float64)
    costs = np.asarray(cost_matrix, dtype=np.float64)
    if probs.ndim != 2 or costs.shape != (probs.shape[1], probs.shape[1]):
        raise ValueError("probabilities and cost_matrix have incompatible shapes")
    if (
        not np.isfinite(costs).all()
        or (costs < 0).any()
        or not np.allclose(np.diag(costs), 0.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("cost_matrix must be finite, non-negative, and zero-diagonal")
    if not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(probs.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("probabilities must sum to one")
    action_cost = probs @ costs
    action = action_cost.argmin(axis=1).astype(np.int64)
    risk = action_cost[np.arange(len(probs)), action]
    return action, risk.astype(np.float64)


def select_adaptive_actions(
    predicted_net_utility_by_sector: np.ndarray,
    revealed_mask: np.ndarray,
    probabilities: np.ndarray,
    *,
    defer_cost: float,
    stop_risk_threshold: float | None,
    view_cost: float,
    cost_matrix: np.ndarray = DEFAULT_OPERATIONAL_COST,
) -> AdaptiveActionBatch:
    """Minimize locked STOP/ACQUIRE/DEFER Q-values under a stop-safety constraint.

    Candidate utility is defined as risk reduction minus acquisition cost, so
    ``Q_acquire = Q_stop - predicted_net_utility``. STOP is unavailable when
    current model-implied risk exceeds ``stop_risk_threshold``; ``None`` means
    no threshold was certified and disables automatic STOP entirely.
    """
    utility = np.asarray(predicted_net_utility_by_sector, dtype=np.float64)
    if utility.ndim != 2:
        raise ValueError("predicted utilities must be a two-dimensional array")
    mask = validate_revealed_mask(revealed_mask, sector_count=utility.shape[1])
    if utility.shape != mask.shape:
        raise ValueError("predicted utilities must match revealed_mask")
    if len(probabilities) != len(mask):
        raise ValueError("probabilities and utility rows are inconsistent")
    numeric_controls = [defer_cost, view_cost]
    if stop_risk_threshold is not None:
        numeric_controls.append(stop_risk_threshold)
    if not np.isfinite(numeric_controls).all() or min(numeric_controls) < 0:
        raise ValueError("cost and risk thresholds must be non-negative")
    available = ~mask
    if not np.isfinite(utility[available]).all():
        raise ValueError("available candidate utilities must be finite")
    bayes_action, bayes_risk = bayes_operational_action_and_risk(
        probabilities, cost_matrix
    )
    maximum_cost = float(np.max(np.asarray(cost_matrix, dtype=np.float64)))
    lower_bound = -maximum_cost - float(view_cost)
    upper_bound = maximum_cost - float(view_cost)
    clipped_utility = np.clip(utility, lower_bound, upper_bound)
    utility_was_clipped = available & ~np.isclose(
        utility, clipped_utility, rtol=0.0, atol=1e-12
    )
    masked_utility = np.where(available, clipped_utility, -np.inf)
    has_candidate = available.any(axis=1)
    best_sector = masked_utility.argmax(axis=1).astype(np.int64)
    best_utility = masked_utility[np.arange(len(mask)), best_sector]
    best_sector[~has_candidate] = -1
    best_utility[~has_candidate] = -np.inf

    stop_q = (
        np.where(bayes_risk <= stop_risk_threshold, bayes_risk, np.inf)
        if stop_risk_threshold is not None
        else np.full(len(mask), np.inf, dtype=np.float64)
    )
    acquire_q = np.where(has_candidate, bayes_risk - best_utility, np.inf)
    defer_q = np.full(len(mask), float(defer_cost), dtype=np.float64)
    q_values = np.stack([stop_q, acquire_q, defer_q], axis=1)
    choice = q_values.argmin(axis=1)
    action_names = np.asarray(["stop", "acquire", "defer"], dtype="U7")
    action = action_names[choice]
    selected_sector = np.where(choice == 1, best_sector, -1).astype(np.int64)
    return AdaptiveActionBatch(
        action=action,
        sector_id=selected_sector,
        predicted_net_utility=best_utility,
        utility_was_clipped=utility_was_clipped,
        bayes_action=bayes_action,
        bayes_risk=bayes_risk,
        stop_q=stop_q,
        acquire_q=acquire_q,
        defer_q=defer_q,
    )
