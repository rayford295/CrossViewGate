"""Pure relative-geometry feature interface for cross-event view selection.

The interface intentionally has no image, label, fitting, calibration, or I/O
path.  Training and zero-shot scoring import this same function so that the
Milton sensitivity cannot silently change its feature projection.
"""

from __future__ import annotations

import numpy as np


RELATIVE_GEOMETRY_FEATURE_SCHEMA = "relative-geometry-utility-features-v1"
REGISTERED_RELATIVE_AZIMUTH_DEG = np.asarray(
    [0.0, 45.0, 90.0, 135.0, -180.0, -135.0, -90.0, -45.0],
    dtype=np.float64,
)
RELATIVE_GEOMETRY_FEATURE_NAMES = (
    "probability_class_0",
    "probability_class_1",
    "probability_class_2",
    "maximum_probability",
    "top_two_probability_margin",
    "probability_entropy",
    *(f"revealed_relative_sector_{index}" for index in range(8)),
    "normalized_revealed_count",
    "relative_revealed_mean_sine",
    "relative_revealed_mean_cosine",
    "normalized_remaining_acquisitions",
    *(f"candidate_relative_sector_{index}" for index in range(8)),
    "candidate_relative_yaw_sine",
    "candidate_relative_yaw_cosine",
    "nearest_revealed_angular_distance",
)


def build_relative_geometry_utility_features(
    probabilities: np.ndarray,
    revealed_mask: np.ndarray,
    candidate_sector: np.ndarray,
    relative_azimuth_deg: np.ndarray,
    *,
    origin_sector: np.ndarray | int,
    remaining_acquisitions: np.ndarray,
) -> np.ndarray:
    """Return the locked 29-dimensional rotation-relative policy features.

    The registered main policy first rolls sector-indexed cache tensors so the
    physical origin is local sector zero; consequently ``origin_sector`` must
    be local zero.  The
    remaining-acquisition feature follows the frozen CVIAN classifier state
    convention exactly: ``remaining_acquisitions / sector_count`` (therefore
    ``(4-k)/8`` for the registered experiment).

    No canonical sector one-hot, compass angle, absolute yaw, hidden candidate
    image, embedding, logit, or label enters this interface.
    """

    probs = np.asarray(probabilities, dtype=np.float64)
    mask = np.asarray(revealed_mask, dtype=bool)
    candidate = np.asarray(candidate_sector, dtype=np.int64).reshape(-1)
    azimuth = np.asarray(relative_azimuth_deg, dtype=np.float64).reshape(-1)
    remaining = np.asarray(remaining_acquisitions, dtype=np.float64).reshape(-1)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("probabilities must have shape (n, 3)")
    if mask.ndim != 2 or mask.shape[1] != len(azimuth):
        raise ValueError("revealed_mask and relative_azimuth_deg are inconsistent")
    row_count, sector_count = mask.shape
    if sector_count != 8:
        raise ValueError("The registered relative-geometry interface requires 8 sectors")
    if not (
        len(probs) == row_count
        and len(candidate) == row_count
        and len(remaining) == row_count
    ):
        raise ValueError("All relative-geometry feature inputs must have equal rows")
    if not np.isfinite(probs).all() or (probs < 0.0).any():
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(probs.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("probabilities must sum to one")
    if not np.array_equal(azimuth, REGISTERED_RELATIVE_AZIMUTH_DEG):
        raise ValueError("relative_azimuth_deg does not match the registered 45-degree geometry")
    if ((candidate < 0) | (candidate >= sector_count)).any():
        raise ValueError("candidate_sector contains an invalid sector")
    if np.any(mask[np.arange(row_count), candidate]):
        raise ValueError("candidate_sector must be unrevealed")
    if not np.isfinite(remaining).all() or (remaining < 0.0).any():
        raise ValueError("remaining_acquisitions must be finite and non-negative")

    origin_values = np.asarray(origin_sector, dtype=np.int64)
    if origin_values.ndim == 0:
        origins = np.full(row_count, int(origin_values), dtype=np.int64)
    else:
        origins = origin_values.reshape(-1)
    if len(origins) != row_count or ((origins < 0) | (origins >= sector_count)).any():
        raise ValueError("origin_sector must provide one valid origin per row")
    if np.any(origins != 0):
        raise ValueError("The registered main interface requires rolled-local origin sector 0")
    if not np.all(mask[np.arange(row_count), origins]):
        raise ValueError("origin_sector must already be revealed")

    relative_index = (
        np.arange(sector_count, dtype=np.int64)[None, :] + origins[:, None]
    ) % sector_count
    relative_mask = np.take_along_axis(mask, relative_index, axis=1)
    candidate_relative = (candidate - origins) % sector_count

    origin_azimuth = azimuth[origins]
    sector_delta_deg = (
        azimuth[None, :] - origin_azimuth[:, None] + 180.0
    ) % 360.0 - 180.0
    relative_order_delta = np.take_along_axis(
        sector_delta_deg, relative_index, axis=1
    )
    relative_radians = np.radians(relative_order_delta)
    count = relative_mask.sum(axis=1).astype(np.float64)
    expected_remaining = np.maximum(4.0 - count, 0.0)
    if not np.allclose(remaining, expected_remaining, rtol=0.0, atol=1e-7):
        raise ValueError("remaining_acquisitions must equal max(4 - revealed_count, 0)")
    safe_count = np.maximum(count, 1.0)
    mean_sine = (
        relative_mask * np.sin(relative_radians)
    ).sum(axis=1) / safe_count
    mean_cosine = (
        relative_mask * np.cos(relative_radians)
    ).sum(axis=1) / safe_count

    candidate_delta_deg = (
        azimuth[candidate] - origin_azimuth + 180.0
    ) % 360.0 - 180.0
    candidate_radians = np.radians(candidate_delta_deg)
    nearest_distance = np.empty(row_count, dtype=np.float64)
    for row in range(row_count):
        revealed = np.flatnonzero(mask[row])
        difference = np.abs(
            (azimuth[revealed] - azimuth[candidate[row]] + 180.0) % 360.0
            - 180.0
        )
        nearest_distance[row] = float(difference.min() / 180.0)

    sorted_probs = np.sort(probs, axis=1)
    entropy = -(
        np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0))
    ).sum(axis=1)
    features = np.concatenate(
        [
            probs,
            probs.max(axis=1, keepdims=True),
            (sorted_probs[:, -1] - sorted_probs[:, -2])[:, None],
            entropy[:, None],
            relative_mask.astype(np.float64),
            (count / sector_count)[:, None],
            mean_sine[:, None],
            mean_cosine[:, None],
            (remaining / sector_count)[:, None],
            np.eye(sector_count, dtype=np.float64)[candidate_relative],
            np.sin(candidate_radians)[:, None],
            np.cos(candidate_radians)[:, None],
            nearest_distance[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    if features.shape[1] != len(RELATIVE_GEOMETRY_FEATURE_NAMES):
        raise RuntimeError("Relative-geometry feature width drifted")
    if not np.isfinite(features).all():
        raise ValueError("Relative-geometry features contain non-finite values")
    return features
