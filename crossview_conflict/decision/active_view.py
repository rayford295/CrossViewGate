from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


DEFAULT_OPERATIONAL_COST = np.asarray(
    [
        [0.0, 1.0, 4.0],
        [1.0, 0.0, 1.0],
        [8.0, 8.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ActiveViewCache:
    sample_id: np.ndarray
    spatial_block_id: np.ndarray
    sequence_id: np.ndarray
    target: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    sector_id: np.ndarray
    relative_azimuth_deg: np.ndarray
    street_embedding: np.ndarray
    overhead_embedding: np.ndarray
    sector_logits: np.ndarray
    panorama_logits: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(len(self.sample_id))

    @property
    def sector_count(self) -> int:
        return int(len(self.sector_id))

    @property
    def embedding_dim(self) -> int:
        return int(self.street_embedding.shape[-1])


def load_active_view_cache(path: str | Path) -> ActiveViewCache:
    with np.load(Path(path), allow_pickle=False) as payload:
        cache = ActiveViewCache(
            sample_id=payload["sample_id"].astype(str),
            spatial_block_id=payload["spatial_block_id"].astype(str),
            sequence_id=payload["sequence_id"].astype(str),
            target=payload["target"].astype(np.int64),
            latitude=payload["latitude"].astype(np.float64),
            longitude=payload["longitude"].astype(np.float64),
            sector_id=payload["sector_id"].astype(np.int64),
            relative_azimuth_deg=payload["relative_azimuth_deg"].astype(np.float64),
            street_embedding=payload["street_embedding"].astype(np.float32),
            overhead_embedding=payload["overhead_embedding"].astype(np.float32),
            sector_logits=payload["sector_logits"].astype(np.float32),
            panorama_logits=payload["panorama_logits"].astype(np.float32),
        )
    validate_active_view_cache(cache)
    return cache


def validate_active_view_cache(cache: ActiveViewCache) -> None:
    count = cache.sample_count
    sectors = cache.sector_count
    if count < 1 or sectors < 2:
        raise ValueError("Active-view cache must contain samples and at least two sectors")
    if len(set(cache.sample_id.tolist())) != count:
        raise ValueError("Active-view sample ids must be unique")
    if sorted(cache.sector_id.tolist()) != list(range(sectors)):
        raise ValueError("Active-view sector ids must be contiguous from zero")
    if cache.street_embedding.shape[:2] != (count, sectors):
        raise ValueError("street_embedding shape is inconsistent")
    if cache.overhead_embedding.shape != (count, cache.embedding_dim):
        raise ValueError("overhead_embedding shape is inconsistent")
    if cache.sector_logits.shape[:2] != (count, sectors):
        raise ValueError("sector_logits shape is inconsistent")
    if cache.panorama_logits.shape[0] != count:
        raise ValueError("panorama_logits shape is inconsistent")
    if not np.isfinite(cache.street_embedding).all():
        raise ValueError("street embeddings contain non-finite values")
    if not np.isfinite(cache.overhead_embedding).all():
        raise ValueError("overhead embeddings contain non-finite values")


def validate_revealed_mask(mask: np.ndarray, *, sector_count: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or values.shape[1] != sector_count:
        raise ValueError(
            f"revealed mask must have shape (n, {sector_count}), got {values.shape}"
        )
    return values


def reveal_sector(mask: np.ndarray, sample_index: int, sector_id: int) -> None:
    if not 0 <= sample_index < mask.shape[0]:
        raise IndexError("sample_index is outside the revealed mask")
    if not 0 <= sector_id < mask.shape[1]:
        raise ValueError(f"Invalid sector_id: {sector_id}")
    if bool(mask[sample_index, sector_id]):
        raise ValueError(f"Sector {sector_id} has already been revealed")
    mask[sample_index, sector_id] = True


def build_state_features(
    cache: ActiveViewCache,
    revealed_mask: np.ndarray,
    *,
    sample_indices: np.ndarray | None = None,
    last_sector: np.ndarray | None = None,
    remaining_budget: np.ndarray | float | None = None,
) -> np.ndarray:
    """Build policy-safe state features from revealed sectors only."""
    mask = validate_revealed_mask(revealed_mask, sector_count=cache.sector_count)
    if sample_indices is None:
        indices = np.arange(cache.sample_count, dtype=np.int64)
    else:
        indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
        if ((indices < 0) | (indices >= cache.sample_count)).any():
            raise IndexError("sample_indices contains an out-of-range value")
    if mask.shape[0] != len(indices):
        raise ValueError("revealed mask sample count does not match sample_indices")
    mask_float = mask.astype(np.float32)
    counts = mask_float.sum(axis=1, keepdims=True)
    safe_counts = np.maximum(counts, 1.0)
    selected_street = cache.street_embedding[indices]
    selected_logits = cache.sector_logits[indices]
    street_sum = np.einsum("ns,nsd->nd", mask_float, selected_street)
    street_mean = street_sum / safe_counts
    masked_street = np.where(
        mask[:, :, None], selected_street, -np.inf
    )
    street_max = masked_street.max(axis=1)
    street_max[~np.isfinite(street_max)] = 0.0
    logit_sum = np.einsum("ns,nsc->nc", mask_float, selected_logits)
    logit_mean = logit_sum / safe_counts
    logit_mean[counts[:, 0] == 0] = 0.0
    angles = np.radians(cache.relative_azimuth_deg)
    mean_sin = (mask_float * np.sin(angles)[None, :]).sum(axis=1, keepdims=True) / safe_counts
    mean_cos = (mask_float * np.cos(angles)[None, :]).sum(axis=1, keepdims=True) / safe_counts
    mean_sin[counts[:, 0] == 0] = 0.0
    mean_cos[counts[:, 0] == 0] = 0.0
    normalized_count = counts / float(cache.sector_count)
    if last_sector is None:
        pass
    else:
        last = np.asarray(last_sector, dtype=np.int64).reshape(-1)
        if len(last) != len(indices):
            raise ValueError("last_sector length does not match the cache")
        if ((last < -1) | (last >= cache.sector_count)).any():
            raise ValueError("last_sector contains an invalid sector id")
    if remaining_budget is None:
        budget = np.zeros((len(indices), 1), dtype=np.float32)
    else:
        budget_values = np.asarray(remaining_budget, dtype=np.float32)
        if budget_values.ndim == 0:
            budget_values = np.full(len(indices), float(budget_values))
        budget = budget_values.reshape(-1, 1) / float(cache.sector_count)
        if len(budget) != len(indices):
            raise ValueError("remaining_budget length does not match the cache")
    return np.concatenate(
        [
            cache.overhead_embedding[indices],
            street_mean,
            street_max,
            logit_mean,
            mask_float,
            normalized_count,
            mean_sin.astype(np.float32),
            mean_cos.astype(np.float32),
            budget,
        ],
        axis=1,
    ).astype(np.float32)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    values = values - values.max(axis=-1, keepdims=True)
    exponential = np.exp(values)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    return -(values * np.log(values)).sum(axis=-1)


def expected_operational_cost(
    probabilities: np.ndarray,
    targets: np.ndarray,
    cost_matrix: np.ndarray = DEFAULT_OPERATIONAL_COST,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64).reshape(-1)
    if probs.shape[0] != len(labels):
        raise ValueError("probabilities and targets have inconsistent lengths")
    return np.sum(probs * np.asarray(cost_matrix)[labels], axis=1)


def realized_operational_cost(
    predictions: np.ndarray,
    targets: np.ndarray,
    cost_matrix: np.ndarray = DEFAULT_OPERATIONAL_COST,
) -> np.ndarray:
    predicted = np.asarray(predictions, dtype=np.int64).reshape(-1)
    labels = np.asarray(targets, dtype=np.int64).reshape(-1)
    if len(predicted) != len(labels):
        raise ValueError("predictions and targets have inconsistent lengths")
    return np.asarray(cost_matrix)[labels, predicted]


def build_selector_features(
    state_features: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    sorted_probs = np.sort(probs, axis=1)
    confidence = probs.max(axis=1, keepdims=True)
    margin = (sorted_probs[:, -1] - sorted_probs[:, -2])[:, None]
    uncertainty = entropy(probs).astype(np.float32)[:, None]
    return np.concatenate(
        [state_features, probs, confidence, margin, uncertainty], axis=1
    ).astype(np.float32)


class ActiveViewMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "Standardizer":
        values = np.asarray(features, dtype=np.float64)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-6] = 1.0
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (
            (np.asarray(features, dtype=np.float32) - self.mean) / self.scale
        ).astype(np.float32)


def fixed_clockwise_order(sector_count: int) -> tuple[int, ...]:
    if sector_count < 2:
        raise ValueError("sector_count must be at least two")
    return tuple(range(sector_count))


def farthest_available_sector(
    revealed: Iterable[int],
    available: Iterable[int],
    *,
    sector_count: int,
) -> int:
    revealed_values = tuple(int(value) for value in revealed)
    available_values = tuple(int(value) for value in available)
    if not available_values:
        raise ValueError("No available sector remains")
    if not revealed_values:
        return min(available_values)

    def circular_distance(left: int, right: int) -> int:
        difference = abs(left - right)
        return min(difference, sector_count - difference)

    return max(
        available_values,
        key=lambda candidate: (
            min(circular_distance(candidate, existing) for existing in revealed_values),
            -candidate,
        ),
    )
