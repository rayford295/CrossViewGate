from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.utils.io import ensure_dir

VISIBILITY_FEATURES = [
    "building_ratio",
    "center_building_ratio",
    "center_minus_global",
    "centroid_distance_norm",
]
DERIVED_FEATURES = [
    "street_confidence",
    "remote_confidence",
    "street_entropy",
    "remote_entropy",
    "confidence_gap",
    "js_divergence",
    "views_disagree",
]
ALL_FEATURES = VISIBILITY_FEATURES + DERIVED_FEATURES
VIEW_NAMES = ("street", "remote", "crossview")
EVIDENCE_METADATA_COLUMNS = (
    "objectid",
    "location",
    "latitude",
    "longitude",
    "tile_id",
    "group_id",
    "sequence_id",
    "spatial_block_id",
    "protocol_role",
    "event_id",
    "event",
    "source_dataset",
)
EVIDENCE_SCHEMA_VERSION = "p0.2-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visibility-conditioned reliability gate. Learns a per-sample street-vs-"
            "remote mixture weight from building visibility, calibrated confidence, "
            "and cross-view disagreement. Trains on val predictions, evaluates on "
            "test, and supports zero-shot cross-disaster transfer."
        )
    )
    parser.add_argument("--multiseed-root", default="outputs/multiseed_main")
    parser.add_argument("--visibility-dir", default="outputs/analysis/visibility_features")
    parser.add_argument("--datasets", default="altadena_3class,ian_original,milton_original")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="outputs/analysis/reliability_gate")
    parser.add_argument("--doc-path", default="docs/reliability_gate_results.md")
    parser.add_argument(
        "--split-manifests",
        default=(
            "altadena_3class=data/splits/altadena_3class_objectid,"
            "ian_original=data/splits/ian_hurricane_original,"
            "milton_original=data/splits/milton_hurricane_original"
        ),
        metavar="DATASET=PATH,...",
        help=(
            "Dataset-to-test-manifest mapping used to attach objectid, "
            "location, and tile_id to per-sample evidence. PATH may be either a "
            "CSV file or a split directory containing test.csv."
        ),
    )
    parser.add_argument(
        "--risk-protocol-manifests",
        default="",
        metavar="DATASET=PATH,...",
        help=(
            "Optional dataset-to-directory mapping. Each directory must contain "
            "gate_fit.csv, risk_calibration.csv, and final_test.csv. When supplied, "
            "temperature/gate fitting uses gate_fit only and fixed-gate evidence is "
            "written for all three roles."
        ),
    )
    return parser.parse_args()


def _canonical_sample_id(value: object) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"(?:sample_|milton_)?(\d+)(?:\.0+)?", text, flags=re.IGNORECASE)
    if match is None:
        match = re.fullmatch(r"tensor\((\d+)\)", text, flags=re.IGNORECASE)
    if match:
        return str(int(match.group(1)))
    return text


def _assert_unique_sample_ids(frame: pd.DataFrame, source: str) -> None:
    if frame["sample_id"].isna().any():
        raise ValueError(f"{source} contains missing sample_id values")
    duplicated = frame["sample_id"].duplicated(keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, "sample_id"].astype(str).unique()[:5].tolist()
        raise ValueError(f"{source} contains duplicate canonical sample_ids: {examples}")


def _numbered_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    pairs: list[tuple[int, str]] = []
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    for column in df.columns:
        match = pattern.match(column)
        if match:
            pairs.append((int(match.group(1)), column))
    return [column for _, column in sorted(pairs)]


def _parse_split_manifest_map(specification: str) -> dict[str, Path]:
    """Parse a comma-separated ``dataset=path`` CLI mapping."""
    mapping: dict[str, Path] = {}
    for raw_entry in specification.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                f"Invalid --split-manifests entry {entry!r}; expected DATASET=PATH"
            )
        dataset, raw_path = (part.strip() for part in entry.split("=", 1))
        if not dataset or not raw_path:
            raise ValueError(
                f"Invalid --split-manifests entry {entry!r}; dataset and path are required"
            )
        if dataset in mapping:
            raise ValueError(f"Duplicate --split-manifests dataset: {dataset}")
        mapping[dataset] = Path(raw_path)
    return mapping


def _read_test_manifest(path: Path) -> pd.DataFrame:
    """Read a mapped test manifest, accepting either its CSV or split directory."""
    csv_path = path / "test.csv" if path.is_dir() else path
    if not csv_path.is_file():
        raise FileNotFoundError(f"Test manifest not found: {csv_path}")
    manifest = pd.read_csv(csv_path)
    if "sample_id" not in manifest.columns:
        raise ValueError(f"Test manifest has no sample_id column: {csv_path}")
    return manifest


def _read_protocol_manifest(directory: Path, role: str) -> pd.DataFrame:
    path = directory / f"{role}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Protocol role manifest not found: {path}")
    manifest = pd.read_csv(path)
    if "sample_id" not in manifest:
        raise ValueError(f"Protocol role manifest has no sample_id: {path}")
    manifest["sample_id"] = manifest["sample_id"].map(_canonical_sample_id)
    _assert_unique_sample_ids(manifest, str(path))
    return manifest


def _subset_split_data(
    data: SplitData,
    sample_ids: set[str],
    *,
    role: str,
) -> SplitData:
    if data.sample_ids is None:
        raise ValueError(f"Cannot subset {role}: SplitData has no sample_ids")
    canonical = np.asarray([_canonical_sample_id(value) for value in data.sample_ids])
    observed = set(canonical)
    missing = sorted(sample_ids - observed)
    if missing:
        raise ValueError(f"{role} manifest ids absent from predictions: {missing[:5]}")
    mask = np.asarray([value in sample_ids for value in canonical], dtype=bool)
    if not mask.any():
        raise ValueError(f"{role} has no aligned prediction rows")
    return SplitData(
        features=data.features[mask],
        street_probs=data.street_probs[mask],
        remote_probs=data.remote_probs[mask],
        targets=data.targets[mask],
        crossview_probs=(
            data.crossview_probs[mask] if data.crossview_probs is not None else None
        ),
        sample_ids=canonical[mask],
    )


def _filter_prediction_frame(
    frame: pd.DataFrame,
    sample_ids: set[str],
    *,
    source: str,
) -> pd.DataFrame:
    filtered = frame.copy()
    filtered["sample_id"] = filtered["sample_id"].map(_canonical_sample_id)
    _assert_unique_sample_ids(filtered, source)
    missing = sorted(sample_ids - set(filtered["sample_id"]))
    if missing:
        raise ValueError(f"{source} is missing gate-fit ids: {missing[:5]}")
    filtered = filtered[filtered["sample_id"].isin(sample_ids)].copy()
    if filtered.empty:
        raise ValueError(f"{source} has no gate-fit rows")
    return filtered


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    scaled = logits / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(targets)), targets].mean())


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 200))
    losses = [_nll(logits, targets, float(t)) for t in grid]
    best = float(grid[int(np.argmin(losses))])
    low, high = best / 1.25, best * 1.25
    for _ in range(40):
        mid1 = low + (high - low) / 3.0
        mid2 = high - (high - low) / 3.0
        if _nll(logits, targets, mid1) < _nll(logits, targets, mid2):
            high = mid2
        else:
            low = mid1
    return float((low + high) / 2.0)


def _entropy(probs: np.ndarray) -> np.ndarray:
    return -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = (p + q) / 2.0
    def kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a * (np.log(np.clip(a, 1e-12, 1.0)) - np.log(np.clip(b, 1e-12, 1.0)))).sum(axis=1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


class SplitData:
    """Aligned per-sample arrays for one dataset/seed/split."""

    def __init__(
        self,
        features: np.ndarray,
        street_probs: np.ndarray,
        remote_probs: np.ndarray,
        targets: np.ndarray,
        crossview_probs: np.ndarray | None = None,
        sample_ids: np.ndarray | None = None,
    ) -> None:
        self.features = features
        self.street_probs = street_probs
        self.remote_probs = remote_probs
        self.targets = targets
        self.crossview_probs = crossview_probs
        self.sample_ids = sample_ids


def load_split(
    root: Path,
    visibility_dir: Path,
    dataset: str,
    seed: int,
    split: str,
    temperatures: dict[str, float],
) -> SplitData:
    frames: dict[str, pd.DataFrame] = {}
    for mode in ("street_only", "remote_only", "crossview"):
        path = root / dataset / f"{mode}_seed{seed}" / f"{split}_predictions.csv"
        df = pd.read_csv(path)
        df["sample_id"] = df["sample_id"].map(_canonical_sample_id)
        _assert_unique_sample_ids(df, str(path))
        frames[mode] = df
    visibility = pd.read_csv(visibility_dir / f"{dataset}_{split}.csv")
    visibility["sample_id"] = visibility["sample_id"].map(_canonical_sample_id)
    if "error" in visibility.columns:
        visibility = visibility[visibility["error"].isna()]
    _assert_unique_sample_ids(visibility, f"{dataset}_{split} visibility")

    expected_ids = set(frames["street_only"]["sample_id"])
    for source_name, source_frame in (
        ("remote_only", frames["remote_only"]),
        ("crossview", frames["crossview"]),
        ("visibility", visibility),
    ):
        observed_ids = set(source_frame["sample_id"])
        if observed_ids != expected_ids:
            missing = sorted(expected_ids - observed_ids)[:5]
            extra = sorted(observed_ids - expected_ids)[:5]
            raise ValueError(
                f"Sample coverage mismatch for {dataset} seed={seed} {split} {source_name}: "
                f"missing={missing}, extra={extra}"
            )

    crossview_logit_cols = _numbered_columns(frames["crossview"], "logit")
    crossview_frame = frames["crossview"][
        ["sample_id", "target"] + crossview_logit_cols
    ].rename(
        columns={
            "target": "target_crossview",
            **{col: f"{col}_crossview" for col in crossview_logit_cols},
        }
    )
    merged = frames["street_only"].merge(
        frames["remote_only"],
        on="sample_id",
        suffixes=("_street", "_remote"),
        validate="one_to_one",
    )
    merged = merged.merge(
        crossview_frame, on="sample_id", how="inner", validate="one_to_one"
    )
    merged = merged.merge(
        visibility[["sample_id"] + VISIBILITY_FEATURES],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError(f"Empty merge for {dataset} seed={seed} {split}")

    street_logit_cols = [f"{c}_street" for c in _numbered_columns(frames["street_only"], "logit")]
    remote_logit_cols = [f"{c}_remote" for c in _numbered_columns(frames["remote_only"], "logit")]
    street_logits = merged[street_logit_cols].to_numpy(dtype=np.float64)
    remote_logits = merged[remote_logit_cols].to_numpy(dtype=np.float64)
    crossview_logits = merged[[f"{c}_crossview" for c in crossview_logit_cols]].to_numpy(dtype=np.float64)
    street_probs = _softmax(street_logits / temperatures["street_only"])
    remote_probs = _softmax(remote_logits / temperatures["remote_only"])
    crossview_probs = _softmax(crossview_logits / temperatures["crossview"])
    targets = merged["target_street"].to_numpy(dtype=np.int64)
    for column in ("target_remote", "target_crossview"):
        other_targets = merged[column].to_numpy(dtype=np.int64)
        if not np.array_equal(targets, other_targets):
            mismatch_ids = merged.loc[targets != other_targets, "sample_id"].head().tolist()
            raise ValueError(
                f"Target mismatch in {dataset} seed={seed} {split} {column}: {mismatch_ids}"
            )

    derived = pd.DataFrame(
        {
            "street_confidence": street_probs.max(axis=1),
            "remote_confidence": remote_probs.max(axis=1),
            "street_entropy": _entropy(street_probs),
            "remote_entropy": _entropy(remote_probs),
            "confidence_gap": street_probs.max(axis=1) - remote_probs.max(axis=1),
            "js_divergence": _js_divergence(street_probs, remote_probs),
            "views_disagree": (street_probs.argmax(axis=1) != remote_probs.argmax(axis=1)).astype(float),
        }
    )
    features = np.concatenate(
        [merged[VISIBILITY_FEATURES].to_numpy(dtype=np.float64), derived.to_numpy(dtype=np.float64)],
        axis=1,
    )
    return SplitData(
        features,
        street_probs,
        remote_probs,
        targets,
        crossview_probs,
        merged["sample_id"].to_numpy(),
    )


class GateModel(nn.Module):
    """num_views=2: sigmoid street weight. num_views>2: softmax mixture weights."""

    def __init__(self, num_features: int, hidden_dim: int, num_views: int = 2) -> None:
        super().__init__()
        self.num_views = num_views
        out_dim = 1 if num_views == 2 else num_views
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.net = nn.Linear(num_features, out_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns per-view weights of shape (N, num_views)."""
        raw = self.net(features)
        if self.num_views == 2:
            street = torch.sigmoid(raw).squeeze(-1)
            return torch.stack([street, 1.0 - street], dim=-1)
        return torch.softmax(raw, dim=-1)


def _view_stack(data: SplitData, num_views: int) -> np.ndarray:
    views = [data.street_probs, data.remote_probs]
    if num_views == 3:
        if data.crossview_probs is None:
            raise ValueError("crossview probs required for 3-view gate")
        views.append(data.crossview_probs)
    return np.stack(views, axis=1)  # (N, num_views, C)


def train_gate(
    train_data: SplitData,
    features_normalized: np.ndarray,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    num_views: int = 2,
    sample_weights: np.ndarray | None = None,
) -> GateModel:
    torch.manual_seed(seed)
    features = torch.tensor(features_normalized, dtype=torch.float32)
    views = torch.tensor(_view_stack(train_data, num_views), dtype=torch.float32)
    targets = torch.tensor(train_data.targets, dtype=torch.long)
    weights = None
    if sample_weights is not None:
        weights = torch.tensor(sample_weights / sample_weights.mean(), dtype=torch.float32)
    model = GateModel(features.shape[1], hidden_dim, num_views)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        view_weights = model(features).unsqueeze(-1)  # (N, num_views, 1)
        mixture = (view_weights * views).sum(dim=1)
        log_mixture = torch.log(torch.clamp(mixture, min=1e-12))
        per_sample = nn.functional.nll_loss(log_mixture, targets, reduction="none")
        loss = (per_sample * weights).mean() if weights is not None else per_sample.mean()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def apply_gate(
    model: GateModel, data: SplitData, features_normalized: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    prediction, view_weights, _ = apply_gate_details(model, data, features_normalized)
    return prediction, view_weights[:, 0]


def apply_gate_details(
    model: GateModel, data: SplitData, features_normalized: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return selected classes, every gate weight, and mixture probabilities."""
    features = torch.tensor(features_normalized, dtype=torch.float32)
    with torch.no_grad():
        view_weights = model(features).numpy()  # (N, num_views)
    mixture = (view_weights[:, :, None] * _view_stack(data, model.num_views)).sum(axis=1)
    return mixture.argmax(axis=1), view_weights, mixture


def _gate_artifact_id(
    two_view_model: GateModel,
    three_view_model: GateModel,
    temperatures: dict[str, float],
    gate_fit: SplitData,
) -> str:
    """Fingerprint the fixed gates and every fitted preprocessing dependency."""
    digest = hashlib.sha256()
    digest.update(b"crossview-gate-artifact-v1\0")
    digest.update(
        json.dumps(temperatures, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name, array in (
        ("features", gate_fit.features),
        ("street_probs", gate_fit.street_probs),
        ("remote_probs", gate_fit.remote_probs),
        ("crossview_probs", gate_fit.crossview_probs),
        ("targets", gate_fit.targets),
        ("normalization_mean", gate_fit.features.mean(axis=0)),
        ("normalization_std", gate_fit.features.std(axis=0)),
    ):
        if array is None:
            raise ValueError(f"Cannot fingerprint gate artifact without {name}")
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(contiguous.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    if gate_fit.sample_ids is None:
        raise ValueError("Cannot fingerprint gate artifact without gate-fit sample ids")
    for sample_id in gate_fit.sample_ids:
        digest.update(_canonical_sample_id(sample_id).encode("utf-8") + b"\0")
    for model_name, model in (
        ("two_view", two_view_model),
        ("three_view", three_view_model),
    ):
        for parameter_name, tensor in sorted(model.state_dict().items()):
            digest.update(f"{model_name}:{parameter_name}".encode("utf-8") + b"\0")
            digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _validated_gate_mixture(
    data: SplitData, view_weights: np.ndarray, num_views: int
) -> np.ndarray:
    """Validate per-view weights and return their probability mixture."""
    weights = np.asarray(view_weights, dtype=np.float64)
    expected_shape = (len(data.targets), num_views)
    if weights.shape != expected_shape:
        raise ValueError(f"Expected {num_views}-view weights shaped {expected_shape}, got {weights.shape}")
    if not np.isfinite(weights).all():
        raise ValueError("Gate weights must all be finite")
    if (weights < -1e-7).any() or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Gate weights must be non-negative and sum to one per sample")
    return (weights[:, :, None] * _view_stack(data, num_views)).sum(axis=1)


def merge_manifest_metadata(
    evidence: pd.DataFrame, manifest: pd.DataFrame | None
) -> pd.DataFrame:
    """Attach stable optional manifest fields without changing evidence row order.

    Metadata is joined one-to-one using canonicalized ``sample_id``. The output
    always contains ``objectid``, ``location``, ``latitude``, ``longitude``, and
    ``tile_id``; absent manifest fields remain null. If only latitude/longitude
    are available, location is emitted as WKT ``POINT (longitude latitude)``.
    A few common aliases are accepted so existing split manifests do not need
    to be rewritten.
    """
    output = evidence.copy()
    output["sample_id"] = output["sample_id"].map(_canonical_sample_id)
    if manifest is not None:
        if "sample_id" not in manifest.columns:
            raise ValueError("Split manifest must contain sample_id")
        source = manifest.copy()
        source["sample_id"] = source["sample_id"].map(_canonical_sample_id)
        lower_columns = {str(column).lower(): column for column in source.columns}
        aliases = {
            "objectid": ("objectid", "object_id"),
            "location": ("location", "position"),
            "latitude": ("latitude", "lat"),
            "longitude": ("longitude", "lon", "lng"),
            "tile_id": (
                "tile_id",
                "tileid",
                "spatial_tile_id",
                "spatial_block_id",
                "remote_tile_filename",
            ),
            "group_id": ("group_id",),
            "sequence_id": ("sequence_id",),
            "spatial_block_id": ("spatial_block_id",),
            "protocol_role": ("protocol_role",),
            "event_id": ("event_id",),
            "event": ("event",),
            "source_dataset": ("dataset", "source_dataset"),
        }
        metadata = pd.DataFrame({"sample_id": source["sample_id"]})
        for output_column, candidates in aliases.items():
            source_column = next(
                (lower_columns[candidate] for candidate in candidates if candidate in lower_columns),
                None,
            )
            if source_column is not None:
                metadata[output_column] = source[source_column]
        if (
            "location" not in metadata.columns
            and "latitude" in metadata.columns
            and "longitude" in metadata.columns
        ):
            locations: list[object] = []
            for latitude, longitude in zip(metadata["latitude"], metadata["longitude"]):
                if pd.isna(latitude) or pd.isna(longitude):
                    locations.append(pd.NA)
                else:
                    locations.append(
                        f"POINT ({float(longitude):.12g} {float(latitude):.12g})"
                    )
            metadata["location"] = locations
        if metadata["sample_id"].duplicated().any():
            duplicates = sorted(metadata.loc[metadata["sample_id"].duplicated(), "sample_id"].unique())
            raise ValueError(
                "Split manifest sample_id must be unique after canonicalization; "
                f"duplicates include {duplicates[:5]}"
            )
        output = output.merge(
            metadata,
            on="sample_id",
            how="left",
            sort=False,
            validate="one_to_one",
        )
    for column in EVIDENCE_METADATA_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA

    identifier_columns = [
        "schema_version",
        "dataset",
        "seed",
        "sample_id",
        *EVIDENCE_METADATA_COLUMNS,
        "num_classes",
        "target",
    ]
    ordered = [column for column in identifier_columns if column in output.columns]
    ordered.extend(column for column in output.columns if column not in ordered)
    return output[ordered]


def build_evidence_frame(
    data: SplitData,
    two_view_weights: np.ndarray,
    three_view_weights: np.ndarray,
    *,
    dataset: str | None = None,
    seed: int | None = None,
    manifest: pd.DataFrame | None = None,
    gate_artifact_id: str | None = None,
) -> pd.DataFrame:
    """Build the auditable P0.2 per-sample evidence table.

    The schema uses ``{view}_probability_{class_index}`` for temperature-
    calibrated view probabilities and analogous ``gate2``/``gate3`` columns for
    the resulting mixtures. Class-indexed columns are generated from the input
    class count. ``js_divergence`` is the street/remote value used as a gate
    feature; the explicitly named pairwise JS columns are also retained.
    ``gate_linear_prediction`` and ``gate3_linear_prediction`` are compatibility
    aliases for the selected two- and three-view mixture predictions.
    """
    street_probs = np.asarray(data.street_probs, dtype=np.float64)
    remote_probs = np.asarray(data.remote_probs, dtype=np.float64)
    if data.crossview_probs is None:
        raise ValueError("crossview probabilities are required for the evidence table")
    crossview_probs = np.asarray(data.crossview_probs, dtype=np.float64)
    probability_shapes = {street_probs.shape, remote_probs.shape, crossview_probs.shape}
    if len(probability_shapes) != 1 or street_probs.ndim != 2:
        raise ValueError(
            "street, remote, and crossview probabilities must share a two-dimensional shape"
        )
    sample_count, num_classes = street_probs.shape
    if sample_count != len(data.targets):
        raise ValueError("Probability and target row counts do not match")
    if data.sample_ids is None or len(data.sample_ids) != sample_count:
        raise ValueError("One sample_id is required for every evidence row")
    features = np.asarray(data.features, dtype=np.float64)
    if (
        features.ndim != 2
        or features.shape[0] != sample_count
        or features.shape[1] < len(VISIBILITY_FEATURES)
    ):
        raise ValueError(
            f"features must have {sample_count} rows and at least {len(VISIBILITY_FEATURES)} columns"
        )

    two_view_weights = np.asarray(two_view_weights, dtype=np.float64)
    three_view_weights = np.asarray(three_view_weights, dtype=np.float64)
    gate2_probs = _validated_gate_mixture(data, two_view_weights, 2)
    gate3_probs = _validated_gate_mixture(data, three_view_weights, 3)

    street_prediction = street_probs.argmax(axis=1)
    remote_prediction = remote_probs.argmax(axis=1)
    crossview_prediction = crossview_probs.argmax(axis=1)
    gate2_prediction = gate2_probs.argmax(axis=1)
    gate3_prediction = gate3_probs.argmax(axis=1)
    hard_conflict = (street_prediction != remote_prediction).astype(np.int64)

    frame = pd.DataFrame(
        {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "dataset": dataset if dataset is not None else pd.NA,
            "seed": seed if seed is not None else pd.NA,
            "gate_artifact_id": gate_artifact_id if gate_artifact_id is not None else pd.NA,
            "sample_id": [_canonical_sample_id(value) for value in data.sample_ids],
            "num_classes": num_classes,
            "target": np.asarray(data.targets, dtype=np.int64),
            "street_prediction": street_prediction,
            "remote_prediction": remote_prediction,
            "crossview_prediction": crossview_prediction,
            "gate2_selected_prediction": gate2_prediction,
            "gate3_selected_prediction": gate3_prediction,
            "gate_linear_prediction": gate2_prediction,
            "gate3_linear_prediction": gate3_prediction,
            "gate2_weight_street": two_view_weights[:, 0],
            "gate2_weight_remote": two_view_weights[:, 1],
            "gate3_weight_street": three_view_weights[:, 0],
            "gate3_weight_remote": three_view_weights[:, 1],
            "gate3_weight_crossview": three_view_weights[:, 2],
        }
    )
    for feature_index, feature_name in enumerate(VISIBILITY_FEATURES):
        frame[feature_name] = features[:, feature_index]

    frame["street_confidence"] = street_probs.max(axis=1)
    frame["remote_confidence"] = remote_probs.max(axis=1)
    frame["crossview_confidence"] = crossview_probs.max(axis=1)
    frame["street_entropy"] = _entropy(street_probs)
    frame["remote_entropy"] = _entropy(remote_probs)
    frame["crossview_entropy"] = _entropy(crossview_probs)
    frame["confidence_gap"] = frame["street_confidence"] - frame["remote_confidence"]
    frame["js_divergence"] = _js_divergence(street_probs, remote_probs)
    frame["street_remote_js_divergence"] = frame["js_divergence"]
    frame["street_crossview_js_divergence"] = _js_divergence(street_probs, crossview_probs)
    frame["remote_crossview_js_divergence"] = _js_divergence(remote_probs, crossview_probs)
    frame["views_disagree"] = hard_conflict
    frame["hard_conflict"] = hard_conflict

    for class_index in range(num_classes):
        for view_name, probabilities in zip(
            VIEW_NAMES, (street_probs, remote_probs, crossview_probs)
        ):
            frame[f"{view_name}_probability_{class_index}"] = probabilities[:, class_index]
        frame[f"gate2_probability_{class_index}"] = gate2_probs[:, class_index]
        frame[f"gate3_probability_{class_index}"] = gate3_probs[:, class_index]
    return merge_manifest_metadata(frame, manifest)


def mcnemar_p_value(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    b = int(((correct_a == 1) & (correct_b == 0)).sum())
    c = int(((correct_a == 0) & (correct_b == 1)).sum())
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    from math import comb

    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return float(min(1.0, 2.0 * tail))


def _macro_f1(target: np.ndarray, prediction: np.ndarray) -> float:
    num_classes = int(max(target.max(initial=0), prediction.max(initial=0)) + 1)
    f1_values: list[float] = []
    for class_index in range(num_classes):
        tp = float(((prediction == class_index) & (target == class_index)).sum())
        fp = float(((prediction == class_index) & (target != class_index)).sum())
        fn = float(((prediction != class_index) & (target == class_index)).sum())
        support = float((target == class_index).sum())
        if support == 0:
            continue
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1_values.append(2.0 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(f1_values)) if f1_values else 0.0


def evaluate(
    dataset: str,
    seed: int,
    variant: str,
    train_source: str,
    prediction: np.ndarray,
    gate_weight: np.ndarray | None,
    test: SplitData,
    crossview_pred: np.ndarray,
) -> dict[str, object]:
    target = test.targets
    street_pred = test.street_probs.argmax(axis=1)
    remote_pred = test.remote_probs.argmax(axis=1)
    conflict_mask = street_pred != remote_pred
    street_correct = street_pred == target
    remote_correct = remote_pred == target
    oracle_conflict = float((street_correct | remote_correct)[conflict_mask].mean())
    best_single_conflict = max(
        float(street_correct[conflict_mask].mean()), float(remote_correct[conflict_mask].mean())
    )
    correct = (prediction == target).astype(np.int64)
    conflict_accuracy = float(correct[conflict_mask].mean())
    gap = oracle_conflict - best_single_conflict
    crossview_correct = (crossview_pred == target).astype(np.int64)
    row: dict[str, object] = {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "train_source": train_source,
        "accuracy": float(correct.mean()),
        "macro_f1": _macro_f1(target, prediction),
        "conflict_n": int(conflict_mask.sum()),
        "conflict_accuracy": conflict_accuracy,
        "best_single_conflict_accuracy": best_single_conflict,
        "oracle_conflict_accuracy": oracle_conflict,
        "oracle_gap_closure": (conflict_accuracy - best_single_conflict) / gap if gap > 1e-9 else None,
        "mcnemar_p_vs_crossview_conflict": mcnemar_p_value(
            correct[conflict_mask], crossview_correct[conflict_mask]
        ),
        "delta_vs_crossview_conflict": float(
            correct[conflict_mask].mean() - crossview_correct[conflict_mask].mean()
        ),
    }
    if gate_weight is not None:
        row["mean_street_weight"] = float(gate_weight.mean())
        row["mean_street_weight_on_conflicts"] = float(gate_weight[conflict_mask].mean())
    return row


def main() -> None:
    args = parse_args()
    root = Path(args.multiseed_root)
    visibility_dir = Path(args.visibility_dir)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    split_manifest_paths = _parse_split_manifest_map(args.split_manifests)
    risk_protocol_paths = _parse_split_manifest_map(args.risk_protocol_manifests)

    rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    cache: dict[tuple[str, int, str], SplitData] = {}
    crossview_preds: dict[tuple[str, int], np.ndarray] = {}
    manifest_cache: dict[str, pd.DataFrame] = {}
    protocol_manifests: dict[tuple[str, str], pd.DataFrame] = {}

    for dataset, directory in risk_protocol_paths.items():
        if dataset not in datasets:
            continue
        for role in ("gate_fit", "risk_calibration", "final_test"):
            protocol_manifests[(dataset, role)] = _read_protocol_manifest(
                directory, role
            )

    for dataset in datasets:
        for seed in seeds:
            gate_fit_ids = None
            if dataset in risk_protocol_paths:
                gate_fit_ids = set(
                    protocol_manifests[(dataset, "gate_fit")]["sample_id"].astype(str)
                )
            temperatures: dict[str, float] = {}
            for mode in ("street_only", "remote_only", "crossview"):
                val_path = root / dataset / f"{mode}_seed{seed}" / "val_predictions.csv"
                df = pd.read_csv(val_path)
                if gate_fit_ids is not None:
                    df = _filter_prediction_frame(
                        df,
                        gate_fit_ids,
                        source=f"{dataset} seed={seed} {mode} gate_fit",
                    )
                logit_cols = _numbered_columns(df, "logit")
                temperatures[mode] = fit_temperature(
                    df[logit_cols].to_numpy(dtype=np.float64), df["target"].to_numpy(dtype=np.int64)
                )
            full_val = load_split(
                root, visibility_dir, dataset, seed, "val", temperatures
            )
            full_test = load_split(
                root, visibility_dir, dataset, seed, "test", temperatures
            )
            if dataset in risk_protocol_paths:
                role_ids = {
                    role: set(
                        protocol_manifests[(dataset, role)]["sample_id"].astype(str)
                    )
                    for role in ("gate_fit", "risk_calibration", "final_test")
                }
                if role_ids["gate_fit"] & role_ids["risk_calibration"]:
                    raise ValueError(f"{dataset} gate-fit and risk-calibration ids overlap")
                cache[(dataset, seed, "gate_fit")] = _subset_split_data(
                    full_val, role_ids["gate_fit"], role="gate_fit"
                )
                cache[(dataset, seed, "risk_calibration")] = _subset_split_data(
                    full_val, role_ids["risk_calibration"], role="risk_calibration"
                )
                cache[(dataset, seed, "test")] = _subset_split_data(
                    full_test, role_ids["final_test"], role="final_test"
                )
            else:
                cache[(dataset, seed, "gate_fit")] = full_val
                cache[(dataset, seed, "test")] = full_test
            crossview_preds[(dataset, seed)] = cache[(dataset, seed, "test")].crossview_probs.argmax(axis=1)

    def _zscore(features: np.ndarray, reference: np.ndarray) -> np.ndarray:
        mean = reference.mean(axis=0)
        std = np.clip(reference.std(axis=0), 1e-6, None)
        return (features - mean) / std

    def pooled_split(source_datasets: list[str], seed: int) -> tuple[SplitData, np.ndarray, np.ndarray]:
        parts = [cache[(name, seed, "gate_fit")] for name in source_datasets]
        data = SplitData(
            np.concatenate([part.features for part in parts]),
            np.concatenate([part.street_probs for part in parts]),
            np.concatenate([part.remote_probs for part in parts]),
            np.concatenate([part.targets for part in parts]),
            np.concatenate([part.crossview_probs for part in parts]),
        )
        znorm_features = np.concatenate([_zscore(part.features, part.features) for part in parts])
        balance = np.concatenate(
            [np.full(len(part.targets), 1.0 / len(part.targets)) for part in parts]
        )
        return data, znorm_features, balance

    variants = (("gate_mlp", args.hidden_dim), ("gate_linear", 0))

    for target_dataset in datasets:
        for seed in seeds:
            test = cache[(target_dataset, seed, "test")]
            crossview_pred = crossview_preds[(target_dataset, seed)]
            target_gate_fit = cache[(target_dataset, seed, "gate_fit")]
            other_datasets = [name for name in datasets if name != target_dataset]
            if other_datasets:
                pooled, pooled_znorm, balance = pooled_split(other_datasets, seed)
                source_label = "+".join(other_datasets)
                # Raw pooled: shared source statistics, unbalanced (reference).
                raw_normalized = _zscore(pooled.features, pooled.features)
                test_raw_normalized = _zscore(test.features, pooled.features)
                # Regime-normalized pooled: each source dataset z-scored with its
                # own statistics, balanced sampling; the target is normalized with
                # its own (label-free) val statistics.
                test_znorm = _zscore(test.features, target_gate_fit.features)
                for variant, hidden in variants:
                    model = train_gate(
                        pooled, raw_normalized, hidden, args.epochs, args.learning_rate,
                        args.weight_decay, seed,
                    )
                    prediction, weight = apply_gate(model, test, test_raw_normalized)
                    rows.append(
                        evaluate(
                            target_dataset, seed, f"{variant}_loo_pooled", source_label,
                            prediction, weight, test, crossview_pred,
                        )
                    )
                    model = train_gate(
                        pooled, pooled_znorm, hidden, args.epochs, args.learning_rate,
                        args.weight_decay, seed, sample_weights=balance,
                    )
                    prediction, weight = apply_gate(model, test, test_znorm)
                    rows.append(
                        evaluate(
                            target_dataset, seed, f"{variant}_loo_pooled_znorm", source_label,
                            prediction, weight, test, crossview_pred,
                        )
                    )
            for source_dataset in datasets:
                train = cache[(source_dataset, seed, "gate_fit")]
                train_normalized = _zscore(train.features, train.features)
                test_normalized = _zscore(test.features, train.features)
                scope = "in_domain" if source_dataset == target_dataset else "transfer"
                for variant, hidden in variants:
                    model = train_gate(
                        train, train_normalized, hidden, args.epochs, args.learning_rate,
                        args.weight_decay, seed,
                    )
                    prediction, weight = apply_gate(model, test, test_normalized)
                    rows.append(
                        evaluate(
                            target_dataset, seed, f"{variant}_{scope}", source_dataset,
                            prediction, weight, test, crossview_pred,
                        )
                    )
                    if variant == "gate_linear" and source_dataset == target_dataset:
                        linear = model.net
                        for name, value in zip(ALL_FEATURES, linear.weight.detach().numpy().ravel()):
                            weight_rows.append(
                                {"dataset": target_dataset, "seed": seed, "feature": name, "coefficient": float(value)}
                            )
                    if source_dataset == target_dataset:
                        if variant == "gate_linear":
                            two_view_model = model
                            evidence_roles: dict[str, SplitData] = {"final_test": test}
                            if target_dataset in risk_protocol_paths:
                                evidence_roles = {
                                    "gate_fit": train,
                                    "risk_calibration": cache[
                                        (target_dataset, seed, "risk_calibration")
                                    ],
                                    "final_test": test,
                                }
                            role_normalized = {
                                role: _zscore(data.features, train.features)
                                for role, data in evidence_roles.items()
                            }
                            two_view_weights_by_role = {
                                role: apply_gate_details(model, data, role_normalized[role])[1]
                                for role, data in evidence_roles.items()
                            }
                        model = train_gate(
                            train, train_normalized, hidden, args.epochs, args.learning_rate,
                            args.weight_decay, seed, num_views=3,
                        )
                        prediction, weight = apply_gate(model, test, test_normalized)
                        rows.append(
                            evaluate(
                                target_dataset, seed, f"gate3_{'mlp' if hidden else 'linear'}_in_domain",
                                source_dataset, prediction, weight, test, crossview_pred,
                            )
                        )
                        if variant == "gate_linear":
                            artifact_id = _gate_artifact_id(
                                two_view_model, model, temperatures, train
                            )
                            pred_dir = Path(args.output_dir) / "predictions"
                            ensure_dir(pred_dir)
                            for role, role_data in evidence_roles.items():
                                three_view_weights = apply_gate_details(
                                    model, role_data, role_normalized[role]
                                )[1]
                                if target_dataset in risk_protocol_paths:
                                    manifest = protocol_manifests[(target_dataset, role)]
                                else:
                                    manifest = None
                                    if target_dataset in split_manifest_paths:
                                        if target_dataset not in manifest_cache:
                                            manifest_cache[target_dataset] = _read_test_manifest(
                                                split_manifest_paths[target_dataset]
                                            )
                                        manifest = manifest_cache[target_dataset]
                                evidence = build_evidence_frame(
                                    role_data,
                                    two_view_weights_by_role[role],
                                    three_view_weights,
                                    dataset=target_dataset,
                                    seed=seed,
                                    manifest=manifest,
                                    gate_artifact_id=artifact_id,
                                )
                                role_path = (
                                    pred_dir
                                    / f"{target_dataset}_seed{seed}_{role}.csv"
                                )
                                evidence.to_csv(role_path, index=False)
                                if role == "final_test":
                                    # Backward-compatible alias for analysis scripts
                                    # that predate explicit protocol-role filenames.
                                    evidence.to_csv(
                                        pred_dir / f"{target_dataset}_seed{seed}.csv",
                                        index=False,
                                    )
            print(f"gated {target_dataset} seed={seed}")

    results = pd.DataFrame(rows)
    coefficients = pd.DataFrame(weight_rows)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    results.to_csv(output_dir / "gate_results_raw.csv", index=False)
    coefficients.to_csv(output_dir / "gate_linear_coefficients.csv", index=False)

    numeric = ["accuracy", "macro_f1", "conflict_accuracy", "oracle_gap_closure", "delta_vs_crossview_conflict"]
    summary = (
        results.groupby(["dataset", "variant", "train_source"])[numeric]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(col).rstrip("_") for col in summary.columns]
    summary.to_csv(output_dir / "gate_results_summary.csv", index=False)

    lines = [
        "# Reliability Gate Results (Phase 1)",
        "",
        "Gate: per-sample street-vs-remote mixture weight predicted from building",
        "visibility, calibrated confidence/entropy, confidence gap, JS divergence,",
        "and a disagreement flag. Trained on gate-fit (legacy runs use full val) by",
        "minimizing the NLL of the gated mixture; evaluated on final test. `transfer`",
        "rows train the gate on a different dataset's gate-fit split.",
        "",
        "| dataset | variant | train source | accuracy | macro_f1 | conflict_acc | gap closure | sig vs crossview |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (dataset, variant, source), group in results.groupby(["dataset", "variant", "train_source"]):
        sig = int((group["mcnemar_p_vs_crossview_conflict"] < 0.05).sum())
        def fmt(column: str) -> str:
            return f"{group[column].mean():.4f} +/- {group[column].std():.4f}"
        lines.append(
            f"| {dataset} | {variant} | {source} | {fmt('accuracy')} | {fmt('macro_f1')} | "
            f"{fmt('conflict_accuracy')} | {fmt('oracle_gap_closure')} | {sig}/{len(group)} |"
        )
    if not coefficients.empty:
        lines += [
            "",
            "## Linear gate coefficients (in-domain, positive = trust street view)",
            "",
            "| dataset | feature | coefficient |",
            "| --- | --- | --- |",
        ]
        coef_summary = coefficients.groupby(["dataset", "feature"])["coefficient"].agg(["mean", "std"]).reset_index()
        for _, row in coef_summary.iterrows():
            lines.append(f"| {row['dataset']} | {row['feature']} | {row['mean']:.4f} +/- {row['std']:.4f} |")
    doc_path = Path(args.doc_path)
    ensure_dir(doc_path.parent)
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {doc_path}")


if __name__ == "__main__":
    main()
