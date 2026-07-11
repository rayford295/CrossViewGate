from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from crossview_conflict.models.backbones import get_backbone_normalization
from crossview_conflict.utils.image import build_transform, load_rgb_image


class _BaseManifestDataset(Dataset):
    def __init__(self, manifest_csv: str | Path) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.df = pd.read_csv(self.manifest_csv).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def metadata_row(self, index: int) -> dict[str, Any]:
        return self.df.iloc[index].to_dict()


def _resolve_label_column(df: pd.DataFrame, label_col: str | None) -> str:
    if label_col and label_col != "auto":
        if label_col not in df.columns:
            raise KeyError(f"Column '{label_col}' not found in manifest.")
        return label_col
    for candidate in ("label", "binary_label"):
        if candidate in df.columns:
            return candidate
    raise KeyError("Manifest must include either a 'label' or 'binary_label' column.")


TRIAGE_VIEW_NAMES = ("pre_street", "post_street", "pre_overhead", "post_overhead")
_DEFAULT_TRIAGE_VIEWS = ("post_street", "post_overhead")
_TRIAGE_VIEW_COLUMN_CANDIDATES = {
    "pre_street": (
        "pre_street_view_path",
        "pre_street_path",
        "pre_street",
    ),
    "post_street": (
        "street_view_path",
        "post_street_view_path",
        "post_street_path",
        "post_street",
    ),
    "pre_overhead": (
        "pre_overhead_view_path",
        "pre_remote_sensing_path",
        "pre_overhead_path",
        "pre_overhead",
    ),
    "post_overhead": (
        "remote_sensing_path",
        "post_overhead_view_path",
        "post_remote_sensing_path",
        "post_overhead_path",
        "post_overhead",
    ),
}
_TRIAGE_VIEW_AVAILABILITY_CANDIDATES = {
    "pre_street": ("pre_street_view_available", "pre_street_available"),
    "post_street": ("post_street_view_available", "post_street_available"),
    "pre_overhead": ("pre_overhead_view_available", "pre_overhead_available"),
    "post_overhead": ("post_overhead_view_available", "post_overhead_available"),
}


def _optional_manifest_boolean(value: object, *, column: str, sample_id: object) -> bool | None:
    """Parse an optional manifest flag without treating non-empty strings as true."""
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(
        f"Invalid availability flag {value!r} (sample_id={sample_id!r}, column='{column}')."
    )


def _normalize_triage_views(views: Sequence[str] | str | None) -> tuple[str, ...]:
    if views is None:
        normalized = _DEFAULT_TRIAGE_VIEWS
    elif isinstance(views, str):
        normalized = (views,)
    else:
        normalized = tuple(views)

    unknown = [view for view in normalized if view not in TRIAGE_VIEW_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown triage view(s): {unknown}. Expected a subset of {TRIAGE_VIEW_NAMES}."
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("Triage views must not contain duplicates.")
    return normalized


def _normalize_view_dropout(
    views: Sequence[str],
    view_dropout: float | Mapping[str, float],
) -> dict[str, float]:
    if isinstance(view_dropout, Mapping):
        unknown = [view for view in view_dropout if view not in TRIAGE_VIEW_NAMES]
        if unknown:
            raise ValueError(
                f"Unknown view_dropout key(s): {unknown}. Expected names from {TRIAGE_VIEW_NAMES}."
            )
        specified_probabilities = {view: float(probability) for view, probability in view_dropout.items()}
        probabilities = {view: specified_probabilities.get(view, 0.0) for view in views}
    else:
        probability = float(view_dropout)
        specified_probabilities = {"all": probability}
        probabilities = {view: probability for view in views}

    invalid = {
        view: probability
        for view, probability in specified_probabilities.items()
        if not 0.0 <= probability <= 1.0
    }
    if invalid:
        raise ValueError(f"view_dropout probabilities must be between 0 and 1; got {invalid}.")
    return probabilities


class CrossViewRetrievalDataset(_BaseManifestDataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        street_size: int = 224,
        overhead_size: int = 224,
        normalize: bool = True,
        street_augment: bool = False,
        overhead_augment: bool = False,
        street_backbone: str = "resnet18",
        overhead_backbone: str = "resnet18",
        load_street: bool = True,
        load_overhead: bool = True,
    ) -> None:
        super().__init__(manifest_csv)
        self.load_street = load_street
        self.load_overhead = load_overhead
        street_mean, street_std = get_backbone_normalization(street_backbone)
        overhead_mean, overhead_std = get_backbone_normalization(overhead_backbone)
        self.street_transform = (
            build_transform(
                street_size,
                normalize=normalize,
                augment=street_augment,
                domain="street",
                mean=street_mean,
                std=street_std,
            )
            if load_street
            else None
        )
        self.overhead_transform = (
            build_transform(
                overhead_size,
                normalize=normalize,
                augment=overhead_augment,
                domain="overhead",
                mean=overhead_mean,
                std=overhead_std,
            )
            if load_overhead
            else None
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        batch: dict[str, Any] = {
            "target": index,
            "sample_id": row["sample_id"],
            "objectid": int(row["objectid"]) if not pd.isna(row["objectid"]) else -1,
            "latitude": float(row["latitude"]) if "latitude" in row and not pd.isna(row["latitude"]) else float("nan"),
            "longitude": float(row["longitude"]) if "longitude" in row and not pd.isna(row["longitude"]) else float("nan"),
            "remote_tile_filename": (
                str(row["remote_tile_filename"])
                if "remote_tile_filename" in row and not pd.isna(row["remote_tile_filename"])
                else ""
            ),
        }
        if self.load_street and self.street_transform is not None:
            batch["street"] = self.street_transform(load_rgb_image(row["street_view_path"]))
        if self.load_overhead and self.overhead_transform is not None:
            batch["overhead"] = self.overhead_transform(load_rgb_image(row["remote_sensing_path"]))
        return batch


class CrossViewTriageDataset(_BaseManifestDataset):
    """Triage data with configurable temporal views and explicit availability masks.

    ``view_names`` defines the order of every returned mask. A blank manifest
    cell (or an absent optional column) is a real missing view. ``view_dropout``
    only masks views whose manifest paths are present, so ``dropout_view_mask``
    never conflates artificial dropout with real missing data.

    The default views and the ``street``/``overhead`` aliases preserve the
    original post-event two-view interface.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        street_size: int = 224,
        overhead_size: int = 224,
        normalize: bool = True,
        include_generated: bool = False,
        street_augment: bool = False,
        overhead_augment: bool = False,
        street_backbone: str = "resnet18",
        overhead_backbone: str = "resnet18",
        label_col: str | None = "auto",
        views: Sequence[str] | str | None = None,
        view_columns: Mapping[str, str] | None = None,
        view_dropout: float | Mapping[str, float] = 0.0,
        view_dropout_seed: int | None = None,
    ) -> None:
        super().__init__(manifest_csv)
        self.label_col = _resolve_label_column(self.df, label_col)
        self.view_names = _normalize_triage_views(views)
        column_overrides = dict(view_columns or {})
        unknown_columns = [view for view in column_overrides if view not in TRIAGE_VIEW_NAMES]
        if unknown_columns:
            raise ValueError(
                f"Unknown view_columns key(s): {unknown_columns}. Expected names from {TRIAGE_VIEW_NAMES}."
            )
        self.view_columns: dict[str, str | None] = {}
        self.view_availability_columns: dict[str, str | None] = {}
        for view in self.view_names:
            if view in column_overrides:
                column = column_overrides[view]
                if column not in self.df.columns:
                    raise KeyError(f"Column '{column}' configured for view '{view}' was not found in manifest.")
                self.view_columns[view] = column
            else:
                self.view_columns[view] = next(
                    (
                        column
                        for column in _TRIAGE_VIEW_COLUMN_CANDIDATES[view]
                        if column in self.df.columns
                    ),
                    None,
                )
            self.view_availability_columns[view] = next(
                (
                    column
                    for column in _TRIAGE_VIEW_AVAILABILITY_CANDIDATES[view]
                    if column in self.df.columns
                ),
                None,
            )
        self.view_dropout = _normalize_view_dropout(self.view_names, view_dropout)
        self.view_dropout_seed = view_dropout_seed
        self.street_size = street_size
        self.overhead_size = overhead_size
        street_mean, street_std = get_backbone_normalization(street_backbone)
        overhead_mean, overhead_std = get_backbone_normalization(overhead_backbone)
        self.street_transform = build_transform(
            street_size,
            normalize=normalize,
            augment=street_augment,
            domain="street",
            mean=street_mean,
            std=street_std,
        )
        self.overhead_transform = build_transform(
            overhead_size,
            normalize=normalize,
            augment=overhead_augment,
            domain="overhead",
            mean=overhead_mean,
            std=overhead_std,
        )
        self.include_generated = include_generated

    @staticmethod
    def _is_street_view(view: str) -> bool:
        return view in {"pre_street", "post_street"}

    def _empty_view(self, view: str) -> torch.Tensor:
        size = self.street_size if self._is_street_view(view) else self.overhead_size
        return torch.zeros((3, size, size), dtype=torch.float32)

    def _manifest_path(self, row: pd.Series, view: str) -> Path | None:
        sample_id = row["sample_id"] if "sample_id" in row else row.name
        availability_column = self.view_availability_columns[view]
        if availability_column is not None:
            availability = _optional_manifest_boolean(
                row[availability_column],
                column=availability_column,
                sample_id=sample_id,
            )
            if availability is False:
                return None
        column = self.view_columns[view]
        if column is None:
            return None
        value = row[column]
        if pd.isna(value) or (isinstance(value, str) and not value.strip()):
            return None
        path = Path(str(value))
        if not path.is_file():
            raise FileNotFoundError(
                f"Manifest path for view '{view}' does not exist or is not a file "
                f"(sample_id={sample_id!r}, column='{column}', path='{path}')."
            )
        return path

    def _should_drop_view(self, index: int, view: str) -> bool:
        probability = self.view_dropout[view]
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        if self.view_dropout_seed is None:
            return bool(torch.rand(()).item() < probability)

        generator = torch.Generator()
        view_index = TRIAGE_VIEW_NAMES.index(view)
        seed = (int(self.view_dropout_seed) + index * len(TRIAGE_VIEW_NAMES) + view_index) % (2**63 - 1)
        generator.manual_seed(seed)
        return bool(torch.rand((), generator=generator).item() < probability)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        target = int(row[self.label_col])
        batch: dict[str, Any] = {
            "target": torch.tensor(target, dtype=torch.long),
            "sample_id": row["sample_id"],
        }

        available_mask: list[bool] = []
        dropout_mask: list[bool] = []
        visible_mask: list[bool] = []
        for view in self.view_names:
            path = self._manifest_path(row, view)
            available = path is not None
            dropped = available and self._should_drop_view(index, view)
            visible = available and not dropped
            if visible and path is not None:
                transform = self.street_transform if self._is_street_view(view) else self.overhead_transform
                image = transform(load_rgb_image(path))
            else:
                image = self._empty_view(view)
            batch[view] = image
            if view == "post_street":
                batch["street"] = image
            elif view == "post_overhead":
                batch["overhead"] = image
            available_mask.append(available)
            dropout_mask.append(dropped)
            visible_mask.append(visible)

        # A temporal experiment may request only a pre-event view. Expose the
        # selected street/overhead tensor through the model's stable aliases,
        # preferring post-event evidence whenever it is present.
        if "street" not in batch:
            for candidate in ("post_street", "pre_street"):
                if candidate in batch:
                    batch["street"] = batch[candidate]
                    break
        if "overhead" not in batch:
            for candidate in ("post_overhead", "pre_overhead"):
                if candidate in batch:
                    batch["overhead"] = batch[candidate]
                    break

        batch["view_mask"] = torch.tensor(visible_mask, dtype=torch.bool)
        batch["available_view_mask"] = torch.tensor(available_mask, dtype=torch.bool)
        batch["missing_view_mask"] = ~batch["available_view_mask"]
        batch["dropout_view_mask"] = torch.tensor(dropout_mask, dtype=torch.bool)
        if self.include_generated and "generated_street_path" in row and isinstance(row["generated_street_path"], str):
            generated_path = row["generated_street_path"]
            if generated_path and Path(generated_path).exists():
                batch["generated"] = self.street_transform(load_rgb_image(generated_path))
        return batch


class CrossViewGenerationDataset(_BaseManifestDataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        street_size: int = 256,
        overhead_size: int = 256,
    ) -> None:
        super().__init__(manifest_csv)
        self.street_transform = build_transform(street_size, normalize=False)
        self.overhead_transform = build_transform(overhead_size, normalize=False)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        return {
            "source": self.overhead_transform(load_rgb_image(row["remote_sensing_path"])),
            "target": self.street_transform(load_rgb_image(row["street_view_path"])),
            "sample_id": row["sample_id"],
            "binary_label": int(row["binary_label"]) if not pd.isna(row["binary_label"]) else -1,
        }


class CrossViewTileDataset(_BaseManifestDataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        tile_to_index: dict[str, int],
        street_size: int = 224,
        normalize: bool = True,
        street_augment: bool = False,
    ) -> None:
        super().__init__(manifest_csv)
        self.tile_to_index = dict(tile_to_index)
        self.street_transform = build_transform(
            street_size,
            normalize=normalize,
            augment=street_augment,
            domain="street",
        )
        valid_mask = self.df["remote_tile_filename"].astype(str).isin(self.tile_to_index)
        self.df = self.df[valid_mask].reset_index(drop=True)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        tile_name = str(row["remote_tile_filename"])
        return {
            "street": self.street_transform(load_rgb_image(row["street_view_path"])),
            "target": torch.tensor(self.tile_to_index[tile_name], dtype=torch.long),
            "sample_id": row["sample_id"],
            "tile_name": tile_name,
            "objectid": int(row["objectid"]) if not pd.isna(row["objectid"]) else -1,
        }


class GroupedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: _BaseManifestDataset,
        group_column: str,
        batch_size: int,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if group_column not in dataset.df.columns:
            raise KeyError(f"Column '{group_column}' not found in dataset manifest.")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        grouped = dataset.df.reset_index().groupby(group_column)["index"].apply(list)
        self.group_indices = {str(group_name): indices for group_name, indices in grouped.items() if str(group_name) != "nan"}
        self.group_names = sorted(self.group_indices.keys())
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self._epoch)

        batches: list[list[int]] = []
        group_order = torch.randperm(len(self.group_names), generator=generator).tolist()
        for group_position in group_order:
            group_name = self.group_names[group_position]
            indices = list(self.group_indices[group_name])
            if len(indices) == 0:
                continue
            shuffle_order = torch.randperm(len(indices), generator=generator).tolist()
            shuffled = [indices[index] for index in shuffle_order]
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        batch_order = torch.randperm(len(batches), generator=generator).tolist() if batches else []
        for batch_position in batch_order:
            yield batches[batch_position]
        self._epoch += 1

    def __len__(self) -> int:
        total = 0
        for indices in self.group_indices.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total
