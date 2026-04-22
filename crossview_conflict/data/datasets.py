from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from crossview_conflict.utils.image import build_transform, load_rgb_image


class _BaseManifestDataset(Dataset):
    def __init__(self, manifest_csv: str | Path) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.df = pd.read_csv(self.manifest_csv).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def metadata_row(self, index: int) -> dict[str, Any]:
        return self.df.iloc[index].to_dict()


class CrossViewRetrievalDataset(_BaseManifestDataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        street_size: int = 224,
        overhead_size: int = 224,
        normalize: bool = True,
        street_augment: bool = False,
        overhead_augment: bool = False,
        load_street: bool = True,
        load_overhead: bool = True,
    ) -> None:
        super().__init__(manifest_csv)
        self.load_street = load_street
        self.load_overhead = load_overhead
        self.street_transform = (
            build_transform(street_size, normalize=normalize, augment=street_augment, domain="street")
            if load_street
            else None
        )
        self.overhead_transform = (
            build_transform(overhead_size, normalize=normalize, augment=overhead_augment, domain="overhead")
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
    def __init__(
        self,
        manifest_csv: str | Path,
        street_size: int = 224,
        overhead_size: int = 224,
        normalize: bool = True,
        include_generated: bool = False,
    ) -> None:
        super().__init__(manifest_csv)
        self.street_transform = build_transform(street_size, normalize=normalize)
        self.overhead_transform = build_transform(overhead_size, normalize=normalize)
        self.include_generated = include_generated

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        street = self.street_transform(load_rgb_image(row["street_view_path"]))
        overhead = self.overhead_transform(load_rgb_image(row["remote_sensing_path"]))
        target = int(row["binary_label"])
        batch = {
            "street": street,
            "overhead": overhead,
            "target": torch.tensor(target, dtype=torch.float32),
            "sample_id": row["sample_id"],
        }
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
