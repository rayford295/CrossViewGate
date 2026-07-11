from __future__ import annotations

import unittest

import pandas as pd

from crossview_conflict.data.cvian import (
    add_spatial_block_ids,
    assert_group_isolation,
    purge_spatial_buffer,
    split_grouped_frame,
)


class CVIANGroupedSplitTest(unittest.TestCase):
    def test_spatial_block_ids_and_grouped_split_are_isolated(self) -> None:
        rows = []
        for group_index in range(15):
            for row_index in range(3):
                rows.append(
                    {
                        "sample_id": f"{group_index:02d}-{row_index}",
                        "latitude": 26.0 + group_index * 0.01,
                        "longitude": -82.0,
                        "label": (group_index + row_index) % 3,
                    }
                )
        frame = add_spatial_block_ids(pd.DataFrame(rows), grid_degrees=0.005)
        splits = split_grouped_frame(
            frame,
            group_col="spatial_block_id",
            label_col="label",
            train_fraction=0.7,
            val_fraction=0.15,
            seed=7,
        )
        self.assertEqual(sum(map(len, splits.values())), len(frame))
        self.assertTrue(all(len(split) > 0 for split in splits.values()))
        assert_group_isolation(splits, "spatial_block_id")
        self.assertEqual(
            {name: split["sample_id"].tolist() for name, split in splits.items()},
            {
                name: split["sample_id"].tolist()
                for name, split in split_grouped_frame(
                    frame,
                    group_col="spatial_block_id",
                    label_col="label",
                    train_fraction=0.7,
                    val_fraction=0.15,
                    seed=7,
                ).items()
            },
        )

    def test_invalid_fractions_fail(self) -> None:
        frame = pd.DataFrame(
            {
                "group": ["a", "b", "c"],
                "label": [0, 1, 0],
            }
        )
        with self.assertRaises(ValueError):
            split_grouped_frame(
                frame,
                group_col="group",
                label_col="label",
                train_fraction=0.9,
                val_fraction=0.2,
            )

    def test_spatial_blocking_rejects_invalid_coordinates_and_duplicate_ids(self) -> None:
        base = pd.DataFrame(
            {"sample_id": ["a"], "latitude": [26.0], "longitude": [-82.0]}
        )
        for column, value in (
            ("latitude", float("nan")),
            ("longitude", float("inf")),
            ("latitude", 999.0),
            ("longitude", -999.0),
        ):
            invalid = base.copy()
            invalid.loc[0, column] = value
            with self.assertRaises(ValueError):
                add_spatial_block_ids(invalid)
        duplicate = pd.concat([base, base], ignore_index=True)
        with self.assertRaises(ValueError):
            add_spatial_block_ids(duplicate)

    def test_spatial_buffer_preserves_test_and_purges_lower_priority_neighbors(self) -> None:
        splits = {
            "train": pd.DataFrame(
                {"sample_id": ["train-near", "train-far"], "latitude": [0.0, 1.0], "longitude": [0.0001, 1.0]}
            ),
            "val": pd.DataFrame(
                {"sample_id": ["val-near", "val-far"], "latitude": [0.0, 2.0], "longitude": [0.0002, 2.0]}
            ),
            "test": pd.DataFrame(
                {"sample_id": ["test"], "latitude": [0.0], "longitude": [0.0]}
            ),
        }
        purged, excluded = purge_spatial_buffer(splits, 25.0)
        self.assertEqual(purged["test"]["sample_id"].tolist(), ["test"])
        self.assertEqual(purged["val"]["sample_id"].tolist(), ["val-far"])
        self.assertEqual(purged["train"]["sample_id"].tolist(), ["train-far"])
        self.assertEqual(set(excluded["sample_id"]), {"train-near", "val-near"})


if __name__ == "__main__":
    unittest.main()
