from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from crossview_conflict.data.datasets import CrossViewTriageDataset, TRIAGE_VIEW_NAMES


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (6, 5), color=color).save(path)


class OptionalViewDatasetTest(unittest.TestCase):
    def test_legacy_default_keeps_street_and_overhead(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            street_path = root / "street.png"
            overhead_path = root / "overhead.png"
            _write_image(street_path, (10, 20, 30))
            _write_image(overhead_path, (40, 50, 60))
            manifest_path = root / "manifest.csv"
            pd.DataFrame(
                [
                    {
                        "sample_id": "legacy-1",
                        "street_view_path": street_path,
                        "remote_sensing_path": overhead_path,
                        "label": 2,
                    }
                ]
            ).to_csv(manifest_path, index=False)

            dataset = CrossViewTriageDataset(
                manifest_path,
                street_size=8,
                overhead_size=10,
                normalize=False,
            )
            sample = dataset[0]

            self.assertEqual(dataset.view_names, ("post_street", "post_overhead"))
            self.assertIs(sample["street"], sample["post_street"])
            self.assertIs(sample["overhead"], sample["post_overhead"])
            self.assertEqual(sample["street"].shape, (3, 8, 8))
            self.assertEqual(sample["overhead"].shape, (3, 10, 10))
            self.assertEqual(sample["target"].item(), 2)
            self.assertEqual(sample["view_mask"].tolist(), [True, True])
            self.assertEqual(sample["available_view_mask"].tolist(), [True, True])
            self.assertEqual(sample["missing_view_mask"].tolist(), [False, False])
            self.assertEqual(sample["dropout_view_mask"].tolist(), [False, False])

            batch = next(iter(DataLoader(dataset, batch_size=1)))
            self.assertEqual(batch["street"].shape, (1, 3, 8, 8))
            self.assertEqual(batch["overhead"].shape, (1, 3, 10, 10))

    def test_real_missing_and_artificial_dropout_have_distinct_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pre_street_path = root / "pre_street.png"
            post_overhead_path = root / "post_overhead.png"
            _write_image(pre_street_path, (80, 90, 100))
            _write_image(post_overhead_path, (110, 120, 130))
            manifest_path = root / "manifest.csv"
            pd.DataFrame(
                [
                    {
                        "sample_id": "temporal-1",
                        "pre_street": pre_street_path,
                        "post_street": "",
                        "pre_overhead": "",
                        "post_overhead": post_overhead_path,
                        "label": 1,
                    }
                ]
            ).to_csv(manifest_path, index=False)

            dataset = CrossViewTriageDataset(
                manifest_path,
                street_size=7,
                overhead_size=9,
                normalize=False,
                views=TRIAGE_VIEW_NAMES,
                view_dropout={"post_overhead": 1.0},
                view_dropout_seed=7,
            )
            sample = dataset[0]

            self.assertEqual(sample["available_view_mask"].tolist(), [True, False, False, True])
            self.assertEqual(sample["missing_view_mask"].tolist(), [False, True, True, False])
            self.assertEqual(sample["dropout_view_mask"].tolist(), [False, False, False, True])
            self.assertEqual(sample["view_mask"].tolist(), [True, False, False, False])
            self.assertGreater(torch.count_nonzero(sample["pre_street"]).item(), 0)
            self.assertEqual(torch.count_nonzero(sample["post_street"]).item(), 0)
            self.assertEqual(torch.count_nonzero(sample["pre_overhead"]).item(), 0)
            self.assertEqual(torch.count_nonzero(sample["post_overhead"]).item(), 0)
            self.assertIs(sample["street"], sample["post_street"])
            self.assertIs(sample["overhead"], sample["post_overhead"])

    def test_pre_only_configuration_does_not_require_post_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pre_street_path = root / "pre.png"
            _write_image(pre_street_path, (25, 50, 75))
            manifest_path = root / "manifest.csv"
            pd.DataFrame(
                [{"sample_id": "pre-only", "pre_street_view_path": pre_street_path, "label": 0}]
            ).to_csv(manifest_path, index=False)

            sample = CrossViewTriageDataset(
                manifest_path,
                street_size=6,
                normalize=False,
                views=("pre_street",),
            )[0]

            self.assertIn("pre_street", sample)
            self.assertIs(sample["street"], sample["pre_street"])
            self.assertNotIn("overhead", sample)
            self.assertEqual(sample["view_mask"].tolist(), [True])

    def test_populated_missing_path_raises_even_when_view_is_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.csv"
            missing_path = root / "does-not-exist.png"
            pd.DataFrame(
                [{"sample_id": "bad-path", "pre_street_view_path": missing_path, "label": 0}]
            ).to_csv(manifest_path, index=False)
            dataset = CrossViewTriageDataset(
                manifest_path,
                views=("pre_street",),
                view_dropout=1.0,
            )

            with self.assertRaisesRegex(FileNotFoundError, "pre_street"):
                dataset[0]

    def test_explicit_unavailable_flag_overrides_placeholder_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.csv"
            pd.DataFrame(
                [
                    {
                        "sample_id": "known-missing",
                        "pre_street_view_path": root / "placeholder-does-not-exist.png",
                        "pre_street_view_available": False,
                        "label": 0,
                    }
                ]
            ).to_csv(manifest_path, index=False)
            sample = CrossViewTriageDataset(
                manifest_path,
                views=("pre_street",),
                normalize=False,
            )[0]
            self.assertEqual(sample["available_view_mask"].tolist(), [False])
            self.assertEqual(sample["missing_view_mask"].tolist(), [True])
            self.assertEqual(torch.count_nonzero(sample["pre_street"]).item(), 0)

    def test_custom_path_column_still_respects_availability_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.csv"
            pd.DataFrame(
                [
                    {
                        "sample_id": "custom-known-missing",
                        "custom_pre": root / "missing.png",
                        "pre_street_available": False,
                        "label": 0,
                    }
                ]
            ).to_csv(manifest_path, index=False)
            sample = CrossViewTriageDataset(
                manifest_path,
                views=("pre_street",),
                view_columns={"pre_street": "custom_pre"},
            )[0]
            self.assertEqual(sample["available_view_mask"].tolist(), [False])


if __name__ == "__main__":
    unittest.main()
