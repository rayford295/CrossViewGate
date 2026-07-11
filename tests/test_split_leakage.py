from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from scripts.check_split_leakage import (
    audit_split_frames,
    audit_split_paths,
    check_leakage,
    main,
)


class SplitLeakageAuditTests(unittest.TestCase):
    def test_group_audit_reports_overlapping_groups_and_affected_samples(self) -> None:
        train = pd.DataFrame(
            {
                "objectid": [1, 1, 2],
                "sequence_id": ["s1", "s1", "s2"],
                "spatial_block_id": ["north", "north", "south"],
            }
        )
        val = pd.DataFrame(
            {
                # The string form verifies stable group normalisation across
                # differing CSV type inference.
                "objectid": ["1", "3"],
                "sequence_id": ["s3", "s4"],
                "spatial_block_id": ["east", "east"],
            }
        )
        test = pd.DataFrame(
            {
                "objectid": [2, 4],
                "sequence_id": ["s5", "s6"],
                "spatial_block_id": ["east", "west"],
            }
        )

        result = audit_split_frames(
            train,
            val,
            test,
            "groups",
            group_columns=["objectid", "sequence_id", "spatial_block_id"],
        )

        self.assertTrue(result["leakage_found"])
        self.assertTrue(bool(result))
        object_train_val = result["group_audits"]["objectid"]["pairs"]["train_val"]
        self.assertEqual(object_train_val["overlap_group_count"], 1)
        self.assertEqual(object_train_val["overlap_sample_count"], 3)
        self.assertEqual(
            object_train_val["overlap_sample_count_by_split"], {"train": 2, "val": 1}
        )
        object_train_test = result["group_audits"]["objectid"]["pairs"]["train_test"]
        self.assertEqual(object_train_test["overlap_group_count"], 1)
        self.assertEqual(object_train_test["overlap_sample_count"], 2)
        self.assertFalse(result["group_audits"]["sequence_id"]["leakage_found"])
        block_val_test = result["group_audits"]["spatial_block_id"]["pairs"]["val_test"]
        self.assertEqual(block_val_test["overlap_group_count"], 1)
        self.assertEqual(block_val_test["overlap_sample_count"], 3)

    def test_missing_requested_group_column_is_structured_incomplete_result(self) -> None:
        frame = pd.DataFrame({"objectid": [1]})
        result = audit_split_frames(
            frame,
            pd.DataFrame({"objectid": [2]}),
            pd.DataFrame({"objectid": [3]}),
            group_columns=["objectid", "sequence_id"],
        )

        self.assertFalse(result["leakage_found"])
        self.assertFalse(result["audit_complete"])
        sequence = result["group_audits"]["sequence_id"]
        self.assertFalse(sequence["available"])
        self.assertEqual(sequence["missing_splits"], ["train", "val", "test"])

    def test_empty_group_values_do_not_produce_false_clean_audit(self) -> None:
        train = pd.DataFrame({"objectid": [None, None]})
        val = pd.DataFrame({"objectid": ["v"]})
        test = pd.DataFrame({"objectid": ["t"]})

        result = audit_split_frames(train, val, test)

        audit = result["group_audits"]["objectid"]
        self.assertFalse(result["leakage_found"])
        self.assertFalse(result["audit_complete"])
        self.assertFalse(audit["available"])
        self.assertEqual(audit["empty_splits"], ["train"])
        self.assertFalse(audit["pairs"]["train_val"]["available"])

    def test_spatial_threshold_flags_close_cross_split_samples(self) -> None:
        train = pd.DataFrame({"objectid": ["a"], "latitude": [0.0], "longitude": [0.0]})
        val = pd.DataFrame({"objectid": ["b"], "latitude": [0.0], "longitude": [0.01]})
        test = pd.DataFrame(
            {"objectid": ["c"], "latitude": [0.0], "longitude": [0.0005]}
        )

        result = audit_split_frames(
            train,
            val,
            test,
            spatial_threshold_m=100.0,
        )

        self.assertTrue(result["spatial_audit"]["leakage_found"])
        self.assertTrue(result["leakage_found"])
        pair = result["spatial_audit"]["pairs"]["train_test"]
        self.assertAlmostEqual(pair["minimum_distance_m"], 55.6, delta=0.2)
        self.assertEqual(pair["within_threshold_pair_count"], 1)
        self.assertEqual(pair["within_threshold_sample_count"], 2)
        self.assertEqual(
            pair["within_threshold_sample_count_by_split"], {"train": 1, "test": 1}
        )
        self.assertFalse(result["spatial_audit"]["pairs"]["train_val"]["leakage_found"])
        self.assertFalse(result["spatial_audit"]["pairs"]["val_test"]["leakage_found"])

    def test_nearest_neighbor_diagnostics_without_threshold_do_not_assert_leakage(self) -> None:
        train = pd.DataFrame({"objectid": [1], "latitude": [10.0], "longitude": [20.0]})
        val = pd.DataFrame({"objectid": [2], "latitude": [10.1], "longitude": [20.0]})
        test = pd.DataFrame({"objectid": [3], "latitude": [10.2], "longitude": [20.0]})
        result = audit_split_frames(train, val, test, audit_coordinates=True)

        self.assertFalse(result["leakage_found"])
        self.assertTrue(result["spatial_audit"]["available"])
        self.assertIsNotNone(
            result["spatial_audit"]["pairs"]["train_val"]["minimum_distance_m"]
        )
        self.assertIsNone(result["spatial_audit"]["threshold_m"])


class SplitLeakageCliTests(unittest.TestCase):
    def _write_split(self, root: Path, name: str, objectid: str) -> Path:
        path = root / f"{name}.csv"
        pd.DataFrame({"objectid": [objectid]}).to_csv(path, index=False)
        return path

    def test_repeated_dataset_cli_returns_one_when_any_dataset_leaks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean_train = self._write_split(root, "clean_train", "a")
            clean_val = self._write_split(root, "clean_val", "b")
            clean_test = self._write_split(root, "clean_test", "c")
            leak_train = self._write_split(root, "leak_train", "shared")
            leak_val = self._write_split(root, "leak_val", "v")
            leak_test = self._write_split(root, "leak_test", "shared")
            json_path = root / "audit.json"

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--dataset",
                        "clean",
                        str(clean_train),
                        str(clean_val),
                        str(clean_test),
                        "--dataset",
                        "leaky",
                        str(leak_train),
                        str(leak_val),
                        str(leak_test),
                        "--json-output",
                        str(json_path),
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertIn("OVERALL: leakage found", stdout.getvalue())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["leakage_found"])
            self.assertEqual(len(payload["datasets"]), 2)
            self.assertEqual(payload["datasets"][0]["label"], "clean")
            self.assertFalse(payload["datasets"][0]["leakage_found"])
            self.assertIn("group_audits", payload["datasets"][0])

    def test_incomplete_requested_audit_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for split in ("train", "val", "test"):
                path = root / f"{split}.csv"
                pd.DataFrame({"sample_id": [split]}).to_csv(path, index=False)
                paths.append(path)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--dataset",
                        "missing-group",
                        *(str(path) for path in paths),
                        "--group-cols",
                        "objectid",
                    ]
                )
            self.assertEqual(exit_code, 2)
            self.assertIn("audit incomplete", stdout.getvalue())

    def test_single_legacy_hurricane_argument_set_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = self._write_split(root, "train", "a")
            val = self._write_split(root, "val", "b")
            test = self._write_split(root, "test", "c")

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "--hurricane-train",
                        str(train),
                        "--hurricane-val",
                        str(val),
                        "--hurricane-test",
                        str(test),
                    ]
                )
            self.assertEqual(exit_code, 0)

    def test_path_api_is_structured_and_old_api_remains_boolean(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = self._write_split(root, "train", "shared")
            val = self._write_split(root, "val", "v")
            test = self._write_split(root, "test", "shared")

            result = audit_split_paths(train, val, test, print_report=False)
            legacy_result = check_leakage(train, val, test, print_report=False)

            self.assertIsInstance(result, dict)
            self.assertTrue(result["leakage_found"])
            self.assertTrue(bool(result))
            self.assertIs(legacy_result, True)


if __name__ == "__main__":
    unittest.main()
