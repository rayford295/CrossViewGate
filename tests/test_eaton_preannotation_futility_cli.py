from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from crossview_conflict.analysis.eaton_component_direction import (
    PROBABILITY_COLUMNS,
    SEVERITY_CLASS_ORDER,
)
from scripts import run_eaton_preannotation_futility_gate as cli


REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONFIG = REPO_ROOT / "configs" / "eaton_component_direction_v1.json"
SEEDS = (42, 123, 456, 789, 1011)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _canonical_digest(value: dict[str, object], digest_field: str) -> str:
    unsigned = {key: item for key, item in value.items() if key != digest_field}
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha(encoded)


class SyntheticBundle:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.protocol = root / "protocol.json"
        self.commitments = root / "commitments.json"
        self.prediction_commitments = root / "prediction_commitments.json"
        self.summary = root / "protocol_summary.json"
        self.manifest = root / "study_development.csv"
        self.predictions = root / "study_development_predictions.csv"
        self.prediction_metadata = root / "prediction_metadata.json"
        self.output = root / "result.json"

        protocol_raw = FROZEN_CONFIG.read_bytes()
        protocol = json.loads(protocol_raw.decode("utf-8"))
        protocol_version = protocol["protocol_version"]
        protocol_hash = _sha(protocol_raw)

        manifest = pd.DataFrame(
            [
                {
                    "pair_id": f"pair-{index}",
                    "spatial_block_id": f"block-{index // 2}",
                    "protocol_role": "study_development",
                    "label": index % 3,
                }
                for index in range(6)
            ]
        )
        manifest_raw = _csv_bytes(manifest)

        rows: list[dict[str, object]] = []
        for seed in SEEDS:
            for index, item in manifest.iterrows():
                row: dict[str, object] = {
                    "pair_id": item["pair_id"],
                    "seed": seed,
                    "spatial_block_id": item["spatial_block_id"],
                    "protocol_role": "study_development",
                    "target": int(item["label"]),
                    "class_order": json.dumps(list(SEVERITY_CLASS_ORDER)),
                }
                for column in PROBABILITY_COLUMNS:
                    row[column] = 0.05
                row["street_prob_no_or_trace_damage"] = 0.90
                if index % 2 == 0:
                    row["overhead_prob_damaged_repairable"] = 0.90
                else:
                    row["overhead_prob_no_or_trace_damage"] = 0.90
                rows.append(row)
        predictions = pd.DataFrame(rows)
        prediction_raw = _csv_bytes(predictions)

        metadata = {
            "schema_version": "eaton-component-direction-predictions-v1",
            "protocol_version": protocol_version,
            "status": "PROTOCOL_DEVELOPMENT_RUN",
            "protocol_config_sha256": protocol_hash,
            "role_manifest_sha256": {
                "study_development": _sha(manifest_raw),
            },
            "prediction_role": "study_development",
            "spatial_confirmation_read_or_scored": False,
            "class_order": list(SEVERITY_CLASS_ORDER),
            "seeds": list(SEEDS),
            "prediction_csv_sha256": _sha(prediction_raw),
        }
        metadata_raw = _json_bytes(metadata)
        summary = {
            "schema_version": "eaton-component-direction-spatial-summary-v1",
            "protocol_version": protocol_version,
            "protocol_config_sha256": protocol_hash,
            "analysis_status": "PROTOCOL_READY_CONFIRMATION_UNTOUCHED",
            "role_manifest_sha256": {
                "study_development": _sha(manifest_raw),
            },
            "role_rows": {"study_development": len(manifest)},
            "confirmation_commitment": {
                "status": "UNSCORED_RESERVED_SAME_EVENT_SPATIAL_CONFIRMATION",
                "manifest_sha256": "f" * 64,
            },
        }
        summary_raw = _json_bytes(summary)
        commitments = {
            "schema_version": cli.COMMITMENT_SCHEMA,
            "protocol_version": protocol_version,
            "event_id": "synthetic-eaton",
            "status": "FROZEN_BEFORE_REGISTERED_DIRECTIONAL_ANALYSIS",
            "protocol_config_sha256": protocol_hash,
            "protocol_summary_sha256": _sha(summary_raw),
            "role_manifest_sha256": {
                "study_development": _sha(manifest_raw),
            },
            "confirmation_status": (
                "UNSCORED_RESERVED_SAME_EVENT_SPATIAL_CONFIRMATION"
            ),
        }
        prediction_commitments = {
            "schema_version": cli.PREDICTION_COMMITMENT_SCHEMA,
            "protocol_version": protocol_version,
            "status": "FROZEN_BEFORE_PRE_ANNOTATION_FUTILITY_AUDIT",
            "protocol_config_sha256": protocol_hash,
            "protocol_summary_sha256": _sha(summary_raw),
            "prediction_csv_sha256": _sha(prediction_raw),
            "prediction_metadata_sha256": _sha(metadata_raw),
        }

        self.protocol.write_bytes(protocol_raw)
        self.manifest.write_bytes(manifest_raw)
        self.predictions.write_bytes(prediction_raw)
        self.prediction_metadata.write_bytes(metadata_raw)
        self.summary.write_bytes(summary_raw)
        self.commitments.write_bytes(_json_bytes(commitments))
        self.prediction_commitments.write_bytes(
            _json_bytes(prediction_commitments)
        )
        self.expected_summary = _sha(summary_raw)
        self.expected_predictions = _sha(prediction_raw)
        self.expected_prediction_metadata = _sha(metadata_raw)

    def kwargs(self, *, output: Path | None = None) -> dict[str, object]:
        return {
            "protocol_config_path": self.protocol,
            "artifact_commitments_path": self.commitments,
            "prediction_commitments_path": self.prediction_commitments,
            "protocol_summary_path": self.summary,
            "development_manifest_path": self.manifest,
            "predictions_path": self.predictions,
            "prediction_metadata_path": self.prediction_metadata,
            "expected_protocol_summary_sha256": self.expected_summary,
            "expected_prediction_csv_sha256": self.expected_predictions,
            "expected_prediction_metadata_sha256": (
                self.expected_prediction_metadata
            ),
            "output_path": self.output if output is None else output,
        }


class PreannotationFutilityCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bundle = SyntheticBundle(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_synthetic_structural_stop_preserves_core_artifact_digest(self) -> None:
        wrapper = cli.run_preannotation_futility_gate(**self.bundle.kwargs())
        artifact = wrapper["futility_artifact"]
        self.assertTrue(artifact["structurally_futile"])
        self.assertEqual(
            artifact["phase_status"],
            "STOP_PRE_ANNOTATION_STRUCTURAL_FUTILITY",
        )
        self.assertFalse(artifact["observed_outcome_accessed"])
        self.assertFalse(artifact["signed_direction_constructed"])
        self.assertIsNone(artifact["h_b1_result"])
        self.assertIsNone(artifact["h_b2_result"])
        self.assertEqual(
            artifact["futility_artifact_sha256"],
            _canonical_digest(artifact, "futility_artifact_sha256"),
        )
        self.assertEqual(
            wrapper["core_futility_artifact_sha256"],
            artifact["futility_artifact_sha256"],
        )
        self.assertEqual(
            json.loads(self.bundle.output.read_text(encoding="utf-8")), wrapper
        )

    def test_all_input_paths_and_raw_hashes_are_recorded(self) -> None:
        wrapper = cli.run_preannotation_futility_gate(**self.bundle.kwargs())
        expected = {
            "artifact_commitments": self.bundle.commitments,
            "prediction_commitments": self.bundle.prediction_commitments,
            "protocol_config": self.bundle.protocol,
            "protocol_summary": self.bundle.summary,
            "study_development_manifest": self.bundle.manifest,
            "prediction_csv": self.bundle.predictions,
            "prediction_metadata": self.bundle.prediction_metadata,
        }
        for name, path in expected.items():
            metadata = wrapper["input_artifacts"][name]
            self.assertTrue(Path(metadata["absolute_path"]).is_absolute())
            self.assertEqual(Path(metadata["absolute_path"]), path.resolve())
            self.assertEqual(metadata["sha256"], _sha(path.read_bytes()))
            self.assertEqual(metadata["byte_count"], len(path.read_bytes()))
        self.assertEqual(
            wrapper["trusted_anchors"]["prediction_commitments_raw_sha256"],
            _sha(self.bundle.prediction_commitments.read_bytes()),
        )

    def test_consistently_resigned_truncation_fails_trusted_summary_anchor(self) -> None:
        original_summary_hash = self.bundle.expected_summary
        predictions = pd.read_csv(self.bundle.predictions)
        kept_ids = set(predictions["pair_id"].unique()[:2])
        predictions = predictions.loc[predictions["pair_id"].isin(kept_ids)]
        prediction_raw = _csv_bytes(predictions)
        self.bundle.predictions.write_bytes(prediction_raw)

        manifest = pd.read_csv(self.bundle.manifest)
        manifest = manifest.loc[manifest["pair_id"].isin(kept_ids)]
        manifest_raw = _csv_bytes(manifest)
        self.bundle.manifest.write_bytes(manifest_raw)

        metadata = json.loads(self.bundle.prediction_metadata.read_text("utf-8"))
        metadata["prediction_csv_sha256"] = _sha(prediction_raw)
        metadata["role_manifest_sha256"]["study_development"] = _sha(manifest_raw)
        metadata_raw = _json_bytes(metadata)
        self.bundle.prediction_metadata.write_bytes(metadata_raw)

        summary = json.loads(self.bundle.summary.read_text("utf-8"))
        summary["role_manifest_sha256"]["study_development"] = _sha(manifest_raw)
        summary["role_rows"]["study_development"] = len(manifest)
        summary_raw = _json_bytes(summary)
        self.bundle.summary.write_bytes(summary_raw)

        # Also re-sign the untrusted local commitments.  The externally supplied
        # original summary anchor still makes this coordinated truncation fail.
        commitments = json.loads(self.bundle.commitments.read_text("utf-8"))
        commitments["protocol_summary_sha256"] = _sha(summary_raw)
        commitments["role_manifest_sha256"]["study_development"] = _sha(
            manifest_raw
        )
        self.bundle.commitments.write_bytes(_json_bytes(commitments))

        prediction_commitments = json.loads(
            self.bundle.prediction_commitments.read_text("utf-8")
        )
        prediction_commitments["protocol_summary_sha256"] = _sha(summary_raw)
        prediction_commitments["prediction_csv_sha256"] = _sha(prediction_raw)
        prediction_commitments["prediction_metadata_sha256"] = _sha(metadata_raw)
        self.bundle.prediction_commitments.write_bytes(
            _json_bytes(prediction_commitments)
        )

        kwargs = self.bundle.kwargs()
        kwargs["expected_protocol_summary_sha256"] = original_summary_hash
        kwargs["expected_prediction_csv_sha256"] = _sha(prediction_raw)
        kwargs["expected_prediction_metadata_sha256"] = _sha(metadata_raw)
        with self.assertRaisesRegex(ValueError, "expected argument disagrees"):
            cli.run_preannotation_futility_gate(**kwargs)
        self.assertFalse(self.bundle.output.exists())

    def test_prediction_values_and_metadata_resigned_fail_post_run_anchors(self) -> None:
        predictions = pd.read_csv(self.bundle.predictions)
        first = predictions.index[0]
        predictions.loc[first, "street_prob_no_or_trace_damage"] = 0.05
        predictions.loc[first, "street_prob_damaged_repairable"] = 0.90
        prediction_raw = _csv_bytes(predictions)
        self.bundle.predictions.write_bytes(prediction_raw)

        metadata = json.loads(self.bundle.prediction_metadata.read_text("utf-8"))
        metadata["prediction_csv_sha256"] = _sha(prediction_raw)
        self.bundle.prediction_metadata.write_bytes(_json_bytes(metadata))

        # The mutually consistent changed files still differ from the trusted
        # post-run commitments passed to the CLI.
        with self.assertRaisesRegex(ValueError, "frozen prediction commitments"):
            cli.run_preannotation_futility_gate(**self.bundle.kwargs())
        self.assertFalse(self.bundle.output.exists())

    def test_prediction_commitment_status_tamper_is_rejected(self) -> None:
        commitment = json.loads(
            self.bundle.prediction_commitments.read_text(encoding="utf-8")
        )
        commitment["status"] = "AFTER_AUDIT"
        self.bundle.prediction_commitments.write_bytes(_json_bytes(commitment))
        with self.assertRaisesRegex(ValueError, "not frozen before"):
            cli.run_preannotation_futility_gate(**self.bundle.kwargs())
        self.assertFalse(self.bundle.output.exists())

    def test_expected_hash_must_agree_with_prediction_commitment(self) -> None:
        kwargs = self.bundle.kwargs()
        kwargs["expected_prediction_csv_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "expected argument disagrees"):
            cli.run_preannotation_futility_gate(**kwargs)
        self.assertFalse(self.bundle.output.exists())

    def test_missing_default_prediction_commitment_fails_closed(self) -> None:
        kwargs = self.bundle.kwargs()
        kwargs.pop("prediction_commitments_path")
        missing = self.root / "missing-default-prediction-commitment.json"
        with patch.object(cli, "DEFAULT_PREDICTION_COMMITMENTS_PATH", missing):
            with self.assertRaises(FileNotFoundError):
                cli.run_preannotation_futility_gate(**kwargs)
        self.assertFalse(self.bundle.output.exists())

    def test_input_symlink_is_rejected(self) -> None:
        link = self.root / "prediction-link.csv"
        try:
            link.symlink_to(self.bundle.predictions)
        except OSError as error:
            # Windows may deny creating symlinks without Developer Mode.  Keep
            # the fail-closed branch covered on those hosts as well.
            link.write_bytes(self.bundle.predictions.read_bytes())
            path_type = type(link)
            original = path_type.is_symlink
            kwargs = self.bundle.kwargs()
            kwargs["predictions_path"] = link
            with patch.object(
                path_type,
                "is_symlink",
                autospec=True,
                side_effect=lambda candidate: candidate == link
                or original(candidate),
            ):
                with self.assertRaisesRegex(ValueError, "symlink"):
                    cli.run_preannotation_futility_gate(**kwargs)
            self.assertIsInstance(error, OSError)
            return
        kwargs = self.bundle.kwargs()
        kwargs["predictions_path"] = link
        with self.assertRaisesRegex(ValueError, "symlink"):
            cli.run_preannotation_futility_gate(**kwargs)

    def test_exclusive_output_and_overwrite_scope(self) -> None:
        cli.run_preannotation_futility_gate(**self.bundle.kwargs())
        with self.assertRaises(FileExistsError):
            cli.run_preannotation_futility_gate(**self.bundle.kwargs())

        kwargs = self.bundle.kwargs()
        kwargs["overwrite"] = True
        with self.assertRaisesRegex(ValueError, "known futility output"):
            cli.run_preannotation_futility_gate(**kwargs)

        with patch.object(cli, "DEFAULT_OUTPUT_PATH", self.bundle.output):
            rewritten = cli.run_preannotation_futility_gate(**kwargs)
        self.assertEqual(rewritten["run_status"], "COMPLETE")

    def test_output_symlink_is_rejected(self) -> None:
        real = self.root / "real.json"
        real.write_text("do not replace", encoding="utf-8")
        link = self.root / "result-link.json"
        try:
            link.symlink_to(real)
        except OSError as error:
            link.write_text("synthetic link placeholder", encoding="utf-8")
            path_type = type(link)
            original = path_type.is_symlink
            kwargs = self.bundle.kwargs(output=link)
            with patch.object(
                path_type,
                "is_symlink",
                autospec=True,
                side_effect=lambda candidate: candidate == link
                or original(candidate),
            ):
                with self.assertRaisesRegex(ValueError, "symlink"):
                    cli.run_preannotation_futility_gate(**kwargs)
            self.assertIsInstance(error, OSError)
            self.assertEqual(real.read_text(encoding="utf-8"), "do not replace")
            return
        kwargs = self.bundle.kwargs(output=link)
        with self.assertRaisesRegex(ValueError, "symlink"):
            cli.run_preannotation_futility_gate(**kwargs)
        self.assertEqual(real.read_text(encoding="utf-8"), "do not replace")


if __name__ == "__main__":
    unittest.main()
