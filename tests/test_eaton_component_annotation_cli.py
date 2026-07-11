from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from crossview_conflict.analysis.eaton_component_direction import ANNOTATION_FIELDS
from scripts.build_eaton_component_annotation_packet import sha256_file
from scripts.validate_eaton_component_annotations import (
    ADJUDICATION_TEMPLATE_COLUMNS,
    validate_completed_annotations,
)
from test_eaton_component_annotation_packet import PacketFixture


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "eaton_component_direction_v1.json"


class AnnotationValidationCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = PacketFixture(self.root)
        self.packet = self.root / "packet"
        self.fixture.write(self.packet)
        self.rater_a, self.rater_b = self.fixture.complete_templates(self.packet)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_validation(
        self,
        output: Path,
        *,
        completed_adjudication: Path | None = None,
        rater_a: Path | None = None,
        rater_b: Path | None = None,
        overwrite: bool = False,
    ) -> dict[str, object]:
        return validate_completed_annotations(
            public_packet_dir=self.packet / "public",
            restricted_packet_dir=self.packet / "restricted",
            rater_a_path=rater_a or self.rater_a,
            rater_b_path=rater_b or self.rater_b,
            protocol_path=PROTOCOL_PATH,
            completed_adjudication_path=completed_adjudication,
            output_dir=output,
            overwrite=overwrite,
        )

    def make_one_dispute(self) -> str:
        frame = pd.read_csv(self.rater_b, dtype=str, keep_default_na=False)
        opaque_id = str(frame.loc[0, "pair_id"])
        frame.loc[0, "component_dominance"] = "facade"
        frame.to_csv(self.rater_b, index=False, lineterminator="\n")
        return opaque_id

    def completed_adjudication(
        self, template_path: Path, *, destination: Path | None = None
    ) -> Path:
        frame = pd.read_csv(template_path, dtype=str, keep_default_na=False)
        frame["adjudicator_id"] = "qualified-third-rater"
        frame["component_dominance"] = "roof"
        frame["street_roof_assessability"] = "not_assessable"
        frame["street_facade_assessability"] = "indeterminate"
        frame["overhead_roof_assessability"] = "assessable"
        frame["overhead_facade_assessability"] = "indeterminate"
        frame["adjudication_note"] = "Resolved from the blinded paired images."
        path = destination or self.root / "completed_adjudication.csv"
        frame.to_csv(path, index=False, lineterminator="\n")
        return path

    def test_clean_agreement_builds_restricted_reference_and_hash_inventory(self) -> None:
        output = self.root / "validation-clean"
        metadata = self.run_validation(output)

        self.assertEqual(metadata["status"], "REFERENCE_READY_NO_ADJUDICATION")
        self.assertTrue(metadata["reference_ready"])
        self.assertEqual(metadata["disputed_item_count"], 0)
        self.assertEqual(list((output / "public").iterdir()), [])
        restricted = output / "restricted"
        for name in (
            "remapped_raw_annotations.csv",
            "annotation_reliability.csv",
            "component_reference.csv",
            "annotation_validation_metadata.json",
            "annotation_validation_metadata.sha256",
        ):
            self.assertTrue((restricted / name).is_file())
        reference = pd.read_csv(restricted / "component_reference.csv", dtype=str)
        self.assertEqual(len(reference), 2)
        self.assertNotIn("third_rater_adjudication_template.csv", metadata["output_sha256"])

        for item in metadata["input_sha256"]:
            self.assertEqual(sha256_file(Path(item["path"])), item["sha256"])
        for relative, digest in metadata["output_sha256"].items():
            self.assertEqual(sha256_file(output / relative), digest)
        metadata_path = restricted / "annotation_validation_metadata.json"
        expected_metadata_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        self.assertTrue(
            (restricted / "annotation_validation_metadata.sha256")
            .read_text(encoding="ascii")
            .startswith(expected_metadata_hash)
        )

        with self.assertRaises(FileExistsError):
            self.run_validation(output)
        overwritten = self.run_validation(output, overwrite=True)
        self.assertEqual(overwritten["status"], "REFERENCE_READY_NO_ADJUDICATION")

    def test_dispute_emits_only_blinded_opaque_public_template(self) -> None:
        disputed_opaque_id = self.make_one_dispute()
        output = self.root / "validation-dispute"
        metadata = self.run_validation(output)

        self.assertEqual(metadata["status"], "AWAITING_BLINDED_THIRD_RATER")
        self.assertFalse(metadata["reference_ready"])
        self.assertEqual(metadata["disputed_item_count"], 1)
        public_files = list((output / "public").iterdir())
        self.assertEqual(
            [path.name for path in public_files],
            ["third_rater_adjudication_template.csv"],
        )
        template = pd.read_csv(public_files[0], dtype=str, keep_default_na=False)
        self.assertEqual(tuple(template.columns), ADJUDICATION_TEMPLATE_COLUMNS)
        self.assertEqual(template["opaque_item_id"].tolist(), [disputed_opaque_id])
        self.assertTrue(all(template[field].eq("").all() for field in ANNOTATION_FIELDS))
        self.assertRegex(template.loc[0, "street_media_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(template.loc[0, "overhead_media_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(template.loc[0, "protocol_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(template.loc[0, "codebook_sha256"], r"^[0-9a-f]{64}$")

        private = pd.read_csv(
            self.packet / "restricted" / "private_key.csv",
            dtype=str,
            keep_default_na=False,
        )
        public_bytes = public_files[0].read_bytes()
        for source_pair_id in private["pair_id"]:
            self.assertNotIn(source_pair_id.encode("utf-8"), public_bytes)
        secret = (self.packet / "restricted" / "blinding_secret.key").read_bytes()
        self.assertNotIn(secret, public_bytes)
        self.assertNotIn(secret.hex().encode("ascii"), public_bytes)
        forbidden_markers = (
            "prediction",
            "probability",
            "confidence",
            "direction",
            "source_label",
            "spatial_block",
            "protocol_role",
            "target",
            "seed",
        )
        self.assertFalse(
            any(
                marker in column.casefold()
                for column in template.columns
                for marker in forbidden_markers
            )
        )
        self.assertFalse((output / "public" / "component_reference.csv").exists())
        self.assertTrue(
            (output / "restricted" / "remapped_raw_annotations.csv").is_file()
        )
        self.assertTrue((output / "restricted" / "annotation_reliability.csv").is_file())

    def test_completed_adjudication_builds_provenance_reference_restricted_only(self) -> None:
        self.make_one_dispute()
        handoff = self.root / "handoff"
        self.run_validation(handoff)
        completed = self.completed_adjudication(
            handoff / "public" / "third_rater_adjudication_template.csv"
        )
        output = self.root / "validation-adjudicated"
        metadata = self.run_validation(output, completed_adjudication=completed)

        self.assertEqual(metadata["status"], "REFERENCE_READY_AFTER_ADJUDICATION")
        self.assertTrue(metadata["reference_ready"])
        self.assertEqual(list((output / "public").iterdir()), [])
        restricted = output / "restricted"
        adjudications = pd.read_csv(
            restricted / "adjudications_true_id.csv", dtype=str, keep_default_na=False
        )
        reference = pd.read_csv(
            restricted / "component_reference.csv", dtype=str, keep_default_na=False
        )
        self.assertTrue(set(adjudications["pair_id"]).issubset(set(reference["pair_id"])))
        resolved = reference.loc[reference["adjudicated"].eq("True")]
        self.assertEqual(len(resolved), 1)
        provenance = json.loads(resolved["annotation_provenance"].iat[0])
        self.assertEqual(
            provenance["adjudication"]["adjudicator_id"], "qualified-third-rater"
        )

    def test_wrong_adjudicator_item_hash_or_protocol_fails_before_output(self) -> None:
        self.make_one_dispute()
        handoff = self.root / "handoff-invalid"
        self.run_validation(handoff)
        base_path = self.completed_adjudication(
            handoff / "public" / "third_rater_adjudication_template.csv"
        )
        base = pd.read_csv(base_path, dtype=str, keep_default_na=False)
        cases = {
            "adjudicator": ("adjudicator_id", "human-rater-a", "distinct"),
            "item": ("opaque_item_id", "item_not_in_packet", "exactly equal"),
            "hash": ("street_media_sha256", "f" * 64, "identity mismatch"),
            "protocol": ("protocol_version", "wrong-protocol", "identity mismatch"),
        }
        for name, (column, value, message) in cases.items():
            with self.subTest(name=name):
                frame = base.copy()
                frame.loc[0, column] = value
                path = self.root / f"bad-{name}.csv"
                frame.to_csv(path, index=False, lineterminator="\n")
                output = self.root / f"failed-{name}"
                with self.assertRaisesRegex(ValueError, message):
                    self.run_validation(output, completed_adjudication=path)
                self.assertFalse(output.exists())

    def test_output_cannot_escape_into_packet_tree_directly_or_through_symlink(self) -> None:
        direct = self.packet / "public" / "validation_inside_public"
        with self.assertRaisesRegex(ValueError, "cannot overlap"):
            self.run_validation(direct)
        self.assertFalse(direct.exists())

        ancestor = self.packet
        with self.assertRaisesRegex(ValueError, "cannot overlap"):
            self.run_validation(ancestor, overwrite=True)

        link = self.root / "public-packet-link"
        try:
            link.symlink_to(self.packet / "public", target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Directory symlinks are unavailable: {error}")
        escaped = link / "validation_via_symlink"
        with self.assertRaisesRegex(ValueError, "cannot overlap"):
            self.run_validation(escaped)
        self.assertFalse(escaped.exists())

    def test_completed_answers_cannot_be_read_from_any_public_handoff_tree(self) -> None:
        for letter, completed in (("a", self.rater_a), ("b", self.rater_b)):
            with self.subTest(rater=letter):
                public_template = (
                    self.packet / "public" / f"rater_{letter}_template.csv"
                )
                public_template.write_bytes(completed.read_bytes())
                kwargs = {f"rater_{letter}": public_template}
                output = self.root / f"public-rater-{letter}-must-fail"
                with self.assertRaisesRegex(ValueError, "restricted working"):
                    self.run_validation(output, **kwargs)
                self.assertFalse(output.exists())

        self.make_one_dispute()
        handoff = self.root / "adjudication-handoff"
        self.run_validation(handoff)
        in_place = handoff / "public" / "third_rater_adjudication_template.csv"
        self.completed_adjudication(in_place, destination=in_place)
        output = self.root / "public-adjudication-must-fail"
        with self.assertRaisesRegex(ValueError, "restricted working"):
            self.run_validation(output, completed_adjudication=in_place)
        self.assertFalse(output.exists())

        link = self.root / "completed-adjudication-link.csv"
        try:
            link.symlink_to(in_place)
        except OSError:
            return
        linked_output = self.root / "linked-public-adjudication-must-fail"
        with self.assertRaisesRegex(ValueError, "restricted working"):
            self.run_validation(linked_output, completed_adjudication=link)
        self.assertFalse(linked_output.exists())

        public_link = self.packet / "public" / "linked-completed-rater-a.csv"
        public_link.symlink_to(self.rater_a)
        linked_rater_output = self.root / "public-link-to-private-must-fail"
        with self.assertRaisesRegex(ValueError, "restricted working"):
            self.run_validation(linked_rater_output, rater_a=public_link)
        self.assertFalse(linked_rater_output.exists())


if __name__ == "__main__":
    unittest.main()
