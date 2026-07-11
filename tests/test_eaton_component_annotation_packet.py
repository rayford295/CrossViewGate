from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from PIL import Image, PngImagePlugin
from PIL.TiffImagePlugin import IFDRational

from crossview_conflict.analysis.eaton_component_direction import (
    ANNOTATION_FIELDS,
    PROBABILITY_COLUMNS,
    SEVERITY_CLASS_ORDER,
)
from scripts.build_eaton_component_annotation_packet import (
    FROZEN_SELECTION_RULE,
    PRIVATE_KEY_COLUMNS,
    PUBLIC_PACKET_COLUMNS,
    RATER_A_PLACEHOLDER,
    RATER_B_PLACEHOLDER,
    RATER_TEMPLATE_COLUMNS,
    build_packet_tables,
    metadata_stripped_derivative,
    opaque_item_id,
    packet_order_sha256,
    remap_completed_rater_templates,
    sha256_file,
    write_annotation_packet,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "eaton_component_direction_v1.json"


def _rgb_sha256(data: bytes) -> tuple[str, tuple[int, int]]:
    with Image.open(BytesIO(data)) as image:
        image.load()
        return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest(), image.size


def _jpeg_with_private_exif(path: Path, pair_id: str, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (48, 40))
    image.putdata(
        [
            (
                (color[0] + 17 * x + 31 * y) % 256,
                (color[1] + 43 * x + 11 * y) % 256,
                (color[2] + 7 * x + 53 * y) % 256,
            )
            for y in range(40)
            for x in range(48)
        ]
    )
    exif = Image.Exif()
    exif[270] = f"private source id {pair_id}"
    exif[271] = "PrivateCamera"
    exif[274] = 1
    exif[306] = "2026:01:02 03:04:05"
    exif[34853] = {
        1: "N",
        2: (
            IFDRational(34, 1),
            IFDRational(10, 1),
            IFDRational(0, 1),
        ),
        3: "W",
        4: (
            IFDRational(118, 1),
            IFDRational(2, 1),
            IFDRational(0, 1),
        ),
    }
    image.save(
        path,
        format="JPEG",
        quality=91,
        exif=exif,
        icc_profile=b"private-device-profile-for-blinding-test",
    )


def _png_with_private_text(path: Path, pair_id: str, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (16, 12), color)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Comment", f"private source id {pair_id}")
    metadata.add_text("Creation Time", "2026-01-02T03:04:05Z")
    metadata.add_itxt(
        "XML:com.adobe.xmp",
        "<x:xmpmeta><private>location</private></x:xmpmeta>",
    )
    image.save(path, format="PNG", pnginfo=metadata)


class PacketFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.blinding_secret = bytes.fromhex("11" * 32)
        self.seeds = tuple(self.protocol["models"]["seeds"])
        self.pair_specs = {
            "source-pair-a": (0, "block-a", 0, 1),
            "source-pair-b": (1, "block-b", 2, 0),
            "source-pair-c": (2, "block-c", 1, 1),
        }
        media_dir = root / "source_media"
        media_dir.mkdir()
        manifest_rows = []
        ledger_rows = []
        prediction_rows = []
        for index, (pair_id, (target, block, street_class, overhead_class)) in enumerate(
            self.pair_specs.items()
        ):
            street_path = media_dir / f"{pair_id}_street.jpg"
            overhead_path = media_dir / f"{pair_id}_overhead.png"
            _jpeg_with_private_exif(
                street_path, pair_id, (10 + index, 30 + index, 60 + index)
            )
            _png_with_private_text(
                overhead_path, pair_id, (80 + index, 40 + index, 20 + index)
            )
            street_hash = sha256_file(street_path)
            overhead_hash = sha256_file(overhead_path)
            common = {
                "pair_id": pair_id,
                "street_view_path": str(street_path),
                "remote_sensing_path": str(overhead_path),
                "street_sha256": street_hash,
                "remote_sha256": overhead_hash,
            }
            manifest_rows.append(
                {
                    **common,
                    "spatial_block_id": block,
                    "protocol_role": "study_development",
                    "label": target,
                }
            )
            ledger_rows.append(common)
            for seed in self.seeds:
                row = {
                    "pair_id": pair_id,
                    "seed": seed,
                    "spatial_block_id": block,
                    "protocol_role": "study_development",
                    "target": target,
                    "class_order": json.dumps(list(SEVERITY_CLASS_ORDER)),
                }
                for column in PROBABILITY_COLUMNS:
                    row[column] = 0.05
                for view, predicted in (
                    ("street", street_class),
                    ("overhead", overhead_class),
                ):
                    for class_index, class_name in enumerate(SEVERITY_CLASS_ORDER):
                        row[f"{view}_prob_{class_name}"] = (
                            0.90 if class_index == predicted else 0.05
                        )
                prediction_rows.append(row)
        self.predictions = pd.DataFrame(prediction_rows)
        self.manifest = pd.DataFrame(manifest_rows)
        self.ledger = pd.DataFrame(ledger_rows)
        self.predictions_path = root / "predictions.csv"
        self.manifest_path = root / "study_development.csv"
        self.ledger_path = root / "media_hash_ledger.csv"
        self.predictions.to_csv(self.predictions_path, index=False, lineterminator="\n")
        self.manifest.to_csv(self.manifest_path, index=False, lineterminator="\n")
        self.ledger.to_csv(self.ledger_path, index=False, lineterminator="\n")

        protocol_hash = sha256_file(PROTOCOL_PATH)
        role_hashes = {
            "model_fit": "a" * 64,
            "model_validation": "b" * 64,
            "study_development": sha256_file(self.manifest_path),
        }
        self.protocol_summary_path = root / "protocol_summary.json"
        self.protocol_summary = {
            "schema_version": "eaton-component-direction-spatial-summary-v1",
            "protocol_version": self.protocol["protocol_version"],
            "protocol_config_sha256": protocol_hash,
            "role_manifest_sha256": {
                **role_hashes,
                "spatial_confirmation": "c" * 64,
            },
            "media_hash_ledger_sha256": sha256_file(self.ledger_path),
            "role_rows": {"study_development": len(self.manifest)},
            "analysis_status": "PROTOCOL_READY_CONFIRMATION_UNTOUCHED",
            "confirmation_commitment": {
                "status": "UNSCORED_RESERVED_SAME_EVENT_SPATIAL_CONFIRMATION"
            },
        }
        self.protocol_summary_path.write_text(
            json.dumps(self.protocol_summary), encoding="utf-8"
        )
        self.prediction_metadata_path = root / "prediction_metadata.json"
        self.prediction_metadata = {
            "schema_version": "eaton-component-direction-predictions-v1",
            "protocol_version": self.protocol["protocol_version"],
            "status": "PROTOCOL_DEVELOPMENT_RUN",
            "protocol_config_sha256": protocol_hash,
            "role_manifest_sha256": role_hashes,
            "prediction_role": "study_development",
            "spatial_confirmation_read_or_scored": False,
            "class_order": self.protocol["label_scheme"]["class_order"],
            "seeds": list(self.seeds),
            "epochs": self.protocol["models"]["epochs"],
            "batch_size": self.protocol["models"]["batch_size"],
            "ensemble_rule": self.protocol["models"]["ensemble_rule"],
            "prediction_csv_sha256": sha256_file(self.predictions_path),
        }
        self.prediction_metadata_path.write_text(
            json.dumps(self.prediction_metadata), encoding="utf-8"
        )

    def write(self, output: Path, *, overwrite: bool = False) -> dict[str, object]:
        return write_annotation_packet(
            predictions_path=self.predictions_path,
            manifest_path=self.manifest_path,
            media_ledger_path=self.ledger_path,
            protocol_path=PROTOCOL_PATH,
            output_dir=output,
            overwrite=overwrite,
            blinding_secret=self.blinding_secret,
        )

    def complete_templates(self, output: Path) -> tuple[Path, Path]:
        completed_dir = self.root / f"completed-{output.name}"
        completed_dir.mkdir(exist_ok=True)
        paths = []
        for letter, rater_id in (("a", "human-rater-a"), ("b", "human-rater-b")):
            frame = pd.read_csv(
                output / "public" / f"rater_{letter}_template.csv",
                dtype=str,
                keep_default_na=False,
            )
            frame["rater_id"] = rater_id
            frame["component_dominance"] = "roof"
            frame["street_roof_assessability"] = "not_assessable"
            frame["street_facade_assessability"] = "indeterminate"
            frame["overhead_roof_assessability"] = "assessable"
            frame["overhead_facade_assessability"] = "indeterminate"
            path = completed_dir / f"rater_{letter}_completed.csv"
            frame.to_csv(path, index=False, lineterminator="\n")
            paths.append(path)
        return paths[0], paths[1]


class MetadataDerivativeTest(unittest.TestCase):
    def test_jpeg_gps_datetime_and_png_text_are_stripped_with_exact_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jpeg = root / "private.jpg"
            png = root / "private.png"
            _jpeg_with_private_exif(jpeg, "secret-source-id", (20, 40, 80))
            _png_with_private_text(png, "secret-source-id", (90, 50, 10))
            with Image.open(jpeg) as source_jpeg:
                self.assertIn(34853, source_jpeg.getexif())
                self.assertEqual(source_jpeg.getexif()[306], "2026:01:02 03:04:05")
                self.assertIn("icc_profile", source_jpeg.info)
            with Image.open(png) as source_png:
                self.assertIn("XML:com.adobe.xmp", source_png.info)
                self.assertIn("Comment", source_png.info)

            for source in (jpeg, png):
                derivative, lineage = metadata_stripped_derivative(source)
                source_pixels, source_size = _rgb_sha256(source.read_bytes())
                derivative_pixels, derivative_size = _rgb_sha256(derivative)
                self.assertEqual(source_pixels, derivative_pixels)
                self.assertEqual(source_size, derivative_size)
                self.assertEqual(source_pixels, lineage["decoded_rgb_sha256"])
                self.assertNotEqual(
                    lineage["source_media_sha256"],
                    lineage["derivative_media_sha256"],
                )
                self.assertNotIn(b"secret-source-id", derivative)
                with Image.open(BytesIO(derivative)) as clean:
                    self.assertEqual(len(clean.getexif()), 0)
                    leaked_keys = {
                        key.casefold()
                        for key in clean.info
                        if any(
                            marker in key.casefold()
                            for marker in (
                                "exif",
                                "xmp",
                                "icc",
                                "profile",
                                "comment",
                                "text",
                                "time",
                            )
                        )
                    }
                    self.assertEqual(leaked_keys, set())

    def test_mpo_is_normalized_to_pixel_equivalent_primary_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "two-frame.mpo"
            primary = Image.new("RGB", (17, 13), (12, 34, 56))
            secondary = Image.new("RGB", (17, 13), (210, 20, 30))
            primary.save(
                path,
                format="MPO",
                save_all=True,
                append_images=[secondary],
                quality=93,
                icc_profile=b"private-mpo-device-profile",
            )
            source_pixels, source_size = _rgb_sha256(path.read_bytes())
            derivative, lineage = metadata_stripped_derivative(path)
            derivative_pixels, derivative_size = _rgb_sha256(derivative)
            self.assertEqual((source_pixels, source_size), (derivative_pixels, derivative_size))
            self.assertEqual(lineage["source_format"], "MPO")
            self.assertEqual(lineage["source_frame"], 0)
            self.assertEqual(lineage["format"], "JPEG")
            with Image.open(BytesIO(derivative)) as clean:
                self.assertEqual(clean.format, "JPEG")
                self.assertEqual(getattr(clean, "n_frames", 1), 1)
                self.assertNotIn("icc_profile", clean.info)

    def test_nontrivial_exif_orientation_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "orientation-6.jpg"
            image = Image.new("RGB", (10, 8), (1, 2, 3))
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, format="JPEG", exif=exif)
            with self.assertRaisesRegex(ValueError, "Orientation"):
                metadata_stripped_derivative(path)


class BlindedPacketTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = PacketFixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_deterministic_selection_separation_codebook_and_no_public_leak(self) -> None:
        output_a = self.root / "packet-a"
        output_b = self.root / "packet-b"
        metadata_a = self.fixture.write(output_a)
        metadata_b = self.fixture.write(output_b)
        public_dir = output_a / "public"
        restricted_dir = output_a / "restricted"

        self.assertEqual(
            {path.name for path in output_a.iterdir()}, {"public", "restricted"}
        )
        self.assertTrue((restricted_dir / "private_key.csv").is_file())
        self.assertTrue((restricted_dir / "packet_metadata.json").is_file())
        self.assertFalse((public_dir / "private_key.csv").exists())
        self.assertFalse((public_dir / "packet_metadata.json").exists())

        public = pd.read_csv(public_dir / "public_packet.csv", dtype=str)
        private = pd.read_csv(restricted_dir / "private_key.csv", dtype=str)
        self.assertEqual(tuple(public.columns), PUBLIC_PACKET_COLUMNS)
        self.assertEqual(tuple(private.columns), PRIVATE_KEY_COLUMNS)
        self.assertEqual(len(public), 2)
        self.assertEqual(set(private["pair_id"]), {"source-pair-a", "source-pair-b"})
        self.assertEqual(set(private["selection_rule"]), {FROZEN_SELECTION_RULE})

        protocol_hash = sha256_file(PROTOCOL_PATH)
        expected_ids = {
            opaque_item_id(self.fixture.blinding_secret, protocol_hash, pair_id)
            for pair_id in ("source-pair-a", "source-pair-b")
        }
        self.assertEqual(set(public["opaque_item_id"]), expected_ids)
        expected_order = [
            pair_id
            for _, pair_id in sorted(
                (
                    packet_order_sha256(
                        self.fixture.blinding_secret, protocol_hash, pair_id
                    ),
                    pair_id,
                )
                for pair_id in ("source-pair-a", "source-pair-b")
            )
        ]
        self.assertEqual(private["pair_id"].tolist(), expected_order)

        staged_codebook = json.loads(
            (public_dir / "annotation_codebook.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            staged_codebook,
            {
                "codebook_version": self.fixture.protocol["annotation"][
                    "codebook_version"
                ],
                "codebook": self.fixture.protocol["annotation"]["codebook"],
            },
        )
        self.assertEqual(
            metadata_a["codebook_sha256"],
            sha256_file(public_dir / "annotation_codebook.json"),
        )

        for public_path in public_dir.rglob("*"):
            if not public_path.is_file():
                continue
            content = public_path.read_bytes()
            for source_pair_id in self.fixture.pair_specs:
                self.assertNotIn(source_pair_id.encode(), content)
            self.assertNotIn(self.fixture.blinding_secret, content)
            self.assertNotIn(self.fixture.blinding_secret.hex().encode("ascii"), content)
        wrong_secret = bytes.fromhex("22" * 32)
        self.assertTrue(
            expected_ids.isdisjoint(
                {
                    opaque_item_id(wrong_secret, protocol_hash, pair_id)
                    for pair_id in self.fixture.pair_specs
                }
            )
        )
        for row in public.itertuples(index=False):
            street = public_dir / row.street_image_path
            overhead = public_dir / row.overhead_image_path
            self.assertEqual(sha256_file(street), row.street_media_sha256)
            self.assertEqual(sha256_file(overhead), row.overhead_media_sha256)
            with Image.open(street) as image:
                self.assertEqual(len(image.getexif()), 0)
            with Image.open(overhead) as image:
                self.assertEqual(len(image.getexif()), 0)
                self.assertNotIn("Comment", image.info)
                self.assertNotIn("XML:com.adobe.xmp", image.info)

        for relative in (
            "public/public_packet.csv",
            "public/rater_a_template.csv",
            "public/rater_b_template.csv",
            "public/annotation_codebook.json",
            "restricted/private_key.csv",
            "restricted/packet_metadata.json",
        ):
            self.assertEqual(
                (output_a / relative).read_bytes(), (output_b / relative).read_bytes()
            )
        self.assertEqual(metadata_a, metadata_b)
        self.assertTrue(metadata_a["development_pilot_only"])
        self.assertFalse(metadata_a["selection"]["uses_signed_direction"])
        self.assertEqual(metadata_a["input_counts"]["prediction_seeds"], 5)

    def test_completed_templates_import_only_through_bound_real_packet(self) -> None:
        output = self.root / "packet"
        self.fixture.write(output)
        rater_a_path, rater_b_path = self.fixture.complete_templates(output)
        remapped = remap_completed_rater_templates(
            rater_a_path=rater_a_path,
            rater_b_path=rater_b_path,
            public_packet_path=output / "public" / "public_packet.csv",
            private_key_path=output / "restricted" / "private_key.csv",
            packet_metadata_path=output / "restricted" / "packet_metadata.json",
            protocol_path=PROTOCOL_PATH,
        )
        self.assertEqual(set(remapped["pair_id"]), {"source-pair-a", "source-pair-b"})
        self.assertEqual(set(remapped["rater_id"]), {"human-rater-a", "human-rater-b"})

        evil_a = pd.read_csv(rater_a_path, dtype=str, keep_default_na=False)
        evil_b = pd.read_csv(rater_b_path, dtype=str, keep_default_na=False)
        evil_a["protocol_sha256"] = "e" * 64
        evil_b["protocol_sha256"] = "e" * 64
        evil_a_path = self.root / "evil-a.csv"
        evil_b_path = self.root / "evil-b.csv"
        evil_a.to_csv(evil_a_path, index=False, lineterminator="\n")
        evil_b.to_csv(evil_b_path, index=False, lineterminator="\n")
        with self.assertRaisesRegex(ValueError, "real protocol"):
            remap_completed_rater_templates(
                rater_a_path=evil_a_path,
                rater_b_path=evil_b_path,
                public_packet_path=output / "public" / "public_packet.csv",
                private_key_path=output / "restricted" / "private_key.csv",
                packet_metadata_path=output / "restricted" / "packet_metadata.json",
                protocol_path=PROTOCOL_PATH,
            )

    def test_swapped_private_mapping_fails_even_if_attacker_rehashes_metadata(self) -> None:
        output = self.root / "packet"
        self.fixture.write(output)
        rater_a_path, rater_b_path = self.fixture.complete_templates(output)
        restricted = output / "restricted"
        swapped = pd.read_csv(
            restricted / "private_key.csv", dtype=str, keep_default_na=False
        )
        swapped.loc[[0, 1], "pair_id"] = swapped.loc[[1, 0], "pair_id"].to_numpy()
        swapped_path = restricted / "swapped_private_key.csv"
        swapped.to_csv(swapped_path, index=False, lineterminator="\n")
        with self.assertRaisesRegex(ValueError, "Artifact hash"):
            remap_completed_rater_templates(
                rater_a_path=rater_a_path,
                rater_b_path=rater_b_path,
                public_packet_path=output / "public" / "public_packet.csv",
                private_key_path=swapped_path,
                packet_metadata_path=restricted / "packet_metadata.json",
                protocol_path=PROTOCOL_PATH,
            )
        metadata = json.loads(
            (restricted / "packet_metadata.json").read_text(encoding="utf-8")
        )
        metadata["output_sha256"]["restricted/private_key.csv"] = sha256_file(
            swapped_path
        )
        evil_metadata = restricted / "evil_packet_metadata.json"
        evil_metadata.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "opaque mapping"):
            remap_completed_rater_templates(
                rater_a_path=rater_a_path,
                rater_b_path=rater_b_path,
                public_packet_path=output / "public" / "public_packet.csv",
                private_key_path=swapped_path,
                packet_metadata_path=evil_metadata,
                protocol_path=PROTOCOL_PATH,
            )

    def test_tampered_blinding_secret_fails_even_if_output_hash_is_rewritten(self) -> None:
        output = self.root / "packet"
        self.fixture.write(output)
        rater_a_path, rater_b_path = self.fixture.complete_templates(output)
        restricted = output / "restricted"
        tampered_secret = restricted / "tampered_blinding_secret.key"
        tampered_secret.write_bytes(bytes.fromhex("33" * 32))
        metadata = json.loads(
            (restricted / "packet_metadata.json").read_text(encoding="utf-8")
        )
        metadata["output_sha256"]["restricted/blinding_secret.key"] = sha256_file(
            tampered_secret
        )
        evil_metadata = restricted / "evil_secret_metadata.json"
        evil_metadata.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "secret commitment"):
            remap_completed_rater_templates(
                rater_a_path=rater_a_path,
                rater_b_path=rater_b_path,
                public_packet_path=output / "public" / "public_packet.csv",
                private_key_path=restricted / "private_key.csv",
                packet_metadata_path=evil_metadata,
                protocol_path=PROTOCOL_PATH,
                blinding_secret_path=tampered_secret,
            )
    def test_refuses_smoke_role_id_hash_and_overwrite_failures(self) -> None:
        wrong_role = self.fixture.predictions.copy()
        wrong_role["protocol_role"] = "spatial_confirmation"
        with self.assertRaisesRegex(ValueError, "study_development"):
            build_packet_tables(
                predictions=wrong_role,
                manifest=self.fixture.manifest,
                media_ledger=self.fixture.ledger,
                protocol=self.fixture.protocol,
                protocol_sha256="a" * 64,
                blinding_secret=self.fixture.blinding_secret,
            )
        missing_id = self.fixture.ledger.iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "missing development pair ids"):
            build_packet_tables(
                predictions=self.fixture.predictions,
                manifest=self.fixture.manifest,
                media_ledger=missing_id,
                protocol=self.fixture.protocol,
                protocol_sha256="a" * 64,
                blinding_secret=self.fixture.blinding_secret,
            )

        smoke_metadata = dict(self.fixture.prediction_metadata)
        smoke_metadata["status"] = "NON_PROTOCOL_SMOKE_OR_PARTIAL_RUN"
        self.fixture.prediction_metadata_path.write_text(
            json.dumps(smoke_metadata), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "smoke or partial"):
            self.fixture.write(self.root / "forbidden-smoke")
        self.fixture.prediction_metadata_path.write_text(
            json.dumps(self.fixture.prediction_metadata), encoding="utf-8"
        )

        output = self.root / "packet"
        self.fixture.write(output)
        with self.assertRaises(FileExistsError):
            self.fixture.write(output)
        original = (output / "public" / "public_packet.csv").read_bytes()
        self.fixture.write(output, overwrite=True)
        self.assertEqual(
            original, (output / "public" / "public_packet.csv").read_bytes()
        )


if __name__ == "__main__":
    unittest.main()
