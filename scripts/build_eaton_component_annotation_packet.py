#!/usr/bin/env python
"""Build a blinded development-only Eaton component annotation packet.

The public packet stages opaque-named media copies and never contains source
pair ids, labels, predictions, probabilities, confidence, disagreement sign,
spatial blocks, or protocol roles.  A separately stored private key is the
only artifact that maps opaque ids back to the frozen development manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import secrets
import sys
from typing import Any, Mapping
import zlib

import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.analysis.eaton_component_direction import (
    ANNOTATION_FIELDS,
    PROTOCOL_ROLE,
    ensemble_seed_predictions,
    validate_annotation_packet,
    validate_seed_predictions,
)
from scripts.build_eaton_component_direction_protocol import load_protocol


PACKET_SCHEMA = "eaton-component-blinded-annotation-packet-v1"
PREDICTION_METADATA_SCHEMA = "eaton-component-direction-predictions-v1"
PROTOCOL_SUMMARY_SCHEMA = "eaton-component-direction-spatial-summary-v1"
FROZEN_SELECTION_RULE = (
    "all unique pairs with ensemble street-overhead disagreement in the named "
    "role; selection never uses signed direction"
)
PUBLIC_PACKET_COLUMNS = (
    "opaque_item_id",
    "street_image_path",
    "overhead_image_path",
    "street_media_sha256",
    "overhead_media_sha256",
)
PRIVATE_KEY_COLUMNS = (
    "opaque_item_id",
    "pair_id",
    "spatial_block_id",
    "protocol_role",
    "protocol_version",
    "protocol_sha256",
    "codebook_version",
    "codebook_sha256",
    "selection_rule",
    "selected_for_ensemble_disagreement",
    "packet_position",
    "packet_order_sha256",
    "source_street_media_sha256",
    "derivative_street_media_sha256",
    "street_decoded_rgb_sha256",
    "street_width",
    "street_height",
    "street_format",
    "street_entropy_or_idat_sha256",
    "source_overhead_media_sha256",
    "derivative_overhead_media_sha256",
    "overhead_decoded_rgb_sha256",
    "overhead_width",
    "overhead_height",
    "overhead_format",
    "overhead_entropy_or_idat_sha256",
    "media_derivative_rule",
)
RATER_TEMPLATE_COLUMNS = (
    "pair_id",
    "rater_id",
    *ANNOTATION_FIELDS,
    "protocol_version",
    "protocol_sha256",
    "codebook_version",
    "codebook_sha256",
    "blinded_to_model_outputs",
    "street_media_sha256",
    "overhead_media_sha256",
)
MANIFEST_COLUMNS = {
    "pair_id",
    "spatial_block_id",
    "protocol_role",
    "label",
    "street_view_path",
    "remote_sensing_path",
    "street_sha256",
    "remote_sha256",
}
LEDGER_COLUMNS = {
    "pair_id",
    "street_view_path",
    "remote_sensing_path",
    "street_sha256",
    "remote_sha256",
}
ROOT_OUTPUT_NAMES = {
    "public",
    "restricted",
}
PUBLIC_OUTPUT_NAMES = {
    "public_packet.csv",
    "rater_a_template.csv",
    "rater_b_template.csv",
    "annotation_codebook.json",
    "public_media",
}
RESTRICTED_OUTPUT_NAMES = {
    "private_key.csv",
    "packet_metadata.json",
    "blinding_secret.key",
}
RATER_A_PLACEHOLDER = "REPLACE_WITH_UNIQUE_RATER_A_ID"
RATER_B_PLACEHOLDER = "REPLACE_WITH_UNIQUE_RATER_B_ID"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_JPEG_SUFFIXES = {".jpg", ".jpeg"}
_PNG_SUFFIXES = {".png"}
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_STRIPPED_CHUNKS = {b"eXIf", b"tEXt", b"zTXt", b"iTXt", b"tIME"}
_JPEG_METADATA_MARKERS = {0xE1, 0xED, 0xFE}
MEDIA_DERIVATIVE_RULE = (
    "metadata-stripped, decoded-RGB-pixel-equivalent derivative; JPEG APP1/APP13/"
    "COM and recognized EXIF/XMP/IPTC/ICC APP segments removed without changing "
    "primary-frame scan entropy; MPO sources are normalized to their primary JPEG "
    "frame; PNG eXIf/text/time chunks removed without changing critical pixel data"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _codebook_artifact(protocol: Mapping[str, Any]) -> tuple[dict[str, object], bytes, str]:
    annotation = protocol.get("annotation", {})
    version = str(annotation.get("codebook_version", "")).strip()
    codebook = annotation.get("codebook")
    if not version or not isinstance(codebook, dict) or not codebook:
        raise ValueError("Frozen annotation codebook_version/codebook is missing")
    payload: dict[str, object] = {
        "codebook_version": version,
        "codebook": codebook,
    }
    data = _canonical_json_bytes(payload)
    return payload, data, hashlib.sha256(data).hexdigest()


def _jpeg_segment_contains_metadata(payload: bytes) -> bool:
    lowered = payload.lower()
    signatures = (
        b"exif\x00\x00",
        b"http://ns.adobe.com/xap/1.0/",
        b"http://ns.adobe.com/xmp/extension/",
        b"photoshop 3.0",
        b"iptc",
        b"icc_profile\x00",
        b"mpf\x00",
    )
    return any(signature in lowered for signature in signatures)


def _strip_jpeg_metadata(data: bytes) -> tuple[bytes, str]:
    """Strip sensitive JPEG marker segments without touching scan entropy bytes."""

    if not data.startswith(b"\xff\xd8"):
        raise ValueError("JPEG derivative input is missing the SOI marker")
    output = bytearray(data[:2])
    entropy = hashlib.sha256()
    index = 2
    saw_eoi = False
    while index < len(data):
        if data[index] != 0xFF:
            raise ValueError("Malformed JPEG data outside an entropy-coded scan")
        marker_start = index
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            raise ValueError("Truncated JPEG marker")
        marker = data[index]
        index += 1
        marker_prefix = data[marker_start:index]
        if marker == 0x00:
            raise ValueError("Unexpected stuffed byte outside a JPEG scan")
        if marker == 0xD9:
            output.extend(marker_prefix)
            saw_eoi = True
            break
        if marker in {0xD8, 0x01, *range(0xD0, 0xD8)}:
            output.extend(marker_prefix)
            continue
        if index + 2 > len(data):
            raise ValueError("Truncated JPEG segment length")
        segment_length = int.from_bytes(data[index : index + 2], "big")
        if segment_length < 2 or index + segment_length > len(data):
            raise ValueError("Invalid JPEG segment length")
        segment_end = index + segment_length
        payload = data[index + 2 : segment_end]
        segment = marker_prefix + data[index:segment_end]
        remove = marker in _JPEG_METADATA_MARKERS or (
            0xE0 <= marker <= 0xEF and _jpeg_segment_contains_metadata(payload)
        )
        if not remove:
            output.extend(segment)
        index = segment_end
        if marker != 0xDA:
            continue
        if remove:
            raise RuntimeError("A JPEG SOS segment cannot be classified as metadata")
        # Copy every entropy byte verbatim until a non-stuffed, non-restart marker.
        scan_start = index
        while True:
            marker_candidate = data.find(b"\xff", index)
            if marker_candidate < 0:
                raise ValueError("JPEG entropy scan has no terminating marker")
            after_ff = marker_candidate
            while after_ff < len(data) and data[after_ff] == 0xFF:
                after_ff += 1
            if after_ff >= len(data):
                raise ValueError("Truncated JPEG entropy marker")
            scan_code = data[after_ff]
            if scan_code == 0x00 or 0xD0 <= scan_code <= 0xD7:
                chunk = data[scan_start : after_ff + 1]
                output.extend(chunk)
                entropy.update(chunk)
                index = after_ff + 1
                scan_start = index
                continue
            chunk = data[scan_start:marker_candidate]
            output.extend(chunk)
            entropy.update(chunk)
            index = marker_candidate
            break
    if not saw_eoi:
        raise ValueError("JPEG derivative input has no EOI marker")
    return bytes(output), entropy.hexdigest()


def _strip_png_metadata(data: bytes) -> tuple[bytes, str]:
    """Strip PNG EXIF/text/time chunks while preserving all pixel-bearing chunks."""

    if not data.startswith(_PNG_SIGNATURE):
        raise ValueError("PNG derivative input has an invalid signature")
    output = bytearray(_PNG_SIGNATURE)
    idat = hashlib.sha256()
    index = len(_PNG_SIGNATURE)
    saw_ihdr = False
    saw_iend = False
    while index < len(data):
        if index + 12 > len(data):
            raise ValueError("Truncated PNG chunk")
        length = int.from_bytes(data[index : index + 4], "big")
        end = index + 12 + length
        if end > len(data):
            raise ValueError("PNG chunk length exceeds the file")
        chunk_type = data[index + 4 : index + 8]
        chunk_data = data[index + 8 : index + 8 + length]
        expected_crc = int.from_bytes(data[index + 8 + length : end], "big")
        actual_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            raise ValueError(f"PNG chunk {chunk_type!r} has an invalid CRC")
        if chunk_type == b"IHDR":
            saw_ihdr = True
        if chunk_type == b"IDAT":
            idat.update(chunk_data)
        if chunk_type not in _PNG_STRIPPED_CHUNKS:
            output.extend(data[index:end])
        index = end
        if chunk_type == b"IEND":
            saw_iend = True
            break
    if not saw_ihdr or not saw_iend:
        raise ValueError("PNG must contain IHDR and IEND chunks")
    return bytes(output), idat.hexdigest()


def _decoded_rgb_fingerprint(
    data: bytes, *, require_safe_orientation: bool
) -> dict[str, object]:
    with Image.open(BytesIO(data)) as image:
        image.load()
        exif = image.getexif()
        orientation = exif.get(274)
        if require_safe_orientation and orientation not in (None, 1):
            raise ValueError(
                "Frozen protocol excludes source media with EXIF Orientation other than 1"
            )
        rgb = image.convert("RGB")
        return {
            "format": str(image.format or "").upper(),
            "width": int(image.width),
            "height": int(image.height),
            "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "exif_count": int(len(exif)),
            "info_keys": tuple(sorted(str(key).casefold() for key in image.info)),
        }


def _verify_derivative_has_no_sensitive_metadata(
    data: bytes, image_format: str
) -> None:
    fingerprint = _decoded_rgb_fingerprint(data, require_safe_orientation=False)
    if fingerprint["exif_count"]:
        raise RuntimeError("Metadata-stripped derivative still contains EXIF")
    forbidden_markers = (
        "exif",
        "xmp",
        "iptc",
        "icc",
        "profile",
        "comment",
        "description",
        "date",
        "time",
        "xml",
    )
    leaked = [
        key
        for key in fingerprint["info_keys"]
        if any(marker in key for marker in forbidden_markers)
    ]
    if leaked:
        raise RuntimeError(f"Metadata-stripped derivative exposes metadata keys: {leaked}")
    if image_format == "JPEG":
        restaged, _ = _strip_jpeg_metadata(data)
        if restaged != data:
            raise RuntimeError("JPEG derivative still contains removable metadata")
    elif image_format == "PNG":
        restaged, _ = _strip_png_metadata(data)
        if restaged != data:
            raise RuntimeError("PNG derivative still contains removable metadata")


def metadata_stripped_derivative(path: Path) -> tuple[bytes, dict[str, object]]:
    """Return a deterministic metadata-free derivative and auditable lineage."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    source_data = source.read_bytes()
    source_fingerprint = _decoded_rgb_fingerprint(
        source_data, require_safe_orientation=True
    )
    source_format = str(source_fingerprint["format"])
    if source_format in {"JPEG", "MPO"}:
        # An MPO is a JPEG container.  The training loader consumes its primary
        # frame; stopping at that frame's EOI produces the same decoded RGB
        # pixels without exposing secondary views or MPF/device metadata.
        image_format = "JPEG"
        derivative, stream_hash = _strip_jpeg_metadata(source_data)
        verified, derivative_stream_hash = _strip_jpeg_metadata(derivative)
    elif source_format == "PNG":
        image_format = "PNG"
        derivative, stream_hash = _strip_png_metadata(source_data)
        verified, derivative_stream_hash = _strip_png_metadata(derivative)
    else:
        raise ValueError(
            f"Unsupported public-derivative format {source_format!r}; only JPEG/MPO/PNG are allowed"
        )
    if verified != derivative or derivative_stream_hash != stream_hash:
        raise RuntimeError("Metadata stripping is not idempotent or changed pixel-stream data")
    derivative_fingerprint = _decoded_rgb_fingerprint(
        derivative, require_safe_orientation=False
    )
    if derivative_fingerprint["format"] != image_format:
        raise RuntimeError("Metadata stripping produced an unexpected image format")
    for key in ("width", "height", "decoded_rgb_sha256"):
        if derivative_fingerprint[key] != source_fingerprint[key]:
            raise RuntimeError(
                "Metadata stripping changed decoded RGB pixels, dimensions, or format"
            )
    _verify_derivative_has_no_sensitive_metadata(derivative, image_format)
    lineage: dict[str, object] = {
        "source_media_sha256": hashlib.sha256(source_data).hexdigest(),
        "derivative_media_sha256": hashlib.sha256(derivative).hexdigest(),
        "decoded_rgb_sha256": source_fingerprint["decoded_rgb_sha256"],
        "width": source_fingerprint["width"],
        "height": source_fingerprint["height"],
        "format": image_format,
        "source_format": source_format,
        "source_frame": 0,
        "entropy_or_idat_sha256": stream_hash,
        "source_exif_count": source_fingerprint["exif_count"],
        "derivative_rule": MEDIA_DERIVATIVE_RULE,
    }
    return derivative, lineage


def _validated_blinding_secret(value: bytes | bytearray) -> bytes:
    if not isinstance(value, (bytes, bytearray)) or len(value) != 32:
        raise ValueError("blinding_secret must contain exactly 32 random bytes")
    return bytes(value)


def _hmac_digest(
    blinding_secret: bytes, domain: str, protocol_sha256: str, pair_id: str
) -> str:
    secret = _validated_blinding_secret(blinding_secret)
    value = f"{domain}\0{protocol_sha256}\0{pair_id}".encode("utf-8")
    return hmac.new(secret, value, hashlib.sha256).hexdigest()


def opaque_item_id(
    blinding_secret: bytes, protocol_sha256: str, pair_id: str
) -> str:
    """Return a secret-keyed opaque id that public artifacts cannot enumerate."""

    return "eaton_" + _hmac_digest(
        blinding_secret, "eaton-component-opaque-item-v1", protocol_sha256, pair_id
    )


def packet_order_sha256(
    blinding_secret: bytes, protocol_sha256: str, pair_id: str
) -> str:
    """Return a separately domain-keyed private packet-ordering digest."""

    return _hmac_digest(
        blinding_secret, "eaton-component-packet-order-v1", protocol_sha256, pair_id
    )


def _identifiers(values: pd.Series, name: str) -> pd.Series:
    result = values.astype("string").str.strip()
    if result.isna().any() or result.eq("").any():
        raise ValueError(f"{name} cannot be null or blank")
    return result.astype(str)


def _hashes(values: pd.Series, name: str) -> pd.Series:
    result = values.astype("string").str.strip().str.casefold()
    if result.isna().any() or not result.map(
        lambda value: bool(_SHA256.fullmatch(str(value)))
    ).all():
        raise ValueError(f"{name} must contain complete SHA-256 values")
    return result.astype(str)


def _normalized_path(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Media paths cannot be blank")
    return os.path.normcase(os.path.abspath(os.path.normpath(text)))


def _validate_protocol(protocol: Mapping[str, Any]) -> tuple[str, tuple[int, ...]]:
    protocol_version = str(protocol.get("protocol_version", "")).strip()
    if not protocol_version:
        raise ValueError("Frozen protocol_version is missing")
    model_config = protocol.get("models", {})
    seeds = tuple(int(value) for value in model_config.get("seeds", ()))
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("The annotation packet requires exactly five frozen model seeds")
    ensemble_rule = str(model_config.get("ensemble_rule", "")).casefold()
    if "mean" not in ensemble_rule or "probabil" not in ensemble_rule:
        raise ValueError("Frozen ensemble_rule must average per-seed probabilities")
    annotation = protocol.get("annotation", {})
    if annotation.get("selection_rule") != FROZEN_SELECTION_RULE:
        raise ValueError("Frozen annotation selection_rule changed")
    if int(annotation.get("required_independent_raters", 0)) != 2:
        raise ValueError("Frozen protocol must require exactly two independent raters")
    _codebook_artifact(protocol)
    return protocol_version, seeds


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read valid {name} JSON from {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _validate_frozen_sidecars(
    *,
    prediction_metadata_path: Path,
    protocol_summary_path: Path,
    predictions_path: Path,
    manifest_path: Path,
    media_ledger_path: Path,
    protocol_path: Path,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prove the inputs come from the frozen full development protocol run."""

    prediction_metadata = _read_json_object(
        prediction_metadata_path, "prediction metadata"
    )
    protocol_summary = _read_json_object(protocol_summary_path, "protocol summary")
    protocol_version, expected_seeds = _validate_protocol(protocol)
    protocol_hash = sha256_file(protocol_path)
    prediction_hash = sha256_file(predictions_path)
    manifest_hash = sha256_file(manifest_path)
    ledger_hash = sha256_file(media_ledger_path)

    if prediction_metadata.get("schema_version") != PREDICTION_METADATA_SCHEMA:
        raise ValueError("Prediction metadata schema is not the frozen protocol schema")
    if prediction_metadata.get("status") != "PROTOCOL_DEVELOPMENT_RUN":
        raise ValueError(
            "Annotation packets require status=PROTOCOL_DEVELOPMENT_RUN; "
            "smoke or partial predictions are prohibited"
        )
    if prediction_metadata.get("protocol_version") != protocol_version:
        raise ValueError("Prediction metadata protocol_version mismatch")
    if prediction_metadata.get("protocol_config_sha256") != protocol_hash:
        raise ValueError("Prediction metadata protocol config hash mismatch")
    if prediction_metadata.get("prediction_csv_sha256") != prediction_hash:
        raise ValueError("Prediction CSV hash does not match prediction metadata")
    if prediction_metadata.get("prediction_role") != PROTOCOL_ROLE:
        raise ValueError("Prediction metadata is not study_development-only")
    if prediction_metadata.get("spatial_confirmation_read_or_scored") is not False:
        raise ValueError("Prediction metadata must attest untouched spatial confirmation")
    if tuple(prediction_metadata.get("seeds", ())) != expected_seeds:
        raise ValueError("Prediction metadata seeds differ from the five frozen seeds")
    model_config = protocol["models"]
    if prediction_metadata.get("epochs") != int(model_config["epochs"]):
        raise ValueError("Prediction metadata epochs differ from the frozen protocol")
    if prediction_metadata.get("batch_size") != int(model_config["batch_size"]):
        raise ValueError("Prediction metadata batch_size differs from the frozen protocol")
    if prediction_metadata.get("ensemble_rule") != model_config["ensemble_rule"]:
        raise ValueError("Prediction metadata ensemble_rule mismatch")
    if prediction_metadata.get("class_order") != protocol["label_scheme"]["class_order"]:
        raise ValueError("Prediction metadata class_order mismatch")

    if protocol_summary.get("schema_version") != PROTOCOL_SUMMARY_SCHEMA:
        raise ValueError("Protocol summary schema mismatch")
    if protocol_summary.get("protocol_version") != protocol_version:
        raise ValueError("Protocol summary protocol_version mismatch")
    if protocol_summary.get("protocol_config_sha256") != protocol_hash:
        raise ValueError("Protocol summary config hash mismatch")
    summary_role_hashes = protocol_summary.get("role_manifest_sha256")
    if not isinstance(summary_role_hashes, dict):
        raise ValueError("Protocol summary is missing role manifest hashes")
    if summary_role_hashes.get(PROTOCOL_ROLE) != manifest_hash:
        raise ValueError("Development manifest hash does not match protocol summary")
    if protocol_summary.get("media_hash_ledger_sha256") != ledger_hash:
        raise ValueError("Media hash ledger hash does not match protocol summary")
    if protocol_summary.get("analysis_status") != (
        "PROTOCOL_READY_CONFIRMATION_UNTOUCHED"
    ):
        raise ValueError("Protocol summary does not attest untouched confirmation")
    commitment = protocol_summary.get("confirmation_commitment", {})
    if commitment.get("status") != (
        "UNSCORED_RESERVED_SAME_EVENT_SPATIAL_CONFIRMATION"
    ):
        raise ValueError("Confirmation commitment is not in its reserved unscored state")

    training_roles = ("model_fit", "model_validation", PROTOCOL_ROLE)
    expected_role_hashes = {
        role: summary_role_hashes.get(role) for role in training_roles
    }
    metadata_role_hashes = prediction_metadata.get("role_manifest_sha256")
    if (
        not isinstance(metadata_role_hashes, dict)
        or set(metadata_role_hashes) != set(training_roles)
        or metadata_role_hashes != expected_role_hashes
        or any(not _SHA256.fullmatch(str(value)) for value in expected_role_hashes.values())
    ):
        raise ValueError(
            "Prediction metadata role manifest hashes do not exactly match the "
            "frozen protocol summary"
        )
    return prediction_metadata, protocol_summary


def _validate_inputs(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    media_ledger: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, tuple[int, ...]]:
    protocol_version, expected_seeds = _validate_protocol(protocol)
    validated_predictions = validate_seed_predictions(predictions)
    actual_seeds = tuple(sorted(validated_predictions["seed"].unique().tolist()))
    if actual_seeds != tuple(sorted(expected_seeds)):
        raise ValueError(
            "Prediction seeds do not exactly match the five frozen protocol seeds"
        )

    missing_manifest = sorted(MANIFEST_COLUMNS - set(manifest.columns))
    if missing_manifest:
        raise ValueError(f"Development manifest missing columns: {missing_manifest}")
    validated_manifest = manifest.copy()
    validated_manifest["pair_id"] = _identifiers(
        validated_manifest["pair_id"], "manifest pair_id"
    )
    if validated_manifest["pair_id"].duplicated().any():
        raise ValueError("Development manifest pair_id must be unique")
    roles = _identifiers(validated_manifest["protocol_role"], "protocol_role")
    if not roles.eq(PROTOCOL_ROLE).all():
        raise ValueError(
            f"Manifest must contain only protocol_role={PROTOCOL_ROLE!r}"
        )
    validated_manifest["protocol_role"] = roles
    validated_manifest["spatial_block_id"] = _identifiers(
        validated_manifest["spatial_block_id"], "manifest spatial_block_id"
    )
    labels = pd.to_numeric(validated_manifest["label"], errors="coerce")
    if labels.isna().any() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError("Manifest label must contain integer class indices")
    validated_manifest["label"] = labels.astype(int)
    for column in ("street_sha256", "remote_sha256"):
        validated_manifest[column] = _hashes(validated_manifest[column], column)

    prediction_ids = set(validated_predictions["pair_id"])
    manifest_ids = set(validated_manifest["pair_id"])
    if prediction_ids != manifest_ids:
        missing = sorted(manifest_ids - prediction_ids)[:5]
        extra = sorted(prediction_ids - manifest_ids)[:5]
        raise ValueError(
            "Prediction/manifest pair_id sets must match exactly; "
            f"missing={missing}, extra={extra}"
        )
    prediction_meta = (
        validated_predictions.groupby("pair_id", sort=True)
        .agg(
            spatial_block_id=("spatial_block_id", "first"),
            target=("target", "first"),
        )
        .sort_index()
    )
    manifest_meta = validated_manifest.set_index("pair_id").sort_index()
    if not prediction_meta["spatial_block_id"].astype(str).eq(
        manifest_meta["spatial_block_id"].astype(str)
    ).all():
        raise ValueError("Prediction spatial_block_id does not match the manifest")
    if not prediction_meta["target"].astype(int).eq(
        manifest_meta["label"].astype(int)
    ).all():
        raise ValueError("Prediction target does not match the frozen manifest label")

    missing_ledger = sorted(LEDGER_COLUMNS - set(media_ledger.columns))
    if missing_ledger:
        raise ValueError(f"Media hash ledger missing columns: {missing_ledger}")
    validated_ledger = media_ledger.copy()
    validated_ledger["pair_id"] = _identifiers(
        validated_ledger["pair_id"], "ledger pair_id"
    )
    if validated_ledger["pair_id"].duplicated().any():
        raise ValueError("Media hash ledger pair_id must be unique")
    for column in ("street_sha256", "remote_sha256"):
        validated_ledger[column] = _hashes(validated_ledger[column], column)
    ledger_index = validated_ledger.set_index("pair_id", drop=False)
    missing_ids = sorted(manifest_ids - set(ledger_index.index))
    if missing_ids:
        raise ValueError(
            "Media hash ledger is missing development pair ids: "
            f"{missing_ids[:5]}"
        )
    aligned_ledger = ledger_index.loc[manifest_meta.index]
    for column in (
        "street_sha256",
        "remote_sha256",
        "street_view_path",
        "remote_sensing_path",
    ):
        left = manifest_meta[column]
        right = aligned_ledger[column]
        if column.endswith("_path"):
            agrees = left.map(_normalized_path).eq(right.map(_normalized_path))
        else:
            agrees = left.astype(str).str.casefold().eq(
                right.astype(str).str.casefold()
            )
        if not agrees.all():
            bad = agrees.index[~agrees].tolist()[:5]
            raise ValueError(
                f"Manifest/media-ledger {column} mismatch for pair ids {bad}"
            )
    return (
        validated_predictions,
        validated_manifest,
        validated_ledger,
        protocol_version,
        expected_seeds,
    )


def build_packet_tables(
    *,
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    media_ledger: pd.DataFrame,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    blinding_secret: bytes,
) -> dict[str, object]:
    """Validate inputs and create deterministic public, key, and rater tables."""

    protocol_sha256 = str(protocol_sha256).strip().casefold()
    if not _SHA256.fullmatch(protocol_sha256):
        raise ValueError("protocol_sha256 must be a complete SHA-256 value")
    blinding_secret = _validated_blinding_secret(blinding_secret)
    (
        validated_predictions,
        validated_manifest,
        _validated_ledger,
        protocol_version,
        _expected_seeds,
    ) = _validate_inputs(predictions, manifest, media_ledger, protocol)
    codebook_payload, _codebook_bytes, codebook_sha256 = _codebook_artifact(protocol)
    codebook_version = str(codebook_payload["codebook_version"])
    ensemble = ensemble_seed_predictions(validated_predictions)
    selected = ensemble.loc[
        ensemble["street_prediction"].ne(ensemble["overhead_prediction"]),
        ["pair_id", "spatial_block_id", "protocol_role", "seed_count"],
    ].copy()
    selected["opaque_item_id"] = selected["pair_id"].map(
        lambda pair_id: opaque_item_id(blinding_secret, protocol_sha256, pair_id)
    )
    selected["packet_order_sha256"] = selected["pair_id"].map(
        lambda pair_id: packet_order_sha256(
            blinding_secret, protocol_sha256, pair_id
        )
    )
    if selected["opaque_item_id"].duplicated().any():
        raise RuntimeError("Opaque item id collision")
    if selected["packet_order_sha256"].duplicated().any():
        raise RuntimeError("Packet ordering hash collision")
    selected = selected.sort_values(
        ["packet_order_sha256", "opaque_item_id"]
    ).reset_index(drop=True)
    selected["packet_position"] = np.arange(1, len(selected) + 1, dtype=int)

    manifest_index = validated_manifest.set_index("pair_id")
    public_rows: list[dict[str, object]] = []
    private_rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        media = manifest_index.loc[row.pair_id]
        derivatives: dict[str, tuple[bytes, dict[str, object]]] = {}
        for view, path_column, hash_column in (
            ("street", "street_view_path", "street_sha256"),
            ("overhead", "remote_sensing_path", "remote_sha256"),
        ):
            source_path = Path(str(media[path_column]))
            source_hash = sha256_file(source_path)
            if source_hash != str(media[hash_column]).casefold():
                raise ValueError(
                    f"Physical source hash does not match frozen {hash_column}"
                )
            derivative, lineage = metadata_stripped_derivative(source_path)
            if lineage["source_media_sha256"] != source_hash:
                raise RuntimeError("Derivative lineage source hash changed unexpectedly")
            derivatives[view] = (derivative, lineage)
        street_lineage = derivatives["street"][1]
        overhead_lineage = derivatives["overhead"][1]
        street_suffix = ".jpg" if street_lineage["format"] == "JPEG" else ".png"
        overhead_suffix = ".jpg" if overhead_lineage["format"] == "JPEG" else ".png"
        public_rows.append(
            {
                "opaque_item_id": row.opaque_item_id,
                "street_image_path": (
                    Path("public_media")
                    / f"{row.opaque_item_id}_street{street_suffix}"
                ).as_posix(),
                "overhead_image_path": (
                    Path("public_media")
                    / f"{row.opaque_item_id}_overhead{overhead_suffix}"
                ).as_posix(),
                "street_media_sha256": street_lineage["derivative_media_sha256"],
                "overhead_media_sha256": overhead_lineage["derivative_media_sha256"],
            }
        )
        private_rows.append(
            {
                "opaque_item_id": row.opaque_item_id,
                "pair_id": row.pair_id,
                "spatial_block_id": row.spatial_block_id,
                "protocol_role": row.protocol_role,
                "protocol_version": protocol_version,
                "protocol_sha256": protocol_sha256,
                "codebook_version": codebook_version,
                "codebook_sha256": codebook_sha256,
                "selection_rule": FROZEN_SELECTION_RULE,
                "selected_for_ensemble_disagreement": True,
                "packet_position": int(row.packet_position),
                "packet_order_sha256": row.packet_order_sha256,
                "source_street_media_sha256": street_lineage["source_media_sha256"],
                "derivative_street_media_sha256": street_lineage[
                    "derivative_media_sha256"
                ],
                "street_decoded_rgb_sha256": street_lineage["decoded_rgb_sha256"],
                "street_width": street_lineage["width"],
                "street_height": street_lineage["height"],
                "street_format": street_lineage["format"],
                "street_entropy_or_idat_sha256": street_lineage[
                    "entropy_or_idat_sha256"
                ],
                "source_overhead_media_sha256": overhead_lineage[
                    "source_media_sha256"
                ],
                "derivative_overhead_media_sha256": overhead_lineage[
                    "derivative_media_sha256"
                ],
                "overhead_decoded_rgb_sha256": overhead_lineage[
                    "decoded_rgb_sha256"
                ],
                "overhead_width": overhead_lineage["width"],
                "overhead_height": overhead_lineage["height"],
                "overhead_format": overhead_lineage["format"],
                "overhead_entropy_or_idat_sha256": overhead_lineage[
                    "entropy_or_idat_sha256"
                ],
                "media_derivative_rule": MEDIA_DERIVATIVE_RULE,
            }
        )
    public = pd.DataFrame(public_rows, columns=PUBLIC_PACKET_COLUMNS)
    private = pd.DataFrame(private_rows, columns=PRIVATE_KEY_COLUMNS)

    def rater_template(rater_id: str) -> pd.DataFrame:
        template = pd.DataFrame(
            {
                # Core validation calls this column pair_id.  Its values are opaque
                # packet ids, never the source manifest's pair ids.
                "pair_id": public["opaque_item_id"],
                "rater_id": rater_id,
                **{field: "" for field in ANNOTATION_FIELDS},
                "protocol_version": protocol_version,
                "protocol_sha256": protocol_sha256,
                "codebook_version": codebook_version,
                "codebook_sha256": codebook_sha256,
                "blinded_to_model_outputs": True,
                "street_media_sha256": public["street_media_sha256"],
                "overhead_media_sha256": public["overhead_media_sha256"],
            }
        )
        return template.loc[:, list(RATER_TEMPLATE_COLUMNS)]

    return {
        "public_packet": public,
        "private_key": private,
        "rater_a_template": rater_template(RATER_A_PLACEHOLDER),
        "rater_b_template": rater_template(RATER_B_PLACEHOLDER),
        "selected_manifest": manifest_index.loc[selected["pair_id"]].reset_index(),
        "codebook_payload": codebook_payload,
    }


def remap_completed_rater_templates(
    *,
    rater_a_path: Path,
    rater_b_path: Path,
    public_packet_path: Path,
    private_key_path: Path,
    packet_metadata_path: Path,
    protocol_path: Path,
    codebook_path: Path | None = None,
    blinding_secret_path: Path | None = None,
) -> pd.DataFrame:
    """Validate completed blinded CSVs against the real packet and protocol.

    The templates deliberately use the core field name ``pair_id`` for their
    opaque ids. This is the only supported transition to source pair ids.
    Every identity, ordering, codebook, CSV hash, derivative lineage, and
    protocol commitment is independently recomputed before remapping.
    """

    paths = {
        "rater_a": Path(rater_a_path),
        "rater_b": Path(rater_b_path),
        "public_packet": Path(public_packet_path),
        "private_key": Path(private_key_path),
        "packet_metadata": Path(packet_metadata_path),
        "protocol": Path(protocol_path),
    }
    codebook_path = (
        Path(codebook_path)
        if codebook_path is not None
        else paths["public_packet"].parent / "annotation_codebook.json"
    )
    paths["codebook"] = codebook_path
    blinding_secret_path = (
        Path(blinding_secret_path)
        if blinding_secret_path is not None
        else paths["private_key"].parent / "blinding_secret.key"
    )
    paths["blinding_secret"] = blinding_secret_path
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if paths["public_packet"].parent.resolve() == paths["private_key"].parent.resolve():
        raise ValueError("Public packet and restricted private key must be separated")

    protocol = load_protocol(paths["protocol"])
    protocol_version, _expected_seeds = _validate_protocol(protocol)
    protocol_sha256 = sha256_file(paths["protocol"])
    codebook_payload, codebook_bytes, codebook_sha256 = _codebook_artifact(protocol)
    codebook_version = str(codebook_payload["codebook_version"])
    if paths["codebook"].read_bytes() != codebook_bytes:
        raise ValueError("Staged codebook does not exactly match the real protocol config")

    metadata = _read_json_object(paths["packet_metadata"], "packet metadata")
    if metadata.get("schema_version") != PACKET_SCHEMA:
        raise ValueError("Packet metadata schema mismatch")
    if metadata.get("development_pilot_only") is not True:
        raise ValueError("Packet metadata is not development_pilot_only")
    if metadata.get("protocol_version") != protocol_version:
        raise ValueError("Packet metadata protocol_version does not match real config")
    if metadata.get("protocol_sha256") != protocol_sha256:
        raise ValueError("Packet metadata protocol hash does not match real config")
    if metadata.get("codebook_version") != codebook_version:
        raise ValueError("Packet metadata codebook_version mismatch")
    if metadata.get("codebook_sha256") != codebook_sha256:
        raise ValueError("Packet metadata codebook hash mismatch")
    if metadata.get("media_derivative_rule") != MEDIA_DERIVATIVE_RULE:
        raise ValueError("Packet metadata derivative rule changed")
    config_input = metadata.get("input_files", {}).get("protocol_config", {})
    if config_input.get("sha256") != protocol_sha256:
        raise ValueError("Packet input commitment does not bind the real protocol config")
    output_hashes = metadata.get("output_sha256")
    if not isinstance(output_hashes, dict):
        raise ValueError("Packet metadata is missing output SHA-256 commitments")
    for name, path in (
        ("public/public_packet.csv", paths["public_packet"]),
        ("restricted/private_key.csv", paths["private_key"]),
        ("public/annotation_codebook.json", paths["codebook"]),
        ("restricted/blinding_secret.key", paths["blinding_secret"]),
    ):
        expected = output_hashes.get(name)
        if expected != sha256_file(path):
            raise ValueError(f"Artifact hash does not match packet metadata: {name}")
    blinding_secret = _validated_blinding_secret(paths["blinding_secret"].read_bytes())
    secret_commitment = hashlib.sha256(blinding_secret).hexdigest()
    if metadata.get("blinding_secret_sha256") != secret_commitment:
        raise ValueError("Restricted blinding-secret commitment does not match")
    if metadata.get("opaque_id_method") != "HMAC-SHA256/restricted-256-bit-secret":
        raise ValueError("Packet opaque-id method is not the frozen keyed method")

    public_packet = pd.read_csv(
        paths["public_packet"], dtype=str, keep_default_na=False
    )
    private_key = pd.read_csv(paths["private_key"], dtype=str, keep_default_na=False)
    rater_a = pd.read_csv(paths["rater_a"], dtype=str, keep_default_na=False)
    rater_b = pd.read_csv(paths["rater_b"], dtype=str, keep_default_na=False)
    if tuple(public_packet.columns) != PUBLIC_PACKET_COLUMNS:
        raise ValueError(
            "Public packet columns changed; refusing a potentially unblinded import"
        )
    if public_packet.empty:
        raise ValueError("Public packet cannot be empty")
    if public_packet["opaque_item_id"].duplicated().any():
        raise ValueError("Public packet opaque_item_id must be unique")
    public = public_packet.copy()
    public["opaque_item_id"] = _identifiers(
        public["opaque_item_id"], "public opaque_item_id"
    )
    for column in ("street_media_sha256", "overhead_media_sha256"):
        public[column] = _hashes(public[column], column)

    if tuple(private_key.columns) != PRIVATE_KEY_COLUMNS:
        raise ValueError("Restricted private-key columns or order changed")
    key = private_key.copy()
    key["opaque_item_id"] = _identifiers(key["opaque_item_id"], "key opaque_item_id")
    key["pair_id"] = _identifiers(key["pair_id"], "key pair_id")
    if key["opaque_item_id"].duplicated().any() or key["pair_id"].duplicated().any():
        raise ValueError("Private key must be one-to-one by opaque_item_id and pair_id")
    expected_opaque = set(public["opaque_item_id"])
    if set(key["opaque_item_id"]) != expected_opaque:
        raise ValueError("Private key/public packet opaque item sets must match exactly")
    lineage_hash_columns = [
        column
        for view in ("street", "overhead")
        for column in (
            f"source_{view}_media_sha256",
            f"derivative_{view}_media_sha256",
            f"{view}_decoded_rgb_sha256",
            f"{view}_entropy_or_idat_sha256",
        )
    ]
    for column in lineage_hash_columns:
        key[column] = _hashes(key[column], f"key {column}")
    if not key["media_derivative_rule"].eq(MEDIA_DERIVATIVE_RULE).all():
        raise ValueError("Private key media derivative rule changed")
    for column, expected_value in (
        ("protocol_version", protocol_version),
        ("protocol_sha256", protocol_sha256),
        ("codebook_version", codebook_version),
        ("codebook_sha256", codebook_sha256),
    ):
        if not key[column].eq(expected_value).all():
            raise ValueError(f"Private key {column} does not match the real protocol")
    roles = _identifiers(key["protocol_role"], "key protocol_role")
    if not roles.eq(PROTOCOL_ROLE).all():
        raise ValueError("Private key contains a non-development protocol role")
    if not key["selection_rule"].astype(str).eq(FROZEN_SELECTION_RULE).all():
        raise ValueError("Private key selection rule changed")
    selected = key["selected_for_ensemble_disagreement"].map(
        lambda value: isinstance(value, (bool, np.bool_))
        and bool(value)
        or isinstance(value, str)
        and value.strip().casefold() == "true"
    )
    if not selected.all():
        raise ValueError("Every imported item must attest ensemble disagreement selection")
    positions = pd.to_numeric(key["packet_position"], errors="coerce")
    if positions.isna().any() or not np.equal(positions, np.floor(positions)).all():
        raise ValueError("Private key packet_position must contain integers")
    key["packet_position"] = positions.astype(int)
    if key["packet_position"].tolist() != list(range(1, len(key) + 1)):
        raise ValueError("Private key packet positions/order are not canonical")
    recomputed = key["pair_id"].map(
        lambda pair_id: opaque_item_id(blinding_secret, protocol_sha256, pair_id)
    )
    if not key["opaque_item_id"].eq(recomputed).all():
        raise ValueError("Private key opaque mapping does not match the real protocol hash")
    recomputed_order = key["pair_id"].map(
        lambda pair_id: packet_order_sha256(
            blinding_secret, protocol_sha256, pair_id
        )
    )
    if not key["packet_order_sha256"].eq(recomputed_order).all():
        raise ValueError("Private key packet ordering hashes do not recompute")
    expected_key_order = key.sort_values(
        ["packet_order_sha256", "opaque_item_id"]
    )["opaque_item_id"].tolist()
    if key["opaque_item_id"].tolist() != expected_key_order:
        raise ValueError("Private key row order is not the deterministic packet order")

    public_index = public.set_index("opaque_item_id")
    key_index = key.set_index("opaque_item_id")
    media_digest = hashlib.sha256()
    for opaque_id in public["opaque_item_id"]:
        public_row = public_index.loc[opaque_id]
        key_row = key_index.loc[opaque_id]
        for view in ("street", "overhead"):
            public_hash_column = f"{view}_media_sha256"
            derivative_key_column = f"derivative_{view}_media_sha256"
            if public_row[public_hash_column] != key_row[derivative_key_column]:
                raise ValueError(f"Public/private derivative {view} hash mismatch")
            image_format = key_row[f"{view}_format"]
            suffix = ".jpg" if image_format == "JPEG" else ".png" if image_format == "PNG" else None
            if suffix is None:
                raise ValueError("Private key contains an unsupported derivative format")
            expected_relative = (
                Path("public_media") / f"{opaque_id}_{view}{suffix}"
            ).as_posix()
            relative_value = public_row[f"{view}_image_path"]
            if relative_value != expected_relative:
                raise ValueError("Public derivative path is not the canonical opaque path")
            media_digest.update(
                f"{relative_value}\0{public_row[public_hash_column]}\n".encode("utf-8")
            )
            derivative_path = (paths["public_packet"].parent / relative_value).resolve()
            if not derivative_path.is_relative_to(paths["public_packet"].parent.resolve()):
                raise ValueError("Public derivative path escapes the public packet directory")
            derivative, lineage = metadata_stripped_derivative(derivative_path)
            if derivative != derivative_path.read_bytes():
                raise ValueError("Public media still contains removable metadata")
            expected_lineage = {
                "derivative_media_sha256": key_row[derivative_key_column],
                "decoded_rgb_sha256": key_row[f"{view}_decoded_rgb_sha256"],
                "width": int(key_row[f"{view}_width"]),
                "height": int(key_row[f"{view}_height"]),
                "format": image_format,
                "entropy_or_idat_sha256": key_row[
                    f"{view}_entropy_or_idat_sha256"
                ],
            }
            for lineage_name, expected_value in expected_lineage.items():
                if lineage[lineage_name] != expected_value:
                    raise ValueError(
                        f"Public {view} derivative lineage mismatch: {lineage_name}"
                    )
    if metadata.get("public_media_ledger_digest_sha256") != media_digest.hexdigest():
        raise ValueError("Public media ledger digest does not match packet metadata")
    counts = metadata.get("output_counts", {})
    if (
        counts.get("selected_disagreement_pairs") != len(public)
        or counts.get("public_media_files") != 2 * len(public)
    ):
        raise ValueError("Packet metadata output counts do not match the public packet")

    expected_template_columns = set(RATER_TEMPLATE_COLUMNS)
    completed: list[pd.DataFrame] = []
    rater_ids: list[str] = []
    public_hashes = public.set_index("opaque_item_id")
    for name, template in (("rater_a", rater_a), ("rater_b", rater_b)):
        if set(template.columns) != expected_template_columns:
            missing = sorted(expected_template_columns - set(template.columns))
            extra = sorted(set(template.columns) - expected_template_columns)
            raise ValueError(
                f"{name} template columns changed; missing={missing}, extra={extra}"
            )
        frame = template.copy()
        frame["pair_id"] = _identifiers(frame["pair_id"], f"{name} opaque pair_id")
        if frame["pair_id"].duplicated().any() or set(frame["pair_id"]) != expected_opaque:
            raise ValueError(f"{name} opaque item set must match the public packet exactly")
        ids = _identifiers(frame["rater_id"], f"{name} rater_id")
        if ids.nunique() != 1:
            raise ValueError(f"{name} must contain exactly one rater_id")
        rater_id = ids.iat[0]
        if rater_id in {RATER_A_PLACEHOLDER, RATER_B_PLACEHOLDER}:
            raise ValueError(f"{name} rater_id placeholder must be replaced")
        frame["rater_id"] = ids
        rater_ids.append(rater_id)
        for column, expected_value in (
            ("protocol_version", protocol_version),
            ("protocol_sha256", protocol_sha256),
            ("codebook_version", codebook_version),
            ("codebook_sha256", codebook_sha256),
        ):
            if not frame[column].astype(str).eq(expected_value).all():
                raise ValueError(f"{name} {column} does not match the real protocol")
        aligned_hashes = public_hashes.loc[frame["pair_id"]]
        for column in ("street_media_sha256", "overhead_media_sha256"):
            template_hashes = _hashes(frame[column], f"{name} {column}")
            expected_hashes = aligned_hashes[column].reset_index(drop=True)
            if not template_hashes.reset_index(drop=True).eq(expected_hashes).all():
                raise ValueError(f"{name} {column} does not match the public packet")
            frame[column] = template_hashes
        completed.append(frame)
    if rater_ids[0] == rater_ids[1]:
        raise ValueError("The two completed templates must use distinct rater ids")

    opaque_annotations = pd.concat(completed, ignore_index=True)
    mapping = key.set_index("opaque_item_id")["pair_id"]
    opaque_annotations["pair_id"] = opaque_annotations["pair_id"].map(mapping)
    if opaque_annotations["pair_id"].isna().any():
        raise RuntimeError("Private-key remapping unexpectedly lost an annotation row")
    return validate_annotation_packet(
        opaque_annotations,
        expected_protocol_version=protocol_version,
        expected_protocol_sha256=protocol_sha256,
    )


def _write_csv_exclusive(path: Path, frame: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("xb") as handle:
        handle.write(_canonical_json_bytes(payload))


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    if output_dir.exists():
        entries = {path.name for path in output_dir.iterdir()}
        if entries and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing packet directory {output_dir}; "
                "pass --overwrite"
            )
        unknown = entries - ROOT_OUTPUT_NAMES
        if overwrite and unknown:
            raise ValueError(
                "Refusing --overwrite because the output directory contains "
                f"unrecognized entries: {sorted(unknown)}"
            )
        if overwrite:
            public_dir = output_dir / "public"
            restricted_dir = output_dir / "restricted"
            for directory, allowed in (
                (public_dir, PUBLIC_OUTPUT_NAMES),
                (restricted_dir, RESTRICTED_OUTPUT_NAMES),
            ):
                if directory.is_symlink():
                    raise ValueError(f"Refusing to overwrite symlink {directory}")
                if directory.is_dir():
                    nested = {path.name for path in directory.iterdir()}
                    nested_unknown = nested - allowed
                    if nested_unknown:
                        raise ValueError(
                            f"Refusing --overwrite due to unknown entries in {directory}: "
                            f"{sorted(nested_unknown)}"
                        )
            for name in ROOT_OUTPUT_NAMES:
                path = output_dir / name
                if path.is_symlink():
                    raise ValueError(f"Refusing to overwrite symlink {path}")
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "public").mkdir(exist_ok=False)
    (output_dir / "restricted").mkdir(exist_ok=False)


def write_annotation_packet(
    *,
    predictions_path: Path,
    manifest_path: Path,
    media_ledger_path: Path,
    protocol_path: Path,
    prediction_metadata_path: Path | None = None,
    protocol_summary_path: Path | None = None,
    output_dir: Path,
    overwrite: bool = False,
    blinding_secret: bytes | None = None,
) -> dict[str, object]:
    """Validate, blind, stage media, and write the complete annotation package."""

    predictions_path = Path(predictions_path)
    manifest_path = Path(manifest_path)
    prediction_metadata_path = (
        Path(prediction_metadata_path)
        if prediction_metadata_path is not None
        else predictions_path.parent / "prediction_metadata.json"
    )
    protocol_summary_path = (
        Path(protocol_summary_path)
        if protocol_summary_path is not None
        else manifest_path.parent / "protocol_summary.json"
    )
    input_paths = {
        "predictions": Path(predictions_path),
        "prediction_metadata": prediction_metadata_path,
        "study_development_manifest": Path(manifest_path),
        "media_hash_ledger": Path(media_ledger_path),
        "protocol_config": Path(protocol_path),
        "protocol_summary": protocol_summary_path,
    }
    for path in input_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol_sha256 = sha256_file(input_paths["protocol_config"])
    protocol = load_protocol(input_paths["protocol_config"])
    blinding_secret = _validated_blinding_secret(
        blinding_secret if blinding_secret is not None else secrets.token_bytes(32)
    )
    _prediction_metadata, protocol_summary = _validate_frozen_sidecars(
        prediction_metadata_path=input_paths["prediction_metadata"],
        protocol_summary_path=input_paths["protocol_summary"],
        predictions_path=input_paths["predictions"],
        manifest_path=input_paths["study_development_manifest"],
        media_ledger_path=input_paths["media_hash_ledger"],
        protocol_path=input_paths["protocol_config"],
        protocol=protocol,
    )
    predictions = pd.read_csv(input_paths["predictions"], dtype={"pair_id": str})
    manifest = pd.read_csv(
        input_paths["study_development_manifest"],
        dtype={"pair_id": str, "spatial_block_id": str},
    )
    ledger = pd.read_csv(input_paths["media_hash_ledger"], dtype={"pair_id": str})
    expected_development_rows = protocol_summary.get("role_rows", {}).get(PROTOCOL_ROLE)
    if expected_development_rows != len(manifest):
        raise ValueError(
            "Development manifest row count does not match the frozen protocol summary"
        )
    tables = build_packet_tables(
        predictions=predictions,
        manifest=manifest,
        media_ledger=ledger,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        blinding_secret=blinding_secret,
    )

    public = tables["public_packet"]
    selected_manifest = tables["selected_manifest"].set_index("pair_id")
    private = tables["private_key"]
    source_by_opaque = private.set_index("opaque_item_id")["pair_id"]
    private_by_opaque = private.set_index("opaque_item_id")

    _prepare_output_dir(Path(output_dir), overwrite)
    output_dir = Path(output_dir)
    public_dir = output_dir / "public"
    restricted_dir = output_dir / "restricted"
    blinding_secret_path = restricted_dir / "blinding_secret.key"
    with blinding_secret_path.open("xb") as handle:
        handle.write(blinding_secret)
    public_media_dir = public_dir / "public_media"
    public_media_dir.mkdir(exist_ok=False)
    _write_json_exclusive(
        public_dir / "annotation_codebook.json", tables["codebook_payload"]
    )
    for row in public.itertuples(index=False):
        pair_id = source_by_opaque.loc[row.opaque_item_id]
        media = selected_manifest.loc[pair_id]
        private_row = private_by_opaque.loc[row.opaque_item_id]
        for view, source_column, destination_value, expected_hash in (
            (
                "street",
                "street_view_path",
                row.street_image_path,
                row.street_media_sha256,
            ),
            (
                "overhead",
                "remote_sensing_path",
                row.overhead_image_path,
                row.overhead_media_sha256,
            ),
        ):
            source = Path(str(media[source_column]))
            derivative, lineage = metadata_stripped_derivative(source)
            destination = public_dir / Path(destination_value)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output_handle:
                output_handle.write(derivative)
            if sha256_file(destination) != expected_hash:
                raise RuntimeError("Staged derivative hash does not match the public packet")
            expected_lineage = {
                "source_media_sha256": private_row[f"source_{view}_media_sha256"],
                "derivative_media_sha256": private_row[
                    f"derivative_{view}_media_sha256"
                ],
                "decoded_rgb_sha256": private_row[f"{view}_decoded_rgb_sha256"],
                "width": int(private_row[f"{view}_width"]),
                "height": int(private_row[f"{view}_height"]),
                "format": private_row[f"{view}_format"],
                "entropy_or_idat_sha256": private_row[
                    f"{view}_entropy_or_idat_sha256"
                ],
            }
            for name, expected_value in expected_lineage.items():
                if lineage[name] != expected_value:
                    raise RuntimeError(f"Staged derivative lineage changed: {view}/{name}")

    public_csv_outputs = {
        "public/public_packet.csv": public,
        "public/rater_a_template.csv": tables["rater_a_template"],
        "public/rater_b_template.csv": tables["rater_b_template"],
    }
    restricted_csv_outputs = {"restricted/private_key.csv": private}
    csv_outputs = {**public_csv_outputs, **restricted_csv_outputs}
    for filename, frame in csv_outputs.items():
        _write_csv_exclusive(output_dir / filename, frame)

    # A public handoff may not contain a source pair id in names, CSVs, codebook,
    # or staged image container bytes. This also catches residual textual metadata.
    source_ids = tuple(value.encode("utf-8") for value in private["pair_id"])
    secret_representations = (blinding_secret, blinding_secret.hex().encode("ascii"))
    for public_path in public_dir.rglob("*"):
        if not public_path.is_file():
            continue
        content = public_path.read_bytes()
        if any(source_id in content for source_id in source_ids):
            raise RuntimeError(
                f"Public handoff leaks a source pair id through {public_path.name}"
            )
        if any(secret in content for secret in secret_representations):
            raise RuntimeError(
                f"Public handoff leaks the blinding secret through {public_path.name}"
            )

    media_digest = hashlib.sha256()
    for row in public.itertuples(index=False):
        for relative_path, digest in (
            (row.street_image_path, row.street_media_sha256),
            (row.overhead_image_path, row.overhead_media_sha256),
        ):
            media_digest.update(f"{relative_path}\0{digest}\n".encode("utf-8"))
    protocol_version, expected_seeds = _validate_protocol(protocol)
    codebook_payload, _codebook_bytes, codebook_sha256 = _codebook_artifact(protocol)
    metadata: dict[str, object] = {
        "schema_version": PACKET_SCHEMA,
        "development_pilot_only": True,
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
        "codebook_version": codebook_payload["codebook_version"],
        "codebook_sha256": codebook_sha256,
        "protocol_role": PROTOCOL_ROLE,
        "selection": {
            "population": "all_ensemble_disagreements",
            "selection_rule": FROZEN_SELECTION_RULE,
            "uses_signed_direction": False,
            "ensemble_rule": protocol["models"]["ensemble_rule"],
            "seeds": list(expected_seeds),
        },
        "input_files": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in input_paths.items()
        },
        "input_counts": {
            "prediction_rows": int(len(predictions)),
            "prediction_seeds": int(predictions["seed"].nunique()),
            "manifest_pairs": int(len(manifest)),
            "media_ledger_pairs": int(len(ledger)),
        },
        "output_counts": {
            "selected_disagreement_pairs": int(len(public)),
            "public_media_files": int(2 * len(public)),
            "rater_templates": 2,
        },
        "output_sha256": {
            filename: sha256_file(output_dir / filename) for filename in csv_outputs
        },
        "public_media_ledger_digest_sha256": media_digest.hexdigest(),
        "public_packet_columns": list(PUBLIC_PACKET_COLUMNS),
        "rater_template_pair_id_semantics": "opaque_item_id_not_source_pair_id",
        "rater_id_placeholders": [RATER_A_PLACEHOLDER, RATER_B_PLACEHOLDER],
        "public_codebook": "public/annotation_codebook.json",
        "media_derivative_rule": MEDIA_DERIVATIVE_RULE,
        "media_lineage_location": "restricted/private_key.csv",
        "private_key_access": "restricted_directory_separate_from_public_handoff",
        "opaque_id_method": "HMAC-SHA256/restricted-256-bit-secret",
        "blinding_secret_sha256": hashlib.sha256(blinding_secret).hexdigest(),
        "blinded_to_model_outputs": True,
    }
    metadata["output_sha256"]["public/annotation_codebook.json"] = sha256_file(
        public_dir / "annotation_codebook.json"
    )
    metadata["output_sha256"]["restricted/blinding_secret.key"] = sha256_file(
        blinding_secret_path
    )
    _write_json_exclusive(restricted_dir / "packet_metadata.json", metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument(
        "--prediction-metadata",
        type=Path,
        help="Defaults to prediction_metadata.json beside --predictions-csv",
    )
    parser.add_argument(
        "--study-development-manifest",
        type=Path,
        default=(
            REPO_ROOT
            / "data"
            / "splits"
            / "eaton_component_direction_v1"
            / "study_development.csv"
        ),
    )
    parser.add_argument(
        "--media-hash-ledger",
        type=Path,
        default=(
            REPO_ROOT
            / "data"
            / "splits"
            / "eaton_component_direction_v1"
            / "media_hash_ledger.csv"
        ),
    )
    parser.add_argument(
        "--protocol-summary",
        type=Path,
        help="Defaults to protocol_summary.json beside the development manifest",
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=REPO_ROOT / "configs" / "eaton_component_direction_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs"
            / "eaton_component_direction_v1"
            / "development_annotation_packet"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    metadata = write_annotation_packet(
        predictions_path=args.predictions_csv,
        manifest_path=args.study_development_manifest,
        media_ledger_path=args.media_hash_ledger,
        protocol_path=args.protocol_config,
        prediction_metadata_path=args.prediction_metadata,
        protocol_summary_path=args.protocol_summary,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
