#!/usr/bin/env python
"""Validate blinded Eaton ratings and prepare or consume third-rater adjudication.

This command deliberately stops at annotation validation.  It never reads model
predictions, never runs H-B analyses, and never emits a GO/NO-GO decision.
Source identifiers, raw ratings, reliability, adjudications, and final references
are restricted artifacts.  The only public artifact this command can create is a
blinded third-rater template keyed by packet opaque identifiers.  Never fill a
template in place under a public handoff directory: first copy it to a restricted
working directory, then pass that completed copy to this command.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.analysis.eaton_component_direction import (
    ADJUDICATION_COLUMNS,
    ANNOTATION_FIELDS,
    build_adjudicated_reference,
    compute_annotation_reliability,
)
from scripts.build_eaton_component_annotation_packet import (
    PRIVATE_KEY_COLUMNS,
    PUBLIC_PACKET_COLUMNS,
    opaque_item_id,
    packet_order_sha256,
    remap_completed_rater_templates,
    sha256_file,
)
from scripts.build_eaton_component_direction_protocol import load_protocol


VALIDATION_SCHEMA = "eaton-component-annotation-validation-v1"
ADJUDICATOR_PLACEHOLDER = "REPLACE_WITH_DISTINCT_THIRD_RATER_ID"
ADJUDICATION_TEMPLATE_COLUMNS = (
    "opaque_item_id",
    "adjudicator_id",
    *ANNOTATION_FIELDS,
    "protocol_version",
    "protocol_sha256",
    "codebook_version",
    "codebook_sha256",
    "blinded_to_model_outputs",
    "street_media_sha256",
    "overhead_media_sha256",
    "adjudication_note",
)
PUBLIC_OUTPUT_NAMES = {"third_rater_adjudication_template.csv"}
RESTRICTED_OUTPUT_NAMES = {
    "remapped_raw_annotations.csv",
    "annotation_reliability.csv",
    "adjudications_true_id.csv",
    "component_reference.csv",
    "annotation_validation_metadata.json",
    "annotation_validation_metadata.sha256",
}
_PUBLIC_FORBIDDEN_MARKERS = (
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


def _write_csv_exclusive(path: Path, frame: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("xb") as handle:
        handle.write(_canonical_json_bytes(payload))


def _normalized_identifiers(values: pd.Series, name: str) -> pd.Series:
    result = values.astype("string").str.strip()
    if result.isna().any() or result.eq("").any():
        raise ValueError(f"{name} cannot be null or blank")
    return result


def _disputed_pair_ids(raw_annotations: pd.DataFrame) -> set[str]:
    disputed: set[str] = set()
    for pair_id, group in raw_annotations.groupby("pair_id", sort=False):
        if any(group[field].nunique(dropna=False) != 1 for field in ANNOTATION_FIELDS):
            disputed.add(str(pair_id))
    return disputed


def _load_verified_private_key(
    *,
    private_key_path: Path,
    packet_metadata_path: Path,
    blinding_secret_path: Path,
    protocol_sha256: str,
    raw_annotations: pd.DataFrame,
) -> pd.DataFrame:
    """Bind one in-memory key snapshot to metadata and the restricted HMAC key."""

    metadata = json.loads(packet_metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(
        metadata.get("output_sha256"), dict
    ):
        raise ValueError("Packet metadata is missing output SHA-256 commitments")
    key_bytes = private_key_path.read_bytes()
    secret = blinding_secret_path.read_bytes()
    if len(secret) != 32:
        raise ValueError("Restricted blinding secret must contain exactly 32 bytes")
    if hashlib.sha256(key_bytes).hexdigest() != metadata["output_sha256"].get(
        "restricted/private_key.csv"
    ):
        raise ValueError("Restricted private-key snapshot does not match packet metadata")
    if hashlib.sha256(secret).hexdigest() != metadata["output_sha256"].get(
        "restricted/blinding_secret.key"
    ):
        raise ValueError("Restricted blinding-secret snapshot does not match metadata")
    if metadata.get("blinding_secret_sha256") != hashlib.sha256(secret).hexdigest():
        raise ValueError("Restricted blinding-secret commitment changed")
    key = pd.read_csv(BytesIO(key_bytes), dtype=str, keep_default_na=False)
    if tuple(key.columns) != PRIVATE_KEY_COLUMNS:
        raise ValueError("Restricted private-key columns or order changed")
    key = key.copy()
    key["opaque_item_id"] = _normalized_identifiers(
        key["opaque_item_id"], "private-key opaque_item_id"
    )
    key["pair_id"] = _normalized_identifiers(key["pair_id"], "private-key pair_id")
    if key["opaque_item_id"].duplicated().any() or key["pair_id"].duplicated().any():
        raise ValueError("Restricted private key must remain one-to-one")
    recomputed_opaque = key["pair_id"].map(
        lambda pair_id: opaque_item_id(secret, protocol_sha256, pair_id)
    )
    if not key["opaque_item_id"].eq(recomputed_opaque).all():
        raise ValueError("Restricted private-key opaque mapping does not recompute")
    recomputed_order = key["pair_id"].map(
        lambda pair_id: packet_order_sha256(secret, protocol_sha256, pair_id)
    )
    if not key["packet_order_sha256"].eq(recomputed_order).all():
        raise ValueError("Restricted private-key ordering commitment does not recompute")
    raw_ids = set(raw_annotations["pair_id"].astype(str))
    key_ids = set(key["pair_id"].astype(str))
    if raw_ids != key_ids:
        raise ValueError(
            "Remapped annotation and verified private-key pair sets must match exactly"
        )
    return key


def _packet_identity(
    *, protocol_path: Path, codebook_path: Path
) -> dict[str, str]:
    protocol = load_protocol(protocol_path)
    codebook = json.loads(codebook_path.read_text(encoding="utf-8"))
    if not isinstance(codebook, dict):
        raise ValueError("Staged annotation codebook must be a JSON object")
    version = str(codebook.get("codebook_version", "")).strip()
    if not version:
        raise ValueError("Staged annotation codebook is missing codebook_version")
    protocol_version = str(protocol.get("protocol_version", "")).strip()
    if not protocol_version:
        raise ValueError("Frozen protocol is missing protocol_version")
    return {
        "protocol_version": protocol_version,
        "protocol_sha256": sha256_file(protocol_path),
        "codebook_version": version,
        "codebook_sha256": sha256_file(codebook_path),
    }


def build_third_rater_template(
    raw_annotations: pd.DataFrame,
    private_key: pd.DataFrame,
    identity: Mapping[str, str],
) -> pd.DataFrame:
    """Build a blank, opaque-only template for every and only disputed item."""

    disputed = _disputed_pair_ids(raw_annotations)
    if not disputed:
        return pd.DataFrame(columns=ADJUDICATION_TEMPLATE_COLUMNS)
    selected = private_key.loc[private_key["pair_id"].isin(disputed)].copy()
    if set(selected["pair_id"]) != disputed:
        raise ValueError("Verified private key does not cover every disputed item")
    positions = pd.to_numeric(selected["packet_position"], errors="coerce")
    if positions.isna().any():
        raise ValueError("Verified private key contains an invalid packet_position")
    selected = selected.assign(_packet_position=positions).sort_values(
        ["_packet_position", "opaque_item_id"]
    )

    raw_by_pair = raw_annotations.groupby("pair_id", sort=False)
    rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        source = raw_by_pair.get_group(str(row.pair_id))
        street_hash = str(row.derivative_street_media_sha256)
        overhead_hash = str(row.derivative_overhead_media_sha256)
        if (
            source["street_media_sha256"].nunique(dropna=False) != 1
            or str(source["street_media_sha256"].iat[0]) != street_hash
            or source["overhead_media_sha256"].nunique(dropna=False) != 1
            or str(source["overhead_media_sha256"].iat[0]) != overhead_hash
        ):
            raise ValueError("Disputed-item derivative hashes do not match raw ratings")
        rows.append(
            {
                "opaque_item_id": str(row.opaque_item_id),
                "adjudicator_id": ADJUDICATOR_PLACEHOLDER,
                **{field: "" for field in ANNOTATION_FIELDS},
                **identity,
                "blinded_to_model_outputs": True,
                "street_media_sha256": street_hash,
                "overhead_media_sha256": overhead_hash,
                "adjudication_note": "",
            }
        )
    result = pd.DataFrame(rows, columns=ADJUDICATION_TEMPLATE_COLUMNS)
    _audit_public_template(result, set(private_key["pair_id"].astype(str)))
    return result


def _audit_public_template(template: pd.DataFrame, source_pair_ids: set[str]) -> None:
    if tuple(template.columns) != ADJUDICATION_TEMPLATE_COLUMNS:
        raise ValueError("Third-rater template schema changed")
    forbidden = [
        column
        for column in template.columns
        if any(marker in column.casefold() for marker in _PUBLIC_FORBIDDEN_MARKERS)
    ]
    if forbidden:
        raise ValueError(f"Public adjudication template contains model fields: {forbidden}")
    values = {
        str(value)
        for column in template.columns
        for value in template[column].tolist()
        if str(value)
    }
    leaked = sorted(source_pair_ids & values)
    if leaked:
        raise ValueError("Public adjudication template leaked source pair identifiers")
    for field in ANNOTATION_FIELDS:
        if not template[field].eq("").all():
            raise ValueError("Public adjudication template leaked raw rater answers")


def _validate_completed_adjudication(
    *,
    completed_path: Path,
    expected_template: pd.DataFrame,
    raw_annotations: pd.DataFrame,
    private_key: pd.DataFrame,
) -> pd.DataFrame:
    completed = pd.read_csv(completed_path, dtype=str, keep_default_na=False)
    if tuple(completed.columns) != ADJUDICATION_TEMPLATE_COLUMNS:
        raise ValueError("Completed adjudication columns or order changed")
    if completed.empty:
        raise ValueError("Completed adjudication cannot be empty")
    completed = completed.copy()
    completed["opaque_item_id"] = _normalized_identifiers(
        completed["opaque_item_id"], "completed opaque_item_id"
    )
    if completed["opaque_item_id"].duplicated().any():
        raise ValueError("Completed adjudication must be unique by opaque_item_id")
    expected_ids = set(expected_template["opaque_item_id"].astype(str))
    supplied_ids = set(completed["opaque_item_id"].astype(str))
    if supplied_ids != expected_ids:
        raise ValueError(
            "Completed adjudication opaque item set must exactly equal the disputed set; "
            f"missing={sorted(expected_ids - supplied_ids)}, "
            f"unexpected={sorted(supplied_ids - expected_ids)}"
        )

    expected = expected_template.set_index("opaque_item_id")
    aligned = completed.set_index("opaque_item_id").loc[expected.index]
    for column in (
        "protocol_version",
        "protocol_sha256",
        "codebook_version",
        "codebook_sha256",
        "street_media_sha256",
        "overhead_media_sha256",
    ):
        if not aligned[column].eq(expected[column]).all():
            raise ValueError(f"Completed adjudication {column} identity mismatch")
    blinded = aligned["blinded_to_model_outputs"].astype(str).str.strip().str.casefold()
    if not blinded.eq("true").all():
        raise ValueError("Completed adjudication must attest blinded_to_model_outputs")

    adjudicator_ids = _normalized_identifiers(
        aligned["adjudicator_id"], "adjudicator_id"
    )
    if adjudicator_ids.nunique() != 1:
        raise ValueError("Completed adjudication must use exactly one adjudicator_id")
    adjudicator_id = str(adjudicator_ids.iat[0])
    rater_ids = set(raw_annotations["rater_id"].astype(str))
    if adjudicator_id == ADJUDICATOR_PLACEHOLDER or adjudicator_id in rater_ids:
        raise ValueError("Adjudicator must be distinct from both independent raters")

    opaque_to_source = private_key.set_index("opaque_item_id")["pair_id"]
    core = aligned.reset_index().rename(columns={"opaque_item_id": "pair_id"})
    core["pair_id"] = core["pair_id"].map(opaque_to_source)
    if core["pair_id"].isna().any():
        raise ValueError("Verified private key could not map a completed opaque item")
    core = core.loc[:, sorted(ADJUDICATION_COLUMNS)].copy()
    return core


def _input_inventory(
    *,
    public_packet_dir: Path,
    restricted_packet_dir: Path,
    rater_a_path: Path,
    rater_b_path: Path,
    protocol_path: Path,
    completed_adjudication_path: Path | None,
) -> list[dict[str, str]]:
    paths: list[tuple[str, Path]] = [
        ("public_packet", public_packet_dir / "public_packet.csv"),
        ("annotation_codebook", public_packet_dir / "annotation_codebook.json"),
        ("private_key", restricted_packet_dir / "private_key.csv"),
        ("blinding_secret", restricted_packet_dir / "blinding_secret.key"),
        ("packet_metadata", restricted_packet_dir / "packet_metadata.json"),
        ("completed_rater_a", rater_a_path),
        ("completed_rater_b", rater_b_path),
        ("protocol_config", protocol_path),
    ]
    if completed_adjudication_path is not None:
        paths.append(("completed_adjudication", completed_adjudication_path))
    public = pd.read_csv(
        public_packet_dir / "public_packet.csv", dtype=str, keep_default_na=False
    )
    if tuple(public.columns) != PUBLIC_PACKET_COLUMNS:
        raise ValueError("Public packet columns changed")
    for row in public.itertuples(index=False):
        paths.extend(
            [
                (
                    f"derivative_media:{row.opaque_item_id}:street",
                    public_packet_dir / str(row.street_image_path),
                ),
                (
                    f"derivative_media:{row.opaque_item_id}:overhead",
                    public_packet_dir / str(row.overhead_image_path),
                ),
            ]
        )
    inventory: list[dict[str, str]] = []
    for role, path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        inventory.append(
            {"role": role, "path": str(path.resolve()), "sha256": sha256_file(path)}
        )
    return inventory


def _audit_public_tree_has_no_blinding_secret(
    public_dir: Path, blinding_secret_path: Path
) -> None:
    secret = blinding_secret_path.read_bytes()
    if len(secret) != 32:
        raise ValueError("Restricted blinding secret must contain exactly 32 bytes")
    forbidden = (secret, secret.hex().encode("ascii"), secret.hex().upper().encode("ascii"))
    for path in public_dir.rglob("*"):
        relative_name = path.relative_to(public_dir).as_posix().encode("utf-8")
        if any(token in relative_name for token in forbidden):
            raise ValueError(
                f"Public tree path leaked the restricted blinding secret: {path}"
            )
        if path.is_symlink():
            raise ValueError(f"Public tree cannot contain a symlink: {path}")
        if not path.is_file():
            continue
        content = path.read_bytes()
        if any(token in content for token in forbidden):
            raise ValueError(f"Public tree leaked the restricted blinding secret: {path}")


def _prepare_output_root(output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    if output_dir.exists():
        root_entries = {path.name for path in output_dir.iterdir()}
        if root_entries and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing validation directory {output_dir}; "
                "pass --overwrite"
            )
        if root_entries - {"public", "restricted"}:
            raise ValueError("Validation output directory contains unrecognized entries")
        for dirname, allowed in (
            ("public", PUBLIC_OUTPUT_NAMES),
            ("restricted", RESTRICTED_OUTPUT_NAMES),
        ):
            directory = output_dir / dirname
            if not directory.exists():
                continue
            if not directory.is_dir() or directory.is_symlink():
                raise ValueError(f"Refusing unsafe output entry {directory}")
            entries = list(directory.iterdir())
            unknown = {path.name for path in entries} - allowed
            if unknown:
                raise ValueError(
                    f"Refusing --overwrite with unrecognized {dirname} artifacts: "
                    f"{sorted(unknown)}"
                )
            if overwrite:
                for path in entries:
                    if not path.is_file() or path.is_symlink():
                        raise ValueError(f"Refusing unsafe output artifact {path}")
                    path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    public_dir = output_dir / "public"
    restricted_dir = output_dir / "restricted"
    public_dir.mkdir(exist_ok=True)
    restricted_dir.mkdir(exist_ok=True)
    return public_dir, restricted_dir


def _paths_overlap(left: Path, right: Path) -> bool:
    return (
        left == right
        or left.is_relative_to(right)
        or right.is_relative_to(left)
    )


def _assert_completed_inputs_are_not_public_handoffs(
    *,
    completed_paths: list[Path],
    public_packet_dir: Path,
) -> None:
    """Reject completed answers stored in any recognizable public handoff tree."""

    explicit_public_paths = {
        Path(os.path.abspath(public_packet_dir)),
        public_packet_dir.resolve(strict=True),
    }
    for path in completed_paths:
        candidate_paths = {
            Path(os.path.abspath(path)),
            path.resolve(strict=True),
        }
        for candidate in candidate_paths:
            if any(
                candidate == public_path or candidate.is_relative_to(public_path)
                for public_path in explicit_public_paths
            ):
                raise ValueError(
                    "Completed annotation files cannot remain in a public handoff "
                    "directory; copy the template to a restricted working directory first"
                )
            for ancestor in (candidate, *candidate.parents):
                if ancestor.name.casefold() != "public":
                    continue
                restricted_sibling = ancestor.parent / "restricted"
                if restricted_sibling.is_dir():
                    raise ValueError(
                        "Completed annotation files cannot remain in a public handoff "
                        "directory; copy the template to a restricted working "
                        "directory first"
                    )


def validate_completed_annotations(
    *,
    public_packet_dir: Path,
    restricted_packet_dir: Path,
    rater_a_path: Path,
    rater_b_path: Path,
    protocol_path: Path,
    output_dir: Path,
    completed_adjudication_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run the fail-closed annotation handoff and return restricted metadata."""

    public_packet_dir = Path(public_packet_dir)
    restricted_packet_dir = Path(restricted_packet_dir)
    rater_a_path = Path(rater_a_path)
    rater_b_path = Path(rater_b_path)
    protocol_path = Path(protocol_path)
    output_dir = Path(output_dir)
    completed_adjudication_path = (
        Path(completed_adjudication_path)
        if completed_adjudication_path is not None
        else None
    )
    input_files = [
        public_packet_dir / "public_packet.csv",
        public_packet_dir / "annotation_codebook.json",
        restricted_packet_dir / "private_key.csv",
        restricted_packet_dir / "blinding_secret.key",
        restricted_packet_dir / "packet_metadata.json",
        rater_a_path,
        rater_b_path,
        protocol_path,
    ]
    if completed_adjudication_path is not None:
        input_files.append(completed_adjudication_path)
    completed_paths = [rater_a_path, rater_b_path]
    if completed_adjudication_path is not None:
        completed_paths.append(completed_adjudication_path)
    _assert_completed_inputs_are_not_public_handoffs(
        completed_paths=completed_paths,
        public_packet_dir=public_packet_dir,
    )
    if output_dir.is_symlink():
        raise ValueError("Validation output directory cannot be a symlink")
    resolved_output = output_dir.resolve(strict=False)
    resolved_public_packet = public_packet_dir.resolve(strict=True)
    resolved_restricted_packet = restricted_packet_dir.resolve(strict=True)
    for name, packet_dir in (
        ("public packet", resolved_public_packet),
        ("restricted packet", resolved_restricted_packet),
    ):
        if _paths_overlap(resolved_output, packet_dir):
            raise ValueError(
                f"Validation output directory cannot overlap the {name} directory"
            )
    if any(path.resolve().is_relative_to(resolved_output) for path in input_files):
        raise ValueError("Validation output directory cannot contain an input artifact")

    public_packet_path = public_packet_dir / "public_packet.csv"
    private_key_path = restricted_packet_dir / "private_key.csv"
    packet_metadata_path = restricted_packet_dir / "packet_metadata.json"
    codebook_path = public_packet_dir / "annotation_codebook.json"
    blinding_secret_path = restricted_packet_dir / "blinding_secret.key"

    # This is the sole supported opaque-to-source transition.  It revalidates the
    # real protocol, codebook, packet metadata, HMAC secret, key, media, and ratings.
    raw = remap_completed_rater_templates(
        rater_a_path=rater_a_path,
        rater_b_path=rater_b_path,
        public_packet_path=public_packet_path,
        private_key_path=private_key_path,
        packet_metadata_path=packet_metadata_path,
        protocol_path=protocol_path,
        codebook_path=codebook_path,
        blinding_secret_path=blinding_secret_path,
    )
    _audit_public_tree_has_no_blinding_secret(
        public_packet_dir, blinding_secret_path
    )
    reliability = compute_annotation_reliability(
        raw,
        expected_protocol_version=str(raw["protocol_version"].iat[0]),
        expected_protocol_sha256=str(raw["protocol_sha256"].iat[0]),
    )
    identity = _packet_identity(protocol_path=protocol_path, codebook_path=codebook_path)
    key = _load_verified_private_key(
        private_key_path=private_key_path,
        packet_metadata_path=packet_metadata_path,
        blinding_secret_path=blinding_secret_path,
        protocol_sha256=identity["protocol_sha256"],
        raw_annotations=raw,
    )
    template = build_third_rater_template(raw, key, identity)
    disputed_count = len(template)

    true_id_adjudications: pd.DataFrame | None = None
    reference: pd.DataFrame | None = None
    if disputed_count == 0:
        if completed_adjudication_path is not None:
            raise ValueError("Completed adjudication was supplied but no item is disputed")
        reference = build_adjudicated_reference(
            raw,
            expected_protocol_version=identity["protocol_version"],
            expected_protocol_sha256=identity["protocol_sha256"],
        )
        status = "REFERENCE_READY_NO_ADJUDICATION"
    elif completed_adjudication_path is None:
        status = "AWAITING_BLINDED_THIRD_RATER"
    else:
        true_id_adjudications = _validate_completed_adjudication(
            completed_path=completed_adjudication_path,
            expected_template=template,
            raw_annotations=raw,
            private_key=key,
        )
        reference = build_adjudicated_reference(
            raw,
            true_id_adjudications,
            expected_protocol_version=identity["protocol_version"],
            expected_protocol_sha256=identity["protocol_sha256"],
        )
        status = "REFERENCE_READY_AFTER_ADJUDICATION"

    inventory = _input_inventory(
        public_packet_dir=public_packet_dir,
        restricted_packet_dir=restricted_packet_dir,
        rater_a_path=rater_a_path,
        rater_b_path=rater_b_path,
        protocol_path=protocol_path,
        completed_adjudication_path=completed_adjudication_path,
    )
    public_dir, restricted_dir = _prepare_output_root(output_dir, overwrite)
    output_paths: list[Path] = []
    if status == "AWAITING_BLINDED_THIRD_RATER":
        path = public_dir / "third_rater_adjudication_template.csv"
        _write_csv_exclusive(path, template)
        output_paths.append(path)
        _audit_public_tree_has_no_blinding_secret(public_dir, blinding_secret_path)
    raw_path = restricted_dir / "remapped_raw_annotations.csv"
    reliability_path = restricted_dir / "annotation_reliability.csv"
    _write_csv_exclusive(raw_path, raw)
    _write_csv_exclusive(reliability_path, reliability)
    output_paths.extend([raw_path, reliability_path])
    if true_id_adjudications is not None:
        path = restricted_dir / "adjudications_true_id.csv"
        _write_csv_exclusive(path, true_id_adjudications)
        output_paths.append(path)
    if reference is not None:
        path = restricted_dir / "component_reference.csv"
        _write_csv_exclusive(path, reference)
        output_paths.append(path)

    metadata: dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMA,
        "status": status,
        "analysis_or_gate_executed": False,
        "protocol_version": identity["protocol_version"],
        "protocol_sha256": identity["protocol_sha256"],
        "codebook_version": identity["codebook_version"],
        "codebook_sha256": identity["codebook_sha256"],
        "pair_count": int(raw["pair_id"].nunique()),
        "raw_annotation_rows": int(len(raw)),
        "disputed_item_count": int(disputed_count),
        "reference_ready": reference is not None,
        "input_sha256": inventory,
        "output_sha256": {
            path.relative_to(output_dir).as_posix(): sha256_file(path)
            for path in output_paths
        },
        "metadata_hash_scope": (
            "all input files and all non-metadata output artifacts; the metadata "
            "file itself is committed by annotation_validation_metadata.sha256"
        ),
    }
    metadata_path = restricted_dir / "annotation_validation_metadata.json"
    _write_json_exclusive(metadata_path, metadata)
    metadata_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    with (restricted_dir / "annotation_validation_metadata.sha256").open(
        "x", encoding="ascii", newline=""
    ) as handle:
        handle.write(f"{metadata_hash}  annotation_validation_metadata.json\n")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Privacy rule: copy each blank public template into a restricted "
            "working directory before entering answers. Completed files stored "
            "inside any public handoff directory are rejected, including through "
            "symlinks."
        ),
    )
    parser.add_argument(
        "--public-packet-dir", "--public-dir", dest="public_packet_dir", type=Path,
        required=True,
    )
    parser.add_argument(
        "--restricted-packet-dir", "--restricted-dir",
        dest="restricted_packet_dir", type=Path, required=True,
    )
    parser.add_argument(
        "--completed-rater-a", "--rater-a", dest="rater_a", type=Path,
        required=True,
        help="Completed rater-A copy stored outside every public handoff directory",
    )
    parser.add_argument(
        "--completed-rater-b", "--rater-b", dest="rater_b", type=Path,
        required=True,
        help="Completed rater-B copy stored outside every public handoff directory",
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=REPO_ROOT / "configs" / "eaton_component_direction_v1.json",
    )
    parser.add_argument(
        "--completed-adjudication",
        type=Path,
        help=(
            "Completed third-rater copy stored in a restricted working directory; "
            "never fill the public handoff template in place"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    metadata = validate_completed_annotations(
        public_packet_dir=args.public_packet_dir,
        restricted_packet_dir=args.restricted_packet_dir,
        rater_a_path=args.rater_a,
        rater_b_path=args.rater_b,
        protocol_path=args.protocol_config,
        completed_adjudication_path=args.completed_adjudication,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
