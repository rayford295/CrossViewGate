"""Run the frozen, outcome-blind Eaton pre-annotation futility audit.

This entry point deliberately exposes no annotation input.  It binds the raw
bytes of the frozen protocol artifacts and fresh five-seed prediction export,
then delegates the scientific audit to ``pre_annotation_futility_gate``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import uuid
from typing import Any, Mapping

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.analysis.eaton_component_gate import (  # noqa: E402
    pre_annotation_futility_gate,
)


SCHEMA_VERSION = "eaton-component-pre-annotation-futility-run-v1"
COMMITMENT_SCHEMA = "eaton-component-direction-artifact-commitments-v1"
PREDICTION_COMMITMENT_SCHEMA = (
    "eaton-component-direction-development-run-commitments-v1"
)
PROTOCOL_ROLE = "study_development"

DEFAULT_PROTOCOL_CONFIG_PATH = (
    REPO_ROOT / "configs" / "eaton_component_direction_v1.json"
)
DEFAULT_ARTIFACT_COMMITMENTS_PATH = (
    REPO_ROOT / "configs" / "eaton_component_direction_v1_commitments.json"
)
DEFAULT_PREDICTION_COMMITMENTS_PATH = (
    REPO_ROOT
    / "configs"
    / "eaton_component_direction_v1_development_run_commitments.json"
)
DEFAULT_SPLIT_DIR = REPO_ROOT / "data" / "splits" / "eaton_component_direction_v1"
DEFAULT_PROTOCOL_SUMMARY_PATH = DEFAULT_SPLIT_DIR / "protocol_summary.json"
DEFAULT_DEVELOPMENT_MANIFEST_PATH = DEFAULT_SPLIT_DIR / "study_development.csv"
DEFAULT_MODEL_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "eaton_component_direction_v1"
    / "development_models_protocol_v1"
)
DEFAULT_PREDICTIONS_PATH = (
    DEFAULT_MODEL_OUTPUT_DIR / "study_development_predictions.csv"
)
DEFAULT_PREDICTION_METADATA_PATH = DEFAULT_MODEL_OUTPUT_DIR / "prediction_metadata.json"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "outputs"
    / "eaton_component_direction_v1"
    / "pre_annotation_futility_v1.json"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _expected_sha256(value: str, name: str) -> str:
    digest = str(value).strip().casefold()
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{name} must be a complete SHA-256 value")
    return digest


def _absolute_without_following(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _reject_symlink_components(path: Path, name: str) -> None:
    """Reject symlinked existing components, including a dangling final link."""

    absolute = _absolute_without_following(path)
    components = [absolute, *absolute.parents]
    for component in components:
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{name} cannot traverse a symlink: {component}")


def _read_raw_file(path: str | Path, name: str) -> tuple[Path, bytes]:
    absolute = _absolute_without_following(path)
    _reject_symlink_components(absolute, name)
    if not absolute.is_file():
        raise FileNotFoundError(absolute)
    try:
        raw = absolute.read_bytes()
    except OSError as error:
        raise ValueError(f"Cannot read {name} from {absolute}") from error
    return absolute.resolve(strict=True), raw


def _json_object(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must be UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _prediction_frame(raw: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(raw), dtype={"pair_id": str})
    except Exception as error:
        raise ValueError("prediction CSV must be a readable CSV") from error


def _validate_pre_run_commitments(
    commitments: Mapping[str, object],
    *,
    protocol_config_sha256: str,
    protocol_summary_sha256: str,
    development_manifest_sha256: str,
) -> None:
    if commitments.get("schema_version") != COMMITMENT_SCHEMA:
        raise ValueError("Artifact commitments have the wrong schema_version")
    if commitments.get("status") != "FROZEN_BEFORE_REGISTERED_DIRECTIONAL_ANALYSIS":
        raise ValueError("Artifact commitments are not in the frozen pre-analysis state")
    if commitments.get("protocol_config_sha256") != protocol_config_sha256:
        raise ValueError("Protocol config does not match the frozen commitments")
    if commitments.get("protocol_summary_sha256") != protocol_summary_sha256:
        raise ValueError("Protocol summary does not match the frozen commitments")
    role_hashes = commitments.get("role_manifest_sha256")
    if not isinstance(role_hashes, Mapping) or role_hashes.get(PROTOCOL_ROLE) != (
        development_manifest_sha256
    ):
        raise ValueError("Development manifest does not match the frozen commitments")
    if commitments.get("confirmation_status") != (
        "UNSCORED_RESERVED_SAME_EVENT_SPATIAL_CONFIRMATION"
    ):
        raise ValueError("Frozen commitments do not reserve untouched confirmation")


def _validate_prediction_commitments(
    commitments: Mapping[str, object],
    *,
    protocol_version: str,
    protocol_config_sha256: str,
    protocol_summary_sha256: str,
    prediction_csv_sha256: str,
    prediction_metadata_sha256: str,
    expected_protocol_summary_sha256: str,
    expected_prediction_csv_sha256: str,
    expected_prediction_metadata_sha256: str,
) -> None:
    """Bind immutable post-run hashes to both expected args and actual bytes."""

    if commitments.get("schema_version") != PREDICTION_COMMITMENT_SCHEMA:
        raise ValueError("Prediction commitments have the wrong schema_version")
    if commitments.get("protocol_version") != protocol_version:
        raise ValueError("Prediction commitments have the wrong protocol_version")
    if commitments.get("status") != (
        "FROZEN_BEFORE_PRE_ANNOTATION_FUTILITY_AUDIT"
    ):
        raise ValueError(
            "Prediction commitments are not frozen before the futility audit"
        )

    required = {
        "protocol_config_sha256": protocol_config_sha256,
        "protocol_summary_sha256": protocol_summary_sha256,
        "prediction_csv_sha256": prediction_csv_sha256,
        "prediction_metadata_sha256": prediction_metadata_sha256,
    }
    for field, actual in required.items():
        recorded = _expected_sha256(
            str(commitments.get(field, "")), f"prediction commitments {field}"
        )
        if recorded != actual:
            raise ValueError(
                f"Actual {field} bytes do not match the frozen prediction commitments"
            )

    expected = {
        "protocol_summary_sha256": expected_protocol_summary_sha256,
        "prediction_csv_sha256": expected_prediction_csv_sha256,
        "prediction_metadata_sha256": expected_prediction_metadata_sha256,
    }
    for field, trusted in expected.items():
        if commitments.get(field) != trusted:
            raise ValueError(
                f"{field} expected argument disagrees with prediction commitments"
            )


def _artifact_metadata(path: Path, raw: bytes) -> dict[str, object]:
    return {
        "absolute_path": str(path),
        "sha256": _sha256(raw),
        "byte_count": len(raw),
    }


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(_absolute_without_following(left))) == os.path.normcase(
        str(_absolute_without_following(right))
    )


def _serialized_json(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json_exclusive(
    output_path: str | Path,
    payload: Mapping[str, object],
    *,
    overwrite: bool,
) -> Path:
    """Write exclusively; explicit replacement is restricted to one known file."""

    output = _absolute_without_following(output_path)
    known = _absolute_without_following(DEFAULT_OUTPUT_PATH)
    _reject_symlink_components(output.parent, "output parent")
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output.parent, "output parent")
    if os.path.lexists(output) and output.is_symlink():
        raise ValueError(f"Output file cannot be a symlink: {output}")
    if overwrite and not _same_path(output, known):
        raise ValueError("--overwrite is restricted to the known futility output file")
    if os.path.lexists(output) and not output.is_file():
        raise ValueError(f"Output path is not a regular file: {output}")

    encoded = _serialized_json(payload)
    binary_flag = getattr(os, "O_BINARY", 0)
    if not overwrite:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | binary_flag
        try:
            descriptor = os.open(output, flags, 0o600)
        except FileExistsError as error:
            raise FileExistsError(
                f"Refusing to overwrite existing output {output}; pass --overwrite "
                "only for the known protocol target"
            ) from error
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        return output

    # Build the replacement with O_EXCL, then atomically replace only the known
    # regular target.  os.replace replaces a swapped link itself; it never
    # follows that link to another file.
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | binary_flag
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if os.path.lexists(output) and output.is_symlink():
            raise ValueError(f"Output file cannot be a symlink: {output}")
        if os.path.lexists(output) and not output.is_file():
            raise ValueError(f"Output path is not a regular file: {output}")
        os.replace(temporary, output)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()
    return output


def run_preannotation_futility_gate(
    *,
    protocol_config_path: str | Path = DEFAULT_PROTOCOL_CONFIG_PATH,
    artifact_commitments_path: str | Path = DEFAULT_ARTIFACT_COMMITMENTS_PATH,
    prediction_commitments_path: str | Path | None = None,
    protocol_summary_path: str | Path = DEFAULT_PROTOCOL_SUMMARY_PATH,
    development_manifest_path: str | Path = DEFAULT_DEVELOPMENT_MANIFEST_PATH,
    predictions_path: str | Path = DEFAULT_PREDICTIONS_PATH,
    prediction_metadata_path: str | Path = DEFAULT_PREDICTION_METADATA_PATH,
    expected_protocol_summary_sha256: str,
    expected_prediction_csv_sha256: str,
    expected_prediction_metadata_sha256: str,
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Validate all frozen bytes, run the signless audit, and write one wrapper."""

    expected_summary = _expected_sha256(
        expected_protocol_summary_sha256, "expected_protocol_summary_sha256"
    )
    expected_predictions = _expected_sha256(
        expected_prediction_csv_sha256, "expected_prediction_csv_sha256"
    )
    expected_prediction_metadata = _expected_sha256(
        expected_prediction_metadata_sha256,
        "expected_prediction_metadata_sha256",
    )

    input_specs = {
        "artifact_commitments": (artifact_commitments_path, "artifact commitments"),
        "prediction_commitments": (
            (
                DEFAULT_PREDICTION_COMMITMENTS_PATH
                if prediction_commitments_path is None
                else prediction_commitments_path
            ),
            "prediction commitments",
        ),
        "protocol_config": (protocol_config_path, "protocol config"),
        "protocol_summary": (protocol_summary_path, "protocol summary"),
        "study_development_manifest": (
            development_manifest_path,
            "study-development manifest",
        ),
        "prediction_csv": (predictions_path, "prediction CSV"),
        "prediction_metadata": (prediction_metadata_path, "prediction metadata"),
    }
    bound: dict[str, tuple[Path, bytes]] = {
        key: _read_raw_file(path, name)
        for key, (path, name) in input_specs.items()
    }
    raw = {key: item[1] for key, item in bound.items()}
    paths = {key: item[0] for key, item in bound.items()}

    actual_summary = _sha256(raw["protocol_summary"])
    actual_predictions = _sha256(raw["prediction_csv"])
    actual_prediction_metadata = _sha256(raw["prediction_metadata"])
    protocol = _json_object(raw["protocol_config"], "protocol config")
    commitments = _json_object(raw["artifact_commitments"], "artifact commitments")
    prediction_commitments = _json_object(
        raw["prediction_commitments"], "prediction commitments"
    )
    predictions = _prediction_frame(raw["prediction_csv"])
    protocol_hash = _sha256(raw["protocol_config"])
    _validate_pre_run_commitments(
        commitments,
        protocol_config_sha256=protocol_hash,
        protocol_summary_sha256=actual_summary,
        development_manifest_sha256=_sha256(raw["study_development_manifest"]),
    )
    protocol_version = str(protocol.get("protocol_version", "")).strip()
    if not protocol_version:
        raise ValueError("Protocol config protocol_version cannot be blank")
    _validate_prediction_commitments(
        prediction_commitments,
        protocol_version=protocol_version,
        protocol_config_sha256=protocol_hash,
        protocol_summary_sha256=actual_summary,
        prediction_csv_sha256=actual_predictions,
        prediction_metadata_sha256=actual_prediction_metadata,
        expected_protocol_summary_sha256=expected_summary,
        expected_prediction_csv_sha256=expected_predictions,
        expected_prediction_metadata_sha256=expected_prediction_metadata,
    )

    # Scientific logic lives exclusively in the core gate.  This wrapper does
    # not inspect prediction ordering and has no annotation inputs.
    futility_artifact = pre_annotation_futility_gate(
        predictions,
        protocol,
        expected_protocol_sha256=protocol_hash,
        expected_protocol_summary_sha256=expected_summary,
        expected_prediction_csv_sha256=expected_predictions,
        expected_prediction_metadata_sha256=expected_prediction_metadata,
        protocol_config_bytes=raw["protocol_config"],
        protocol_config_path=paths["protocol_config"],
        prediction_csv_bytes=raw["prediction_csv"],
        prediction_csv_path=paths["prediction_csv"],
        prediction_metadata_bytes=raw["prediction_metadata"],
        prediction_metadata_path=paths["prediction_metadata"],
        role_manifest_csv_bytes=raw["study_development_manifest"],
        role_manifest_path=paths["study_development_manifest"],
        protocol_summary_bytes=raw["protocol_summary"],
        protocol_summary_path=paths["protocol_summary"],
        expected_role=PROTOCOL_ROLE,
    )

    wrapper: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_status": "COMPLETE",
        "protocol_role": PROTOCOL_ROLE,
        "trusted_anchors": {
            "expected_protocol_summary_sha256": expected_summary,
            "expected_prediction_csv_sha256": expected_predictions,
            "expected_prediction_metadata_sha256": expected_prediction_metadata,
            "prediction_commitments_raw_sha256": _sha256(
                raw["prediction_commitments"]
            ),
        },
        "input_artifacts": {
            key: _artifact_metadata(paths[key], raw[key]) for key in input_specs
        },
        "core_futility_artifact_sha256": futility_artifact[
            "futility_artifact_sha256"
        ],
        # Keep the core artifact byte-for-byte-equivalent as a JSON object; do
        # not mix paths or operational metadata into its scientific digest.
        "futility_artifact": futility_artifact,
    }
    destination = DEFAULT_OUTPUT_PATH if output_path is None else Path(output_path)
    _write_json_exclusive(
        destination,
        wrapper,
        overwrite=overwrite,
    )
    return wrapper


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Eaton pre-annotation structural-futility audit. "
            "No annotation files are accepted or read."
        )
    )
    parser.add_argument("--protocol-config", type=Path, default=DEFAULT_PROTOCOL_CONFIG_PATH)
    parser.add_argument(
        "--artifact-commitments",
        type=Path,
        default=DEFAULT_ARTIFACT_COMMITMENTS_PATH,
    )
    parser.add_argument(
        "--prediction-commitments",
        type=Path,
        default=DEFAULT_PREDICTION_COMMITMENTS_PATH,
        help=(
            "Tracked post-run trust root; the default must exist before the first "
            "formal futility audit"
        ),
    )
    parser.add_argument("--protocol-summary", type=Path, default=DEFAULT_PROTOCOL_SUMMARY_PATH)
    parser.add_argument(
        "--study-development-manifest",
        type=Path,
        default=DEFAULT_DEVELOPMENT_MANIFEST_PATH,
    )
    parser.add_argument("--predictions-csv", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument(
        "--prediction-metadata",
        type=Path,
        default=DEFAULT_PREDICTION_METADATA_PATH,
    )
    parser.add_argument("--expected-protocol-summary-sha256", required=True)
    parser.add_argument("--expected-prediction-csv-sha256", required=True)
    parser.add_argument("--expected-prediction-metadata-sha256", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    wrapper = run_preannotation_futility_gate(
        protocol_config_path=args.protocol_config,
        artifact_commitments_path=args.artifact_commitments,
        prediction_commitments_path=args.prediction_commitments,
        protocol_summary_path=args.protocol_summary,
        development_manifest_path=args.study_development_manifest,
        predictions_path=args.predictions_csv,
        prediction_metadata_path=args.prediction_metadata,
        expected_protocol_summary_sha256=args.expected_protocol_summary_sha256,
        expected_prediction_csv_sha256=args.expected_prediction_csv_sha256,
        expected_prediction_metadata_sha256=args.expected_prediction_metadata_sha256,
        output_path=args.output,
        overwrite=args.overwrite,
    )
    artifact = wrapper["futility_artifact"]
    print(
        json.dumps(
            {
                "output": str(
                    _absolute_without_following(
                        DEFAULT_OUTPUT_PATH if args.output is None else args.output
                    )
                ),
                "phase_status": artifact["phase_status"],
                "structurally_futile": artifact["structurally_futile"],
                "futility_artifact_sha256": artifact["futility_artifact_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
