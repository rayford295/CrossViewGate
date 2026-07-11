from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from crossview_conflict.data.dins import (
    compute_field_statistics,
    extract_field_domains,
    features_to_frame,
    fetch_dins_layer,
    join_dins_fields,
    write_dins_artifacts,
)


def _feature(objectid: int, longitude: float, latitude: float, **properties: object) -> dict[str, object]:
    return {
        "type": "Feature",
        "id": objectid,
        "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
        "properties": {"OBJECTID": objectid, **properties},
    }


class DINSJoinTest(unittest.TestCase):
    def test_unique_business_key_wins_and_objectid_is_only_provenance(self) -> None:
        manifest = pd.DataFrame(
            [
                {
                    "sample_id": "attachment-1",
                    "objectid": 999,
                    "GLOBALID": "{STABLE-A}",
                    "latitude": 34.0,
                    "longitude": -118.001,
                }
            ]
        )
        source = features_to_frame(
            {
                "type": "FeatureCollection",
                "features": [
                    _feature(
                        1,
                        -118.0,
                        34.0,
                        GLOBALID="stable-a",
                        ROOFCONSTRUCTION="Asphalt",
                    ),
                    _feature(
                        999,
                        -118.001,
                        34.0,
                        GLOBALID="different-record",
                        ROOFCONSTRUCTION="Wood",
                    ),
                ],
            }
        )

        joined, summary = join_dins_fields(
            manifest,
            source,
            tolerance_m=5.0,
            service_url="https://example.test/FeatureServer/2",
        )

        self.assertEqual(joined.loc[0, "dins_join_status"], "matched_business_key")
        self.assertEqual(joined.loc[0, "dins_source_objectid"], 1)
        self.assertEqual(joined.loc[0, "dins_roofconstruction"], "Asphalt")
        self.assertTrue(joined.loc[0, "dins_join_distance_warning"])
        self.assertIn("never", summary["objectid_policy"])

    def test_spatial_join_records_distance_candidates_and_ambiguity(self) -> None:
        source = features_to_frame(
            {
                "type": "FeatureCollection",
                "features": [
                    _feature(10, -118.0, 34.0, EAVES="Enclosed"),
                    _feature(11, -117.99995, 34.0, EAVES="Unenclosed"),
                    _feature(20, -118.000005, 34.001, EAVES="Left"),
                    _feature(21, -117.999995, 34.001, EAVES="Right"),
                ],
            }
        )
        manifest = pd.DataFrame(
            [
                {"sample_id": "clear", "latitude": 34.0, "longitude": -118.0},
                {"sample_id": "ambiguous", "latitude": 34.001, "longitude": -118.0},
                {"sample_id": "far", "latitude": 35.0, "longitude": -119.0},
                {"sample_id": "missing", "latitude": None, "longitude": None},
            ]
        )

        joined, summary = join_dins_fields(
            manifest,
            source,
            tolerance_m=10.0,
            ambiguity_margin_m=1.0,
        )

        clear = joined.set_index("sample_id").loc["clear"]
        self.assertEqual(clear["dins_join_status"], "matched_spatial")
        self.assertEqual(clear["dins_join_candidate_count"], 2)
        self.assertAlmostEqual(clear["dins_join_distance_m"], 0.0, places=6)
        self.assertEqual(clear["dins_eaves"], "Enclosed")

        ambiguous = joined.set_index("sample_id").loc["ambiguous"]
        self.assertEqual(ambiguous["dins_join_status"], "spatial_ambiguous")
        self.assertTrue(ambiguous["dins_join_ambiguous"])
        self.assertEqual(ambiguous["dins_join_candidate_count"], 2)
        self.assertTrue(pd.isna(ambiguous["dins_eaves"]))
        self.assertEqual(summary["status_counts"]["unmatched"], 1)
        self.assertEqual(summary["status_counts"]["missing_coordinates"], 1)

    def test_objectid_is_rejected_as_an_explicit_business_key(self) -> None:
        manifest = pd.DataFrame([{"OBJECTID": 1, "latitude": 34.0, "longitude": -118.0}])
        source = features_to_frame(
            {
                "type": "FeatureCollection",
                "features": [_feature(1, -118.0, 34.0, DAMAGE="No Damage")],
            }
        )
        with self.assertRaisesRegex(ValueError, "OBJECTID/OID/FID"):
            join_dins_fields(
                manifest,
                source,
                business_key_candidates=(("OBJECTID",),),
            )

    def test_statistics_and_domains_preserve_null_unknown_distinction(self) -> None:
        source = pd.DataFrame(
            {
                "EAVES": [None, " Unknown ", "", "Enclosed"],
                "DAMAGE": ["No Damage", "No Damage", "Minor", "Major"],
            }
        )
        metadata = {
            "editingInfo": {"lastEditDate": 1_700_000_000_000},
            "fields": [
                {
                    "name": "EAVES",
                    "alias": "Eaves",
                    "type": "esriFieldTypeString",
                    "domain": {
                        "type": "codedValue",
                        "codedValues": [{"name": "Enclosed", "code": "Enclosed"}],
                    },
                },
                {
                    "name": "DAMAGE",
                    "alias": "Damage",
                    "type": "esriFieldTypeString",
                    "domain": None,
                },
            ],
            "types": [],
        }

        statistics = compute_field_statistics(source, metadata)
        eaves = next(field for field in statistics["fields"] if field["name"] == "EAVES")
        self.assertEqual(eaves["null_or_blank_count"], 2)
        self.assertEqual(eaves["unknown_count"], 1)
        self.assertEqual(eaves["unknown_rate_non_null"], 0.5)
        self.assertEqual(eaves["semantic_group"], "visually_observable_candidate")

        domains = extract_field_domains(metadata, service_url="https://example.test/layer")
        self.assertEqual(domains["fields_with_direct_domain"], 1)
        self.assertEqual(domains["fields"][0]["domain"]["type"], "codedValue")

    def test_fetch_uses_ordered_pagination_and_validates_count(self) -> None:
        offsets: list[int] = []
        all_features = [
            _feature(index, -118.0 + index / 10_000, 34.0, DAMAGE="No Damage")
            for index in range(1, 6)
        ]

        def requester(url: str, params: object) -> dict[str, object]:
            params = dict(params)  # type: ignore[arg-type]
            if not url.endswith("/query"):
                return {
                    "objectIdField": "OBJECTID",
                    "globalIdField": "GLOBALID",
                    "maxRecordCount": 2,
                    "editingInfo": {"lastEditDate": 1_700_000_000_000},
                    "fields": [],
                }
            if params.get("returnCountOnly") == "true":
                return {"count": 5}
            offset = int(params["resultOffset"])
            offsets.append(offset)
            size = int(params["resultRecordCount"])
            return {
                "type": "FeatureCollection",
                "properties": {
                    "exceededTransferLimit": offset + size < len(all_features)
                },
                "features": all_features[offset : offset + size],
            }

        _, geojson, provenance = fetch_dins_layer(
            "https://example.test/FeatureServer/2",
            where="INCIDENTNAME = 'Eaton'",
            requester=requester,
        )
        self.assertEqual(offsets, [0, 2, 4])
        self.assertEqual(len(geojson["features"]), 5)
        self.assertEqual(provenance["page_count"], 3)
        self.assertEqual(provenance["pagination_order"], "OBJECTID ASC")

    def test_artifact_writer_refuses_existing_outputs_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            output_dir = Path(tempdir) / "snapshot"
            kwargs = {
                "geojson": {"type": "FeatureCollection", "features": []},
                "service_metadata_artifact": {"field_service_metadata": {}},
                "domains": {"fields": []},
                "statistics": {"fields": []},
                "joined_manifest": pd.DataFrame([{"sample_id": "a"}]),
                "provenance": {"guardrails": {}},
            }
            paths = write_dins_artifacts(output_dir, **kwargs)
            self.assertTrue(paths["joined_manifest"].is_file())
            with self.assertRaises(FileExistsError):
                write_dins_artifacts(output_dir, **kwargs)

    def test_cli_local_geojson_dry_run_is_offline_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            manifest_path = root / "manifest.csv"
            pd.DataFrame(
                [{"sample_id": "local", "latitude": 34.0, "longitude": -118.0}]
            ).to_csv(manifest_path, index=False)
            geojson_path = root / "fields.geojson"
            geojson_path.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            _feature(1, -118.0, 34.0, STRUCTURETYPE="Residence")
                        ],
                    }
                ),
                encoding="utf-8",
            )
            metadata_path = root / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "objectIdField": "OBJECTID",
                        "fields": [
                            {"name": "OBJECTID", "domain": None},
                            {"name": "STRUCTURETYPE", "domain": None},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "must_not_exist"
            repo_root = Path(__file__).resolve().parents[1]
            completed = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "fetch_dins_fields.py"),
                    "--manifest-csv",
                    str(manifest_path),
                    "--source-geojson",
                    str(geojson_path),
                    "--source-metadata",
                    str(metadata_path),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(completed.stdout)
            self.assertEqual(report["mode"], "dry_run_no_files_written")
            self.assertFalse(report["network_mode"])
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
