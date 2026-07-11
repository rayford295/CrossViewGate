from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.train_reliability_gate import main


class ReliabilityGateProtocolIntegrationTest(unittest.TestCase):
    def test_fixed_gate_writes_all_three_protocol_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            multiseed = root / "multiseed"
            visibility_dir = root / "visibility"
            protocol_dir = root / "protocol"
            output_dir = root / "gate"
            visibility_dir.mkdir()
            protocol_dir.mkdir()

            val_ids = [str(index) for index in range(6)]
            test_ids = [str(index) for index in range(6, 9)]
            val_targets = np.array([0, 1, 2, 0, 1, 2])
            test_targets = np.array([0, 1, 2])
            for mode_index, mode in enumerate(("street_only", "remote_only", "crossview")):
                run_dir = multiseed / "demo" / f"{mode}_seed1"
                run_dir.mkdir(parents=True)
                for split, sample_ids, targets in (
                    ("val", val_ids, val_targets),
                    ("test", test_ids, test_targets),
                ):
                    logits = np.full((len(targets), 3), -0.5, dtype=float)
                    for row_index, target in enumerate(targets):
                        logits[row_index, (int(target) + mode_index) % 3] = 1.5
                    frame = pd.DataFrame(
                        {
                            "sample_id": sample_ids,
                            "target": targets,
                            "prediction": logits.argmax(axis=1),
                            **{
                                f"logit_{class_index}": logits[:, class_index]
                                for class_index in range(3)
                            },
                        }
                    )
                    frame.to_csv(run_dir / f"{split}_predictions.csv", index=False)

            for split, sample_ids in (("val", val_ids), ("test", test_ids)):
                pd.DataFrame(
                    {
                        "sample_id": sample_ids,
                        "building_ratio": np.linspace(0.1, 0.6, len(sample_ids)),
                        "center_building_ratio": 0.2,
                        "center_minus_global": 0.0,
                        "centroid_distance_norm": 0.3,
                    }
                ).to_csv(visibility_dir / f"demo_{split}.csv", index=False)

            role_specs = {
                "gate_fit": val_ids[:3],
                "risk_calibration": val_ids[3:],
                "final_test": test_ids,
            }
            for role, sample_ids in role_specs.items():
                pd.DataFrame(
                    {
                        "sample_id": sample_ids,
                        "objectid": [f"{role}-object-{value}" for value in sample_ids],
                        "tile_id": [f"{role}-tile-{value}" for value in sample_ids],
                        "spatial_block_id": [f"{role}-tile-{value}" for value in sample_ids],
                        "sequence_id": [f"{role}-seq-{value}" for value in sample_ids],
                        "latitude": np.linspace(26.0, 26.1, len(sample_ids)),
                        "longitude": np.linspace(-82.0, -82.1, len(sample_ids)),
                        "protocol_role": role,
                        "event_id": "synthetic_event",
                    }
                ).to_csv(protocol_dir / f"{role}.csv", index=False)

            argv = [
                "train_reliability_gate.py",
                "--multiseed-root", str(multiseed),
                "--visibility-dir", str(visibility_dir),
                "--datasets", "demo",
                "--seeds", "1",
                "--hidden-dim", "2",
                "--epochs", "2",
                "--output-dir", str(output_dir),
                "--doc-path", str(root / "gate.md"),
                "--split-manifests", "",
                "--risk-protocol-manifests", f"demo={protocol_dir}",
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                main()

            prediction_dir = output_dir / "predictions"
            for role, sample_ids in role_specs.items():
                path = prediction_dir / f"demo_seed1_{role}.csv"
                self.assertTrue(path.is_file())
                evidence = pd.read_csv(path)
                self.assertEqual(len(evidence), len(sample_ids))
                self.assertEqual(set(evidence["protocol_role"]), {role})
                self.assertTrue(evidence["spatial_block_id"].notna().all())
                self.assertEqual(set(evidence["dataset"]), {"demo"})
                self.assertEqual(set(evidence["event_id"]), {"synthetic_event"})
                self.assertEqual(evidence["gate_artifact_id"].nunique(), 1)
            identities = {
                pd.read_csv(prediction_dir / f"demo_seed1_{role}.csv")[
                    "gate_artifact_id"
                ].iloc[0]
                for role in role_specs
            }
            self.assertEqual(len(identities), 1)
            self.assertTrue((prediction_dir / "demo_seed1.csv").is_file())


if __name__ == "__main__":
    unittest.main()
