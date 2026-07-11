from __future__ import annotations

import unittest

import pandas as pd

from scripts.build_cvian_sequence_four_role_manifests import (
    FIT_ROLE_NAMES,
    ROLE_NAMES,
    assign_grouped_roles,
    build_four_role_protocol,
    derive_sequence_clean_holdout,
    pairwise_role_audit,
    purge_shared_spatial_blocks,
    reconstruct_selector_fit,
)


def _sequence_rows(
    sequence_id: str,
    *,
    block_id: str,
    latitude: float,
    sample_prefix: str | None = None,
) -> list[dict[str, object]]:
    prefix = sample_prefix or sequence_id
    return [
        {
            "sample_id": f"{prefix}-{label}",
            "sequence_id": sequence_id,
            "spatial_block_id": block_id,
            "label": label,
            "latitude": latitude,
            "longitude": -82.0,
        }
        for label in range(3)
    ]


class CVIANSequenceFourRoleProtocolTest(unittest.TestCase):
    def test_sequence_clean_holdout_rejects_an_entire_partly_exposed_sequence(self) -> None:
        frame = pd.DataFrame(
            _sequence_rows("clean", block_id="clean-block", latitude=26.0)
            + _sequence_rows("dirty", block_id="dirty-block", latitude=26.1)
        )
        holdout, remainder, audit = derive_sequence_clean_holdout(
            frame,
            {"dirty-0"},
            minimum_rows=3,
        )

        self.assertEqual(set(holdout["sequence_id"]), {"clean"})
        self.assertEqual(set(remainder["sequence_id"]), {"dirty"})
        dirty = audit.loc[audit["sequence_id"] == "dirty"].iloc[0]
        self.assertEqual(int(dirty["selection_exposed_rows"]), 1)
        self.assertFalse(bool(dirty["eligible_for_prospective_test"]))

    def test_grouped_role_assignment_is_deterministic_class_complete_and_isolated(self) -> None:
        rows: list[dict[str, object]] = []
        for index in range(18):
            rows.extend(
                _sequence_rows(
                    f"sequence-{index:02d}",
                    block_id=f"block-{index:02d}",
                    latitude=26.0 + index * 0.01,
                )
            )
        frame = pd.DataFrame(rows)
        fractions = {"base_fit": 0.7, "selector_fit": 0.2, "validation": 0.1}

        first, first_assignment = assign_grouped_roles(
            frame,
            role_fractions=fractions,
            group_column="sequence_id",
            label_column="label",
            seed=20260710,
        )
        second, second_assignment = assign_grouped_roles(
            frame,
            role_fractions=fractions,
            group_column="sequence_id",
            label_column="label",
            seed=20260710,
        )

        self.assertEqual(first_assignment.to_dict("records"), second_assignment.to_dict("records"))
        for role in FIT_ROLE_NAMES:
            self.assertEqual(set(first[role]["label"]), {0, 1, 2})
            self.assertEqual(first[role]["sample_id"].tolist(), second[role]["sample_id"].tolist())
        for left_index, left in enumerate(FIT_ROLE_NAMES):
            for right in FIT_ROLE_NAMES[left_index + 1 :]:
                self.assertFalse(
                    set(first[left]["sequence_id"]) & set(first[right]["sequence_id"])
                )

    def test_reconstruct_selector_fit_matches_seeded_block_contract(self) -> None:
        rows: list[dict[str, object]] = []
        for block_index in range(10):
            rows.extend(
                _sequence_rows(
                    f"sequence-{block_index}",
                    block_id=f"block-{block_index}",
                    latitude=26.0 + block_index * 0.01,
                )
            )
        train = pd.DataFrame(rows)

        first, first_groups, first_attempt = reconstruct_selector_fit(
            train,
            seed=42,
            block_fraction=0.2,
        )
        second, second_groups, second_attempt = reconstruct_selector_fit(
            train,
            seed=42,
            block_fraction=0.2,
        )

        self.assertEqual(first_attempt, second_attempt)
        self.assertEqual(first_groups, second_groups)
        self.assertEqual(len(first_groups), 2)
        self.assertEqual(first["sample_id"].tolist(), second["sample_id"].tolist())
        self.assertEqual(set(first["label"]), {0, 1, 2})

    def test_shared_block_purge_protects_roles_in_declared_order(self) -> None:
        roles = {
            "base_fit": pd.DataFrame(
                _sequence_rows("base", block_id="shared", latitude=26.0)
                + _sequence_rows("base-kept", block_id="base", latitude=26.3)
            ),
            "selector_fit": pd.DataFrame(
                _sequence_rows("selector", block_id="validation", latitude=26.1)
                + _sequence_rows("selector-kept", block_id="selector", latitude=26.4)
            ),
            "validation": pd.DataFrame(
                _sequence_rows("validation", block_id="validation", latitude=26.1)
            ),
            "prospective_test": pd.DataFrame(
                _sequence_rows("test", block_id="shared", latitude=26.0)
            ),
        }
        retained, excluded = purge_shared_spatial_blocks(
            roles,
            priority=("prospective_test", "validation", "selector_fit", "base_fit"),
        )

        self.assertEqual(set(retained["prospective_test"]["sequence_id"]), {"test"})
        self.assertEqual(set(retained["validation"]["sequence_id"]), {"validation"})
        self.assertEqual(set(retained["selector_fit"]["sequence_id"]), {"selector-kept"})
        self.assertEqual(set(retained["base_fit"]["sequence_id"]), {"base-kept"})
        self.assertEqual(set(excluded["excluded_from"]), {"base_fit", "selector_fit"})
        self.assertTrue((excluded["exclusion_stage"] == "shared_spatial_block").all())

    def test_end_to_end_protocol_is_sequence_block_and_buffer_clean(self) -> None:
        rows: list[dict[str, object]] = []
        exposed_ids: set[str] = set()
        for index in range(15):
            sequence_rows = _sequence_rows(
                f"fit-{index:02d}",
                block_id=f"fit-block-{index:02d}",
                latitude=26.0 + index * 0.01,
            )
            rows.extend(sequence_rows)
            exposed_ids.add(str(sequence_rows[0]["sample_id"]))
        rows.extend(
            _sequence_rows("test-a", block_id="test-a", latitude=27.0)
            + _sequence_rows("test-b", block_id="test-b", latitude=27.1)
        )
        full = pd.DataFrame(rows)

        roles, exclusions, _, _ = build_four_role_protocol(
            full,
            selection_exposed_sample_ids=exposed_ids,
            fit_role_fractions={"base_fit": 0.7, "selector_fit": 0.2, "validation": 0.1},
            spatial_buffer_m=25.0,
            seed=7,
            minimum_test_rows=6,
        )
        audit = pairwise_role_audit(roles, 25.0)

        self.assertEqual(tuple(roles), ROLE_NAMES)
        self.assertEqual(set(roles["prospective_test"]["sequence_id"]), {"test-a", "test-b"})
        self.assertTrue((audit["sample_id_overlap"] == 0).all())
        self.assertTrue((audit["sequence_id_overlap"] == 0).all())
        self.assertTrue((audit["spatial_block_id_overlap"] == 0).all())
        self.assertTrue((audit["minimum_distance_m"] > 25.0).all())
        self.assertTrue(exclusions.empty)


if __name__ == "__main__":
    unittest.main()
