from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from crossview_conflict.data.panorama import (
    absolute_sector_azimuth,
    build_panorama_sectors,
    crop_equirectangular_sector,
    normalize_signed_angle,
    sector_pixel_geometry,
    validate_equirectangular_size,
)
from scripts.build_cvian_active_view_manifests import build_sector_rows, crop_version


def test_eight_sector_geometry_is_forward_first_and_overlapping() -> None:
    sectors = build_panorama_sectors()
    assert [sector.sector_id for sector in sectors] == list(range(8))
    assert [sector.relative_azimuth_deg for sector in sectors] == [
        0.0,
        45.0,
        90.0,
        135.0,
        -180.0,
        -135.0,
        -90.0,
        -45.0,
    ]
    geometry = sector_pixel_geometry((1024, 512), sectors[0])
    assert geometry["center_x_px"] == 512.0
    assert geometry["crop_width_px"] == 256
    assert geometry["crop_height_px"] == 256


def test_crop_wraps_at_panorama_boundary() -> None:
    array = np.zeros((4, 8, 3), dtype=np.uint8)
    for column in range(8):
        array[:, column, :] = column
    image = Image.fromarray(array)
    sector = build_panorama_sectors(
        num_sectors=4,
        horizontal_fov_deg=180.0,
        vertical_fov_deg=180.0,
    )[2]
    crop = np.asarray(crop_equirectangular_sector(image, sector))
    assert crop.shape == (4, 4, 3)
    assert crop[0, :, 0].tolist() == [6, 7, 0, 1]


def test_angles_and_invalid_panorama_fail_closed() -> None:
    assert normalize_signed_angle(180.0) == -180.0
    assert absolute_sector_azimuth(350.0, 45.0) == 35.0
    with pytest.raises(ValueError, match="2:1"):
        validate_equirectangular_size((640, 480))


def test_lazy_sector_manifest_inherits_parent_role(tmp_path) -> None:
    panorama_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 4), color=(10, 20, 30)).save(panorama_path)
    source = pd.DataFrame(
        [
            {
                "sample_id": "000001",
                "mapillary_id": "12345678901234567",
                "label": 2,
                "severity": "2_SevereDamage",
                "spatial_block_id": "block_a",
                "sequence_id": "sequence_a",
                "compass_angle_deg": 350.0,
                "is_pano": 1,
                "street_view_path": str(panorama_path),
                "remote_sensing_path": str(tmp_path / "remote.png"),
            }
        ]
    )
    rows = build_sector_rows(
        source,
        split="test",
        image_dir=tmp_path / "sectors",
        num_sectors=8,
        horizontal_fov_deg=90.0,
        vertical_fov_deg=90.0,
        jpeg_quality=92,
        materialize_crops=False,
        overwrite=False,
    )
    assert len(rows) == 8
    assert set(rows["parent_sample_id"]) == {"000001"}
    assert set(rows["split_role"]) == {"test"}
    assert set(rows["sector_materialized"]) == {0}
    assert rows["sector_path"].eq("").all()
    assert rows["street_view_path"].eq("").all()
    assert set(rows["crop_version"]) == {
        "cvian-8sector-h90-v90-overlap45deg-v1"
    }
    assert crop_version(4, 120.0, 80.0) == "cvian-4sector-h120-v80-overlap30deg-v1"
    assert rows.sort_values("sector_id")["absolute_azimuth_deg"].tolist()[:2] == [
        350.0,
        35.0,
    ]
