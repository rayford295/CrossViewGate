from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PanoramaSector:
    sector_id: int
    relative_azimuth_deg: float
    horizontal_fov_deg: float
    vertical_fov_deg: float


def normalize_signed_angle(angle_deg: float) -> float:
    """Normalize an angle to [-180, 180)."""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def build_panorama_sectors(
    num_sectors: int = 8,
    horizontal_fov_deg: float = 90.0,
    vertical_fov_deg: float = 90.0,
) -> tuple[PanoramaSector, ...]:
    if num_sectors < 2:
        raise ValueError("num_sectors must be at least two")
    if not 0.0 < horizontal_fov_deg <= 360.0:
        raise ValueError("horizontal_fov_deg must be in (0, 360]")
    if not 0.0 < vertical_fov_deg <= 180.0:
        raise ValueError("vertical_fov_deg must be in (0, 180]")
    step = 360.0 / float(num_sectors)
    return tuple(
        PanoramaSector(
            sector_id=index,
            relative_azimuth_deg=normalize_signed_angle(index * step),
            horizontal_fov_deg=float(horizontal_fov_deg),
            vertical_fov_deg=float(vertical_fov_deg),
        )
        for index in range(num_sectors)
    )


def validate_equirectangular_size(
    size: tuple[int, int], *, aspect_tolerance: float = 0.02
) -> None:
    width, height = (int(size[0]), int(size[1]))
    if width < 2 or height < 2:
        raise ValueError(f"Panorama is too small: {size}")
    aspect = width / float(height)
    if not math.isclose(aspect, 2.0, rel_tol=0.0, abs_tol=aspect_tolerance):
        raise ValueError(
            f"Expected a 2:1 equirectangular panorama, got {width}x{height} "
            f"(aspect={aspect:.4f})"
        )


def sector_pixel_geometry(
    size: tuple[int, int],
    sector: PanoramaSector,
) -> dict[str, int | float]:
    validate_equirectangular_size(size)
    width, height = size
    center_fraction = (0.5 + sector.relative_azimuth_deg / 360.0) % 1.0
    center_x = center_fraction * float(width)
    crop_width = max(1, min(width, int(round(width * sector.horizontal_fov_deg / 360.0))))
    crop_height = max(1, min(height, int(round(height * sector.vertical_fov_deg / 180.0))))
    return {
        "center_x_px": center_x,
        "center_y_px": height / 2.0,
        "crop_width_px": crop_width,
        "crop_height_px": crop_height,
    }


def crop_equirectangular_sector(
    image: Image.Image,
    sector: PanoramaSector,
) -> Image.Image:
    """Extract a central-latitude sector with horizontal wrap-around.

    This is the same equirectangular window intervention used by the existing
    FOV experiment. It is an offline sequential-reveal benchmark, not a claim
    that the crop is a perspective camera rendering.
    """
    rgb = image.convert("RGB")
    geometry = sector_pixel_geometry(rgb.size, sector)
    array = np.asarray(rgb)
    height, width = array.shape[:2]
    crop_width = int(geometry["crop_width_px"])
    crop_height = int(geometry["crop_height_px"])
    center_x = int(round(float(geometry["center_x_px"]))) % width
    x_indices = (
        np.arange(crop_width, dtype=np.int64) + center_x - crop_width // 2
    ) % width
    y0 = max(0, (height - crop_height) // 2)
    crop = array[y0 : y0 + crop_height, :][:, x_indices]
    return Image.fromarray(crop)


def absolute_sector_azimuth(
    compass_angle_deg: float,
    relative_azimuth_deg: float,
) -> float:
    return (float(compass_angle_deg) + float(relative_azimuth_deg)) % 360.0


def sector_ids(sectors: Iterable[PanoramaSector]) -> tuple[int, ...]:
    return tuple(sector.sector_id for sector in sectors)
