# FOV Intervention Results (Phase 2)

Causal test of the view-regime claim: hurricane panoramas are cropped to a
256x256 (90-degree) window that is either centered on the SegFormer building
centroid (`fov_building`) or on a uniform random azimuth (`fov_random`).
Geometry is identical between the two variants; only target alignment
differs. `original_panorama` rows are the main multiseed suite. remote_only
is shared across variants, so conflict subsets change only through the
street model.

| dataset | variant | street_acc | crossview_acc | conflict_rate | conflict_gain | gap_closure |
| --- | --- | --- | --- | --- | --- | --- |
| ian_original | fov_building | 0.6478 +/- 0.0069 | 0.6889 +/- 0.0150 | 0.3611 +/- 0.0069 | 0.1086 +/- 0.1020 | 0.2692 +/- 0.2516 |
| ian_original | fov_random | 0.5744 +/- 0.0168 | 0.6733 +/- 0.0133 | 0.3756 +/- 0.0310 | 0.0418 +/- 0.1074 | 0.0723 +/- 0.3574 |
| ian_original | original_panorama | 0.7089 +/- 0.0150 | 0.7211 +/- 0.0379 | 0.3467 +/- 0.0145 | 0.0643 +/- 0.0849 | 0.1663 +/- 0.2322 |
| milton_original | fov_building | 0.6929 +/- 0.0350 | 0.7388 +/- 0.0378 | 0.3373 +/- 0.0316 | 0.1494 +/- 0.0524 | 0.3652 +/- 0.1385 |
| milton_original | fov_random | 0.6535 +/- 0.0104 | 0.6982 +/- 0.0186 | 0.2940 +/- 0.0344 | 0.0552 +/- 0.0585 | 0.1279 +/- 0.1255 |
| milton_original | original_panorama | 0.7402 +/- 0.0205 | 0.7454 +/- 0.0127 | 0.2808 +/- 0.0524 | 0.0669 +/- 0.0376 | 0.1774 +/- 0.0946 |

## Per-seed paired contrasts (conflict gain)

The causal contrast is `fov_building - fov_random` (identical geometry,
alignment is the only difference): positive in 6/6 dataset-seed pairs.
`fov_building - original_panorama` is positive in 5/6.

| dataset | seed | building - random | building - original |
| --- | --- | --- | --- |
| ian_original | 42 | +0.0416 | +0.1797 |
| ian_original | 123 | +0.0693 | -0.0550 |
| ian_original | 456 | +0.0895 | +0.0083 |
| milton_original | 42 | +0.1575 | +0.1588 |
| milton_original | 123 | +0.0483 | +0.0656 |
| milton_original | 456 | +0.0768 | +0.0231 |

## Reading

1. Building-centered cropping raises the hurricane conflict gain toward the
   wildfire regime (IAN closure 0.166 -> 0.269, Milton 0.177 -> 0.365;
   wildfire reference 0.388) even though it lowers street-only accuracy by
   discarding panoramic context.
2. Random cropping with identical geometry does not (0.072 / 0.128), so the
   effect is target alignment, not cropping or resolution.
3. This upgrades the view-regime claim from a cross-dataset observation to a
   within-dataset controlled intervention, and gives the gate's visibility
   features causal standing.
