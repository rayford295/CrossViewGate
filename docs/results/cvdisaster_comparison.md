# CVDisaster Comparison (Closest Prior Work)

CVDisaster (Hao et al., ISPRS J. Photogramm. Remote Sens. 2025;
arXiv:2408.06761; github.com/tum-bgd/CVDisaster) is the closest prior work:
cross-view damage classification on Hurricane Ian with street-view and VHR
satellite imagery over Sanibel Island.

## Dataset relationship

Our `ian_original` dataset is the released CVIAN pairing (identical
sat/svi pair structure and 3-class damage-perception labels:
light/medium/heavy = Minor/Moderate/Severe). Two differences must be stated:

1. **Size**: the paper reports 1,135 manually labeled SVI; the release we use
   contains 4,121 pairs (Minor 1,407 / Moderate 1,538 / Severe 1,176).
2. **Split protocol**: CVDisaster uses random train/test splits (best result
   at 5:5). We use grouped splits by `objectid` to prevent spatial leakage
   between nearly co-located panoramas, which is strictly harder. The IAN
   source release carries no coordinates, so grouping is the only available
   leakage control.

## Published numbers vs ours

CVDisaster-Est (CGCViT-Tiny, 20M params, 100 epochs, 5:5 random split):

| Approach | Overall Accuracy | F1 |
| --- | --- | --- |
| Street-view only | 74.50 | 0.73 |
| Satellite only | 67.07 | 0.65 |
| CVDisaster-Est (cross-view) | 77.96 | 0.77 |

Ours (ResNet18, 3-epoch multiseed protocol, grouped split, 3 seeds):

| Approach | Accuracy | macro F1 |
| --- | --- | --- |
| street_only | 70.89 +/- 1.50 | 0.713 |
| remote_only | 64.56 +/- 3.02 | 0.644 |
| crossview | 72.11 +/- 3.79 | 0.724 |
| gate3_linear (ours) | 73.44 | — |

## Positioning

1. **The qualitative ranking replicates exactly**: street > satellite alone,
   cross-view best. This is mutual validation across independent codebases.
2. **Absolute numbers are not directly comparable**: their split is random
   (subject to spatial leakage the source metadata cannot rule out), ours is
   grouped; their encoder is 2x larger and trained 30x longer. Our converged
   v2 suite will narrow the training gap; the split difference is a feature,
   not a bug, and we state it.
3. **The contributions are orthogonal**: CVDisaster contributes
   geolocalization + a fusion architecture; it evaluates only overall
   accuracy. We contribute conflict-aware evaluation (the oracle gap),
   the reliability gate, the causal FOV intervention, and the unsupervised
   conflict-density map — none of which appear in CVDisaster.
4. **What we adopt from them**: CVIAN as the shared benchmark and their
   published numbers as the external reference row in our results table.
