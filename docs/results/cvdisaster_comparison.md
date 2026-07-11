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
   at 5:5). Our historical headline also used a release-derived/index-grouped
   protocol, but the official CVIAN position GeoJSON is available and has now
   been checksum-joined to all 4,121 pairs. The audit found that 299/300 legacy
   test samples share a Mapillary sequence with train and 297/300 occupy a
   0.005-degree block seen in train. New claims therefore use a repaired
   spatial-block split with a 25 m boundary buffer.

## Published numbers vs ours

CVDisaster-Est (CGCViT-Tiny, 20M params, 100 epochs, 5:5 random split):

| Approach | Overall Accuracy | F1 |
| --- | --- | --- |
| Street-view only | 74.50 | 0.73 |
| Satellite only | 67.07 | 0.65 |
| CVDisaster-Est (cross-view) | 77.96 | 0.77 |

Ours (ResNet18, converged five-seed repaired spatial protocol):

| Approach | Accuracy | macro F1 |
| --- | --- | --- |
| street_only | 64.67 +/- 1.21 | 0.6500 +/- 0.0131 |
| remote_only | 58.02 +/- 0.94 | 0.5825 +/- 0.0079 |
| crossview | 63.66 +/- 3.83 | 0.6416 +/- 0.0401 |
| concat | 65.78 +/- 1.84 | 0.6599 +/- 0.0219 |

## Positioning

1. **The stable qualitative finding is street > satellite alone.** Under the
   repaired spatial holdout, crossview is not reliably better than street and
   has substantially higher seed variability, so the legacy cross-view ranking
   must not be generalized to spatial holdout.
2. **Absolute numbers are not directly comparable**: their split is random;
   ours groups 0.005-degree spatial blocks and enforces a 25 m cross-role
   buffer. Their encoder is also larger and trained longer. The repaired
   protocol lowers five-seed macro-F1 by 0.049--0.097 depending on input mode,
   confirming that the split difference is scientifically material.
3. **The contributions are orthogonal**: CVDisaster contributes
   geolocalization + a fusion architecture; it evaluates only overall
   accuracy. We contribute conflict-aware evaluation (the oracle gap),
   the reliability gate, the causal FOV intervention, and the unsupervised
   conflict-density map — none of which appear in CVDisaster.
4. **What we adopt from them**: CVIAN as the shared benchmark and their
   published numbers as the external reference row in our results table.
