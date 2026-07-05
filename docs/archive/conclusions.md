# Current Conclusions

## Core claim

Cross-view fusion is most valuable under evidence conflict.

## What the current experiments show

### 1. Cross-view wins on both datasets

- wildfire building-view benchmark: `crossview` is best overall
- hurricane panoramic-view benchmark: `crossview` is also best overall

### 2. Conflict cases matter most

The strongest effect is not on average easy samples, but on the subset where:

- `street_only` and `remote_only` disagree

This conflict subset is the most informative slice for evaluating the true value
of cross-view fusion.

### 3. View regime changes the strength of the gain

The gain from cross-view is stronger on wildfire property-centric imagery than
on hurricane panoramic imagery.

This suggests the mechanism is related to target alignment:

- building-centric ground views expose structure-level damage more directly
- panoramic ground views contain more environmental context and weaker direct
  alignment to the target structure

### 4. A lightweight alignment proxy supports the mechanism

A frozen semantic-segmentation analysis on the conflict subset shows:

- wildfire conflict images have much higher `building` pixel coverage
- wildfire conflict images also place that building evidence closer to the image center
- hurricane conflict images are much more environment-dominant

So the wildfire-vs-hurricane difference is no longer only descriptive. It is
backed by an interpretable image-level proxy for target alignment.

### 5. Supplementary statistical checks refine the claim

The new supplementary analyses sharpen the paper's wording:

- permutation testing supports a strong wildfire conflict claim
- hurricane remains a positive trend, but is not yet statistically strong
- threshold sensitivity shows the rank order is robust
- per-sample alignment correlation is weak, so the alignment effect should be framed at the view-regime level

## Working paper interpretation

Cross-view fusion is not equally useful in all settings. Its contribution grows
when:

- the two modalities provide conflicting evidence
- the ground view is tightly aligned with the damaged structure

Its contribution is smaller, though still real, when:

- the ground view is dominated by broad environmental context
