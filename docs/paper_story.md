# Paper Story

## One-sentence story

Cross-view fusion helps most when single-view evidence conflicts, and that help
is strongest when the ground-view image is property-centric rather than
panoramic.

## Proposed paper structure

### 1. Problem

Existing cross-view disaster papers mostly ask whether multimodal fusion improves
average performance.

That is not enough. In practice, the important question is:

**When does cross-view fusion matter most?**

### 2. Main hypothesis

Cross-view fusion is especially valuable on `conflict cases`, where:

- the street-view model and overhead model disagree

Furthermore, the benefit depends on the ground-view regime:

- stronger for building-centric property views
- weaker but still present for panoramic environmental views

### 3. Datasets

- Eaton / Altadena wildfire:
  property-centric building inspection views
- IAN hurricane:
  360-style environmental street views

### 4. Method

Use a unified triage architecture and training protocol:

- `street_only`
- `remote_only`
- `crossview`

Then evaluate:

- overall test performance
- conflict subset performance

### 5. Key result

Cross-view wins in both datasets, but:

- the margin is much larger in wildfire building-view data
- the margin is smaller in hurricane panoramic-view data

### 6. Contribution

This gives a mechanism-oriented conclusion:

Cross-view fusion is not just a stronger average classifier. It is a conflict
resolver whose value depends on the semantic alignment of the ground-view image
to the target structure.
