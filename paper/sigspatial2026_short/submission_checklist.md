# Submission Checklist

This checklist converts the peer-review notes in
`/Users/yifn/Desktop/peer_review_cross_view_disaster_triage.md` into concrete
pre-submission tasks for the short paper.

## P0 Before EasyChair Upload

- Add a compact wildfire data statement: total N, train/validation/test counts,
  positive and negative class counts, image source, collection dates if
  available, annotation protocol, annotator count, and release/access plan.
- Confirm whether inter-annotator agreement exists. If it does not, say so
  directly and describe the quality-control process that was used instead.
- Treat the hurricane grouped-clean endpoint split as the conservative endpoint
  control. Do not make the original endpoint split the main evidence.
- Remove all draft timestamps and ACM template placeholders from the compiled
  PDF. The short-paper source already sets the conference metadata and has no
  `Draft updated` title suffix.
- Compile and confirm the PDF is at most 4 pages including references.

## P1 Strongly Recommended

- Add a one-sentence data leakage note with the known object-level leakage
  issue and why grouped splitting is the cleaner control.
- Report confidence intervals for the key CAE comparisons if space allows.
- Run or report a simple single-view ensemble baseline, such as probability
  averaging or majority vote, because reviewers may ask whether learned fusion
  beats simple aggregation.
- If the tile-level map remains in the paper, add a spatial-autocorrelation
  caveat or compute Moran's I for the 27 wildfire tiles.
- Keep the observation-regime claim calibrated: the two datasets differ by
  disaster type, geography, data source, labels, and metadata, not only by
  ground-view regime.

## P2 Optional If Space Allows

- Add a one-line backbone robustness note: CLIP and DINOv2 underperform the
  supervised CNN baselines in this task.
- Mention that ConflictFocalLoss needs comparison against standard focal loss
  and class-balanced loss before it can be framed as a full method contribution.
- Add a compact figure from the long draft only if the PDF remains under 4
  pages after references.
