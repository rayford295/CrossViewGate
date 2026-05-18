# SIGSPATIAL 2026 Short Paper Draft

This directory starts a separate ACM SIGSPATIAL 2026 Short and Poster Papers
draft. It does not replace or delete the longer draft in `../sigspatial2026_draft.tex`.

## Target

- Venue: ACM SIGSPATIAL 2026 Short and Poster Papers
- Submission type: Experiment, Benchmark & Experience short paper
- Title suffix: `[experiments]`
- Page limit: 4 pages including references
- Template: ACM two-column conference proceedings template
- Submission link: <https://easychair.org/conferences/?conf=acmsigspatial2026sho>

## Important Dates

- Abstract: June 9, 2026, 11:59 PM Pacific Time
- Paper: June 12, 2026, 11:59 PM Pacific Time
- Notification: August 5, 2026, 11:59 PM Pacific Time
- Camera-ready: August 19, 2026, 11:59 PM Pacific Time

## Files

- `main.tex`: 4-page short-paper source
- `references.bib`: short-paper bibliography subset
- `submission_checklist.md`: pre-submission blockers and review-derived tasks

## Build

From this directory:

```bash
tectonic main.tex
```

The current source is intentionally conservative. It treats the grouped-clean
hurricane result as a boundary condition, narrows the ConflictFocalLoss claim to
a wildfire pilot, and keeps the observation-regime argument as an interpretation
rather than a causal proof.
