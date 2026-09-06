# GeoSearch 2026 Short Paper (4 pages)

Four-page short-paper version of the ISPRS manuscript
(`../isprs_manuscript.md`, "Trust the view that sees the target"), reframed for
the 5th ACM SIGSPATIAL International Workshop on Searching and Mining Large
Collections of Geospatial Data (GeoSearch '26). It does not replace the
ISPRS manuscript or the June SIGSPATIAL short draft in `../sigspatial2026_short/`.

## Target

- Venue: GeoSearch '26, co-located with ACM SIGSPATIAL 2026, Riverside, CA
- Workshop day: November 3, 2026
- Paper type: short research paper, 4 pages (we keep references inside the
  4 pages because the CFP does not say they are excluded)
- Review: single-blind (author names and affiliations are listed)
- Template: ACM `acmart` `sigconf`, submission mode (`\setcopyright{none}`,
  `printacmref=false`); the ACM rights block goes in at camera-ready
- Submission: <https://easychair.org/my/conference?conf=geosearch2026>
- CFP: <https://geosearch-workshop.github.io/geosearch2026/>

## Dates

- Submission deadline: September 6, 2026 (extended)
- Notification: September 18, 2026
- Camera-ready: October 12, 2026

## What changed relative to the ISPRS manuscript

- Title gains the "Mining Cross-View Conflicts" hook; the introduction frames
  conflict cases as a search-and-mining problem over paired collections.
- Kept: oracle gap, linear reliability gate, calibration decomposition,
  field-of-view intervention, conflict-density map, CVIAN split-repair note.
- Dropped for space: related-work section, Figure 1 overview, Figure 4 (FOV
  panorama), Figure 6 (qualitative), ordinal metrics, cross-disaster transfer
  details, Moran's I discussion beyond one sentence.
- Table 2 merges the ISPRS main table and oracle table (closure column).
  concat closures (0.29 / 0.15 / 0.16) come from
  `docs/results/calibration_decomposition_v2.md`; the other closures follow
  the ISPRS manuscript / `docs/results/multiseed_v2_results.md`.
- Bibliography trimmed to 11 entries (`references.bib`, subset of
  `../references.bib` plus Guo et al. 2017).

## Build

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Figures are referenced from `../../figures/` (`fig3_gate.pdf`,
`fig5_conflict_density.pdf`). Compiled locally with TinyTeX / acmart v2.20:
4 pages, US Letter, all fonts embedded.
