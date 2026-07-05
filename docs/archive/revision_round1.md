# Revision Suggestions — Round 1
**Paper**: Conflict-Aware Cross-View Disaster Triage Across Spatial Observation Regimes  
**Target venue**: ACM SIGSPATIAL 2026 (short paper, 4 pages)  
**Prepared**: 2026-05-24

---

## Summary Verdict

The paper has a genuinely useful idea (evaluate fusion specifically on the disagreement subset) and two interesting empirical outcomes (positive wildfire gain, negative leakage-controlled hurricane gain). The main risks for rejection are: (1) the contribution scope is unclear — is this a protocol paper or an empirical finding paper?; (2) the conflict subset sample sizes for hurricane settings are very small (nc = 32, 80), undermining statistical credibility; (3) the baseline model (ResNet18 late fusion) will draw scrutiny in 2026 without a stronger justification. Addressing these three issues is the highest-priority action before submission.

---

## Critical Issues (Must Fix)

### 1. Remove "[experiments]" from the title
The current title ends with `[experiments]`. This is a draft artifact that must be removed. It signals an unfinished paper to editors and reviewers.

**Suggested title**: *Conflict-Aware Cross-View Disaster Triage Across Spatial Observation Regimes*

---

### 2. Clarify the primary contribution
The paper currently claims three things: (a) CAE exposes conflict behavior, (b) leakage-controlled hurricane results set a negative boundary, (c) spatial observation regime explains the dominant-view switch. These read as findings, not contributions. The actual contribution is:

> **CAE as a lightweight reporting protocol** + the empirical demonstration that observation regime (property-centric vs. panoramic) predicts when fusion helps or fails.

Pick one as the lead. Recommendation: lead with the empirical finding (observation regime governs fusion value), and frame CAE as the methodological lens that makes this visible. The current framing buries this insight.

Concretely, revise the Introduction's final paragraph ("The paper makes three claims...") to state:

> "The paper's central finding is that spatial observation regime — how a ground view aligns with the labeled building — is a stronger predictor of fusion value than aggregate test-set F1. CAE makes this finding measurable."

---

### 3. Address sample-size credibility for hurricane conflict subsets
Table 2 shows nc = 32 for H end. and nc = 80 for H clean. Reporting F1 differences of +0.0805 and −0.0153 on 32 and 80 samples respectively will invite skepticism. Two options:

**Option A (preferred)**: Add bootstrapped 95% confidence intervals for Δview on all settings. Even a one-sentence note ("CI for Δview on H end.: [−0.04, +0.20], indicating high variance at nc = 32") converts a weakness into a strength — it shows you are aware and honest.

**Option B**: Demote H end. to a footnote or supplement and make H clean the primary hurricane result, acknowledging nc = 80 still has wide intervals.

Either way, add a sentence in Section 3 or the limitations paragraph that explicitly states: "The hurricane conflict subsets are small; the directional pattern is consistent with the regime interpretation but should be replicated on larger paired datasets."

---

### 4. Justify the ResNet18 late-fusion baseline more explicitly
In 2026, a reviewer will immediately ask: "Why ResNet18 late fusion and not a CLIP-based or transformer-based model?" The paper mentions in passing that "generic CLIP and DINOv2 features underperform the supervised CNN baselines." This needs to be:

1. Placed in a table (even a mini-table or footnote table) so readers can see the numbers.
2. Accompanied by a one-sentence explanation: the supervised ResNet18 outperforms off-the-shelf vision-language features on this domain-specific binary task because disaster damage is not well-represented in general-purpose pretraining distributions.

This transforms a potential weakness into a finding that supports your framing (disaster assessment is domain-specific; off-the-shelf models do not transfer directly).

---

## Major Issues (Should Fix)

### 5. Abstract: rewrite the "deliberately mixed" framing
"The main result is deliberately mixed" reads as defensive — as if the authors are preemptively excusing a negative result. Negative results are valuable; state them directly.

**Suggested rewrite** (abstract sentence 5–6):
> "Results differ by observation regime. On the wildfire sensitivity split — where ground imagery is property-centered — cross-view fusion improves conflict F1 by 0.0388. On the leakage-controlled hurricane endpoint split — where ground imagery is panoramic — the gain becomes negative (Δview = −0.0153), though aggregate F1 makes the two models appear nearly tied."

Similarly, remove "deliberately simple" in the Discussion (Section 4). Just say "the fusion model is simple by design to isolate the evaluation question from architectural choice." That's fine, but drop "deliberately" everywhere — it reads as hedging.

---

### 6. Explain leakage before using it as a concept
Section 2.3 mentions "a prior hurricane endpoint split had object-level leakage" and that H clean is the "grouped-clean" fix, but this is the first time leakage is mentioned. Readers unfamiliar with the dataset will not know what leaked or how. Add 1–2 sentences:

> "The original CVDisaster splits allow the same property to appear in both train and test via panoramic images that overlap across multiple labeled footprints. We re-split using object-level grouping so no property identifier appears in both partitions. The grouped-clean result is the conservative control."

This is essential context; its absence weakens the claim that H clean is the proper baseline.

---

### 7. Statistical framing for the threshold sweep
Section 3 says: "At τ = {0.1, 0.3, 0.5}, CV remains strongest on wildfire conflict accuracy (0.8349, 0.7821, 0.7538)." This is presented as a list of numbers without interpretation. Explain what the decay means: as τ increases, only the hardest disagreements remain, so lower absolute accuracy is expected. The point is that the *ordering* (CV > S > R) is preserved — say that explicitly.

---

### 8. Move the tile-level paragraph out of Results
Section 3's last paragraph ("The tile-level panel adds a geographic check...") belongs in Discussion. Results should report what the numbers are; Discussion should explain what they mean. Moving this paragraph improves the flow of Section 3.

---

### 9. Formalize or drop r_τ(m) notation
Equation (3) defines `r_τ(m) = (F1(m; D), F1(m; C_τ), n_c/N)` but this notation is never used again in the paper. Either use it consistently when referring to CAE reports, or remove Eq. (3) and just describe the three quantities in prose. Unused notation adds clutter without clarity.

---

### 10. Report ρ_S→O and ρ_O→S values in a table
Equations (4) and (5) define the directional arbitration rates but their numerical values are only visible in Figure 3(a) as pie-chart proportions. Add a small table or inline the numbers: "In the wildfire setting, ρ_{S→O} = 0.271 and ρ_{O→S} = 0.189; in the hurricane moderate-plus-severe setting, ρ_{S→O} = 0.375 and ρ_{O→S} = 0.344." This lets readers compare directly without reading pie charts.

---

## Minor Issues (Polish)

### 11. Section 2.1: relocate the causal-test disclaimer
"The comparison is not a causal test of wildfire versus hurricane" is buried in the data description. Move it to Section 4 (Discussion) or the limitations paragraph where it fits naturally.

### 12. Section 2.3: expand implementation details for reproducibility
The current implementation section does not report: train/val/test split ratios, number of random seeds, or what metric is used to select the "best-validation-F1 checkpoint." Add these. Example:
> "We use a 70/10/20 train/val/test split stratified by tile. All results are averaged over three random seeds (42, 123, 456); standard deviations are below 0.005 for all reported F1 scores."

### 13. Section 4: sharpen the operational routing rule
"Cases where S and R agree may use the agreement decision; cases where they disagree should route to CV or human review" is a useful insight but is buried. Consider making this a brief itemized list or highlighted box, since it is the direct operational takeaway of CAE.

### 14. Citation [1] needs a cleaner connection
Reference [1] (Geifman and El-Yaniv 2017 — selective classification) is cited as related to "disagreement-aware evaluation" but selective classification is about abstaining on uncertain examples, not about evaluating model disagreement. Either replace this with a more directly related citation on ensemble disagreement / disagreement-based evaluation, or add a sentence explaining precisely how CAE differs from selective classification.

### 15. Figure 3 readability
Figure 3 tries to pack five panels (a)–(e) into a single figure. Panels (c) and (d) (alignment proxy scatter and building alignment distributions) are particularly small and hard to read at print size. Consider:
- Merging (c) and (d) into a single two-panel figure with larger axis labels, or
- Moving (e) (tile maps) to a standalone figure since it tells a distinct spatial story.

### 16. Table 2: clarify the column header "nc"
Rename `nc` to `|C_τ|` or add a footnote "nc = |C_{τ=0.1}|" for readers who read the table before the text.

### 17. Conclusion: the reporting template is useful — make it more prominent
Section 5 proposes a specific reporting template (full-test F1, conflict F1 for each view, n_c, Δview, S→O vs O→S balance, spatial summary). This is the most actionable output of the paper. Consider extracting it as a named box or numbered list rather than embedding it in flowing prose.

---

## Framing / Narrative Suggestions

### On the "deliberate" language
The word "deliberately" appears four times (deliberately mixed, deliberately simple, deliberately straightforward baseline). This pattern signals defensiveness. Replace all instances with direct statements of design choice and rationale.

### On the negative result
The negative grouped-clean hurricane result is the most interesting finding in the paper. Currently it is framed as a "stress test" and a "deployment boundary." That framing is good — but make it even stronger by opening Section 4 with: "The most important result in Table 2 is the negative gain." Right now the paper describes the positive wildfire gain first and the negative hurricane gain second, which de-emphasizes what is novel and unexpected.

### On contribution positioning
If the intended contribution is the CAE protocol itself (not the specific empirical results), the paper should include a formal algorithm box or pseudocode for CAE so that other researchers can adopt it. A 6-line algorithm box would make it easier to cite and reuse.

If the intended contribution is the empirical finding about observation regime, then the protocol details can be compressed and the regime analysis should expand.

Currently the paper tries to do both at equal weight. For a 4-page short paper, pick one and cut the other's length by half.

---

## Checklist Before Resubmission

- [ ] Remove `[experiments]` from title
- [ ] Add confidence intervals or acknowledge small nc for H end. / H clean
- [ ] Add CLIP/DINOv2 comparison numbers (table or footnote)
- [ ] Explain object-level leakage before citing it
- [ ] Rewrite "deliberately mixed" abstract language
- [ ] Move tile-level paragraph from Section 3 to Section 4
- [ ] Drop or use Eq. (3) consistently
- [ ] Report ρ_S→O and ρ_O→S as numbers in the text
- [ ] Add multi-seed and split-ratio details to Section 2.3
- [ ] Sharpen citation [1] connection or replace it
- [ ] Improve Figure 3 readability (especially panels c–d)
- [ ] Rename `nc` column in Table 2
- [ ] Make the reporting template in Section 5 visually distinct
