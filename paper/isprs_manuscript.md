# Trust the view that sees the target: visibility-conditioned reliability gating for conflict-aware cross-view disaster damage assessment

Yifan Yang^a^, Lei Zou^a,\*^

^a^ Department of Geography, Texas A&M University, College Station, TX, USA

\* Corresponding author. E-mail: [to be added]; Y. Yang: yifan.yang@gisphere.info

## Abstract

Fusing ground-level and overhead imagery is increasingly common in post-disaster building damage assessment, yet fusion systems are almost universally symmetric: both views are weighted equally regardless of what each view actually observes. We show that this assumption discards most of the useful information in exactly the cases that matter. Across three real disaster datasets — property-centric inspection photographs from the 2025 Eaton wildfire and panoramic street-view imagery from Hurricanes Ian and Milton — an oracle that selects the correct single view on cross-view conflict cases exceeds every tested fusion method by 0.37–0.41 accuracy, a gap that survives model convergence, calibration, and architectural variation. We introduce a visibility-conditioned reliability gate: a linear model over street-view building-visibility features, calibrated per-view confidences, and cross-view disagreement that predicts per-sample view reliability and mixes the available predictors accordingly. Under a converged five-seed protocol with pooled paired statistics, the gate is the only method that significantly outperforms calibrated probability averaging (wildfire conflicts: +0.051, p = 0.0001) and improves on end-to-end cross-view fusion on both the conflict subset (+0.072, p < 1e-4) and the full test set (+0.018, p < 1e-4), while never degrading performance on the panoramic datasets. A controlled field-of-view intervention establishes the mechanism causally: cropping hurricane panoramas to building-centered views doubles the fusion benefit on conflict cases, whereas geometrically identical random crops do not, in 6/6 dataset-seed pairs. Finally, the spatial density of cross-view conflicts predicts tile-level damage without any labels (wildfire Spearman r = 0.615, p = 0.001), while single-view uncertainty density does not. Together these results replace the question "does fusion help?" with an answerable one — "which view should be trusted, where, and why?" — and provide a deployable, interpretable answer.

**Keywords:** building damage assessment; cross-view fusion; street-view imagery; remote sensing; reliability estimation; disaster response

## 1. Introduction

Rapid, building-level damage assessment is one of the most operationally consequential tasks in disaster response. Overhead imagery — satellite or aerial — provides synoptic coverage within hours of an event and has become the backbone of standardized damage mapping efforts (Gupta et al., 2019; Weber and Kané, 2020). Ground-level imagery, whether crowdsourced street-view panoramas or official inspection photographs, provides the complementary evidence that overhead sensors physically cannot: facade condition, structural detail, and debris context (Zhai and Peng, 2020; Mabon, 2016; Xue et al., 2024; Yang et al., 2025). It is therefore natural that a growing body of work fuses the two views (Xiao et al., 2023; Li et al., 2025; Yang et al., 2026a), and the standard finding is encouraging: cross-view fusion improves average accuracy over either single view.

We argue that this average-accuracy framing conceals both the real value and the real limitation of cross-view fusion. Most samples in a damage dataset are easy: both views agree, and fusion adds little that either view could not provide alone. The operationally interesting cases are the *conflicts* — samples on which independently trained single-view models disagree. Conflicts concentrate precisely where one sensor is blind: a facade intact from above but gutted at street level, a roof destroyed above vegetation that hides it from the road. In this paper we make conflict cases the primary axis of evaluation, and we ask three questions that the field has not answered:

1. **How much information do current fusion methods leave unused on conflict cases?** We quantify this with an *oracle single-view gap*: the difference between an oracle that picks whichever single view is correct per sample and the best available method. Across three disasters and two ground-view regimes, this gap is 0.37–0.41 accuracy under converged training — several times larger than the improvement any fusion method delivers, and stable across seeds, architectures, and calibration.

2. **Can the missing information be recovered — and by what mechanism?** We show that per-sample view reliability is largely predictable from a simple, physically interpretable quantity: whether the ground-level image actually sees the target structure. We operationalize this with building-visibility features extracted by a frozen semantic segmentation model, combine them with calibrated per-view confidences and cross-view disagreement, and train a *linear* gate that mixes the available predictors per sample. The gate is deployable on top of any existing pair of single-view models, adds no deep training, and its coefficients read as an interpretable decision rule: *trust the street view when it is confident and the building is centered in the frame; otherwise lean on the overhead view.*

3. **Is the regime effect causal?** Prior cross-view studies observe that fusion benefits differ across datasets, but datasets differ simultaneously in disaster type, geography, sensors, and labels, so the observation is unattributable. We resolve this with a controlled intervention: we crop hurricane panoramas to building-centered 90° views (using segmentation-derived centroids) and to geometrically identical randomly-centered views, retrain, and compare. Building-centered cropping moves the fusion benefit toward the wildfire (property-centric) regime — doubling the oracle-gap closure on Hurricane Milton — while random cropping does not, in 6/6 dataset-seed pairs. To our knowledge this is the first controlled manipulation of the observation regime in cross-view disaster assessment.

Our experiments yield one additional finding with a direct operational payoff: the *spatial density* of cross-view conflicts is an unsupervised damage map. Aggregated to geographic tiles, conflict density predicts ground-truth damage (wildfire Spearman r = 0.615, p = 0.001) while single-view uncertainty density does not — on wildfire imagery it even anti-correlates with damage. A conflict-density raster can be computed within hours of acquiring paired imagery, before any labels exist.

The contributions of this paper are:

- **A conflict-aware evaluation protocol with an oracle-gap metric** that quantifies how much view-reliability information a fusion system fails to exploit, applied across three disasters under a converged five-seed protocol with pooled paired statistics (Section 4.2, 4.8).
- **A visibility-conditioned reliability gate** — linear, interpretable, and training-free beyond a small validation fit — that is the only method in our study to significantly beat calibrated late fusion, with gains concentrated exactly where the mechanism predicts (Section 4.5, 5.3).
- **A causal field-of-view intervention** establishing that street-view target alignment, not disaster type or dataset idiosyncrasy, controls the value of cross-view fusion (Section 4.6, 5.5).
- **Conflict density as an unsupervised spatial damage signal**, validated against a single-view uncertainty control and spatial autocorrelation diagnostics (Section 4.7, 5.6).
- **A calibration decomposition and negative result** the field should know: once single-view models are trained to convergence and calibrated, simple probability averaging matches or exceeds end-to-end learned fusion; the advantage of learning lies in asymmetric, reliability-aware arbitration rather than in fusion per se (Section 5.2).

All code, evaluation protocols, and derived data are released with the paper.

## 2. Related work

### 2.1 Single-view building damage assessment

Satellite-based damage assessment is the most mature line, anchored by the xBD dataset and the xView2 challenge (Gupta et al., 2019), with subsequent work on multi-temporal fusion (Weber and Kané, 2020) and multi-hazard generalization. Overhead imagery, however, is structurally limited to roof-visible damage. Ground-level assessment has developed in parallel, from early qualitative use of Google Street View for recovery monitoring (Mabon, 2016) and post-hurricane damage audit (Zhai and Peng, 2020) to learned assessment from street-view imagery with structured metadata (Xue et al., 2024) and bi-temporal hyperlocal assessment with pre-trained vision models (Yang et al., 2025). Each view alone has a well-documented blind side; our oracle-gap analysis quantifies how much that complementarity is worth on the cases where the views disagree.

### 2.2 Cross-view fusion for disaster mapping

Cross-view learning connecting street-level and overhead imagery has been applied to disaster geolocalization and mapping (Li et al., 2025; Yin et al., 2026; Li et al., 2026), damage classification with fusion transformers (Xiao et al., 2023), flood vulnerability integration (Xing et al., 2023), and multimodal arbitration frameworks (Yang et al., 2026a). The closest prior work is CVDisaster (Li et al., 2025), which pairs Mapillary street-view panoramas with VHR satellite imagery over Sanibel Island for Hurricane Ian and reports that cross-view classification outperforms either single view. Our Hurricane Ian data is the released CVIAN pairing from that work, which makes our results directly comparable (Section 5.7); the key difference is that CVDisaster — like the field at large — evaluates overall accuracy with a symmetric fusion architecture, whereas we evaluate conflict cases, quantify the oracle gap, and gate asymmetrically. Generative approaches that synthesize one view from the other (Yang et al., 2026b) are complementary: they address missing views, whereas we address contradictory views.

### 2.3 Reliability, calibration, and adaptive fusion

Modern neural networks are systematically miscalibrated (Guo et al., 2017), and any comparison between learned fusion and simple ensembling is confounded unless calibration is controlled. We therefore fit per-view temperatures on validation data and re-evaluate all fusion baselines on calibrated probabilities — a decomposition that, to our knowledge, no prior cross-view disaster study performs. Adaptive weighting of experts dates to mixture-of-experts models (Jacobs et al., 1991); our gate is deliberately the minimal member of this family — a linear model over interpretable features — because the point is not architectural novelty but the demonstration that *physically grounded visibility features carry the reliability signal*. This mirrors a broader movement in GeoAI toward mechanism-oriented, interpretable pipelines for disaster response (Zou et al., 2023; Raj et al., 2025).

### 2.4 Evaluation practice

Benchmark practice in damage assessment reports overall accuracy or F1 on random splits (Gupta et al., 2019; Li et al., 2025). Two aspects of our protocol depart from this norm. First, spatially co-located samples share objects and scenery; random splits therefore leak. We split by object identity throughout and document a leakage-induced inflation in our own early experiments. Second, "significant in k of N seeds" reporting — common where multi-seed results appear at all — has no clear inferential meaning. We instead pool across seeds with the test sample as the unit of analysis and seeds as repeated measures, using sign-flip permutation tests and percentile bootstrap intervals (Section 4.8).

## 3. Study areas and data

We use three datasets spanning two disaster types and, critically, two ground-view observation regimes (Table 1).

**Table 1.** Datasets. "Regime" describes the ground-view imaging style: property-centric images are aimed at a specific building; panoramic images are 360° equirectangular captures of the surrounding scene.

| | Eaton wildfire (Altadena) | Hurricane Ian (CVIAN) | Hurricane Milton |
| --- | --- | --- | --- |
| Event | January 2025 wildfire, Altadena, CA | September 2022 hurricane, Sanibel Island, FL | October 2024 hurricane, FL |
| Ground view | Official damage-inspection photographs (CAL FIRE DINS) | Mapillary 360° street-view panoramas (1024×512) | 360° street-view panoramas (1024×512) |
| Overhead view | Post-event VHR aerial/satellite tiles | VHR satellite (27 cm, post-event) | VHR satellite tiles |
| Regime | Property-centric | Panoramic | Panoramic |
| Classes (ordinal) | no/trace damage; damaged-repairable; destroyed | minor; moderate; severe | mild; moderate; severe |
| Pairs (train/val/test) | ~6,500 / 1,966 / 1,984 | 3,248 / 573 / 300 | 1,740 / 307 / 254 |
| Georeference | Yes | No (not released) | Yes |

**Eaton wildfire (Altadena).** Building-level inspection photographs collected under the CAL FIRE Damage Inspection (DINS) program after the January 2025 Eaton fire, paired with post-event overhead tiles by geolocation. The native six-level damage scale is mapped to an ordinal three-class task (no/trace damage = No Damage + Affected 1–9%; damaged-repairable = Minor 10–25% + Major 26–50%; destroyed = >50%); inaccessible parcels are excluded. The native-label six-class setting is retained as an audit and yields the same qualitative conclusions.

**Hurricane Ian (CVIAN).** The released cross-view pairing from CVDisaster (Li et al., 2025): Mapillary street-view panoramas over Sanibel Island paired with 27 cm post-event satellite imagery and labeled on a three-level damage-perception scale by GIS and disaster experts. The release we use contains 4,121 pairs (minor 1,407 / moderate 1,538 / severe 1,176), larger than the 1,135 manually labeled images reported in the paper; we state this explicitly and use the release as distributed. The release carries no coordinates, which precludes spatial analysis on this dataset (Section 5.6) and makes object-grouped splitting the only available leakage control.

**Hurricane Milton.** A street-view/satellite pairing over Florida following Hurricane Milton (October 2024) with three-level damage labels, constructed analogously to CVIAN.

**Splits and leakage control.** All splits are grouped by object identifier so that images of the same building or panorama location never span train and test. This matters: in early experiments a convenience split of the Ian data inflated the remote-only baseline by over 6 F1 points relative to the grouped split. All results in this paper use grouped splits.

## 4. Methods

Figure 1 gives an overview of the full pipeline.

**Figure 1.** Framework overview. Independently trained single-view models define the conflict cases (10–33% of test samples); the oracle single-view gap (0.37–0.41 accuracy) quantifies the reliability information symmetric fusion leaves unused; a linear visibility-conditioned gate recovers a significant fraction of it; the field-of-view intervention establishes the mechanism causally; and the spatial density of conflicts yields a label-free damage map. The street/overhead pair shown is a real Eaton-fire conflict case.

### 4.1 Base models

For each dataset we train four models under an identical protocol: `street_only` and `remote_only` (ResNet-18 encoders (He et al., 2016) with a linear classification head), `concat` (both encoders, feature concatenation, MLP head), and `crossview` (both encoders with a learned interaction head; the end-to-end fusion reference). Training uses class-balanced cross-entropy, standard augmentation, image size 224, batch size 64, AdamW, up to 15 epochs with patience-4 early stopping on validation macro-F1, and five seeds (42, 123, 456, 789, 1011). A backbone sweep (ResNet-50, CLIP ViT-B/32 (Radford et al., 2021), DINOv2 ViT-S/14 (Oquab et al., 2023)) confirmed that supervised CNN encoders remain strongest on this task and that no conclusion below is an artifact of the ResNet-18 choice.

### 4.2 Conflict-aware evaluation and the oracle gap

Let $s(x)$ and $r(x)$ be the class predictions of the two single-view models. The **conflict subset** of a test set is $\{x : s(x) \neq r(x)\}$. Conflict rates are substantial: 9.6% (Altadena), 33.2% (Ian), 25.7% (Milton) under the converged protocol. For any method $m$ we report accuracy on the conflict subset alongside full-test metrics, and we summarize its use of view-reliability information with the **oracle-gap closure**

$$\mathrm{closure}(m) = \frac{\mathrm{acc}_{\mathrm{conf}}(m) - \mathrm{acc}_{\mathrm{conf}}(\mathrm{best single view})}{\mathrm{acc}_{\mathrm{conf}}(\mathrm{oracle}) - \mathrm{acc}_{\mathrm{conf}}(\mathrm{best single view})},$$

where the oracle counts a conflict sample correct if *either* single view is correct. The oracle is a per-sample view-selection upper bound: it uses no information beyond which of the two existing models to trust.

### 4.3 Calibration decomposition

Fusion comparisons are confounded by calibration: averaging miscalibrated probabilities can win or lose for reasons unrelated to information content. We fit a scalar temperature per view on validation predictions by NLL minimization (Guo et al., 2017) and re-evaluate every non-trained fusion baseline (probability averaging, logit averaging, confidence voting) on calibrated probabilities. Temperature scaling leaves single-view argmax unchanged, so the conflict subset is identical before and after — the decomposition isolates calibration effects exactly.

### 4.4 Street-view visibility features

For every street-view image we extract building-visibility features with a frozen SegFormer-B0 segmentation model trained on ADE20K (Xie et al., 2021): the building pixel ratio, the building ratio within a centered crop of half the image dimensions, their difference (a centering signal), and the normalized distance of the building-mask centroid from the image center. These features are computed once per image, require no disaster-specific training, and cost a single segmentation pass.

### 4.5 The visibility-conditioned reliability gate

The gate predicts, per sample, how to mix the available predictors. Its feature vector concatenates (i) the four visibility features; (ii) calibrated per-view confidence and entropy for both views; (iii) the confidence gap; (iv) the Jensen–Shannon divergence between the two calibrated distributions; and (v) a binary disagreement flag — eleven features in total, all available at inference time. Two variants are evaluated: a **two-view gate** producing a scalar weight $w(x) \in [0,1]$ over street/remote probabilities, and a **three-view gate** producing softmax weights over street, remote, and crossview probabilities. The gate — linear, or a small MLP for comparison — is fit on the validation split by minimizing the NLL of the gated mixture; features are z-scored with validation statistics. The linear variant is our headline method: it matches the MLP throughout, and its coefficients are directly interpretable. We evaluate in-domain gates, pairwise zero-shot cross-disaster transfer (gate fit on one dataset's validation split, applied unchanged to another dataset), and leave-one-disaster-out pooled training with per-dataset feature standardization.

### 4.6 Causal field-of-view intervention

The regime hypothesis — that fusion benefit is controlled by street-view target alignment — cannot be established by comparing datasets that differ in disaster, geography, sensors, and labels. We therefore manipulate alignment *within* the two panoramic datasets. Each 1024×512 equirectangular panorama is cropped to a 256×256 window (90° horizontal field of view, horizon band) in two ways: **building-centered**, at the circular-mean horizontal centroid of the SegFormer building mask (center-crop fallback when no building pixels exist: 7.1% of Ian, 3.1% of Milton panoramas); and **random**, at a per-sample reproducible uniform azimuth. The two variants are geometrically identical — same size, same projection, same information budget — differing only in whether the window is aimed at the built structure. We retrain `street_only` and `crossview` on each variant (three seeds, protocol matched to the main suite), reuse the unchanged `remote_only` models, and compare conflict gains. The regime hypothesis predicts building-centered crops increase the fusion benefit and random crops do not; the prediction was registered in the repository before results were computed.

### 4.7 Conflict density as an unsupervised damage map

For spatial analysis we average per-sample street and remote probabilities across seeds, aggregate to geographic tiles (a lat/lon grid, or native overhead tile groupings), and compute per tile: the **hard conflict density** (fraction of samples with argmax disagreement), the **soft conflict density** (mean Jensen–Shannon divergence), and a **single-view uncertainty control** (mean prediction entropy). Each signal is correlated (Spearman) with the tile's mean ordinal damage label. The uncertainty control is the critical comparison: if conflict density were repackaged model uncertainty, the control would perform equivalently. Spatial autocorrelation of damage and of conflict density is quantified with Moran's I (Moran, 1950) under inverse-distance weights with permutation p-values.

### 4.8 Statistical protocol

All headline numbers come from the converged five-seed suite. For method comparisons we avoid per-seed significance counting. Instead, for each comparison and each sample we average the paired correctness difference over the seeds in which the sample qualifies (for conflict-scope tests, the seeds in which the two single-view models disagree on it), yielding one delta per sample; we test H0: E[delta] = 0 with a two-sided sign-flip permutation test (20,000 permutations) and report percentile bootstrap 95% intervals. Per-seed McNemar tests (McNemar, 1947) and bootstrap intervals are retained as supplementary diagnostics.

## 5. Results

### 5.1 Overall performance

**Table 2.** Main results (test set; mean ± std over five seeds). Conflict accuracy is computed on the per-seed conflict subset. The gate row uses the three-view linear gate.

| Dataset | Method | Accuracy | Macro-F1 | Conflict acc. |
| --- | --- | --- | --- | --- |
| Altadena | street_only | 0.931 ± 0.017 | 0.722 | 0.486 |
| Altadena | remote_only | 0.929 ± 0.024 | 0.700 | 0.475 |
| Altadena | concat | 0.930 ± 0.011 | 0.706 | 0.674 |
| Altadena | crossview | 0.940 ± 0.011 | 0.716 | 0.699 |
| Altadena | calibrated prob. avg. | 0.954 ± 0.014 | 0.722 | 0.732 |
| Altadena | **reliability gate (linear)** | **0.959** | — | **0.768** |
| Ian | street_only | 0.703 ± 0.020 | 0.706 | 0.529 |
| Ian | remote_only | 0.648 ± 0.015 | 0.650 | 0.367 |
| Ian | concat | 0.709 ± 0.028 | 0.709 | 0.588 |
| Ian | crossview | 0.735 ± 0.015 | 0.739 | 0.624 |
| Ian | calibrated prob. avg. | 0.731 ± 0.005 | 0.735 | 0.617 |
| Ian | reliability gate (linear) | 0.731 | — | 0.618 |
| Milton | street_only | 0.759 ± 0.017 | 0.765 | 0.557 |
| Milton | remote_only | 0.721 ± 0.012 | 0.726 | 0.407 |
| Milton | concat | 0.766 ± 0.012 | 0.771 | 0.622 |
| Milton | crossview | 0.776 ± 0.015 | 0.781 | 0.649 |
| Milton | calibrated prob. avg. | 0.776 ± 0.015 | 0.782 | 0.626 |
| Milton | reliability gate (linear) | 0.777 | — | 0.631 |

Three patterns organize the table. First, cross-view fusion beats both single views on conflict cases on all three datasets, and the pooled tests make this significant everywhere (crossview − street on conflicts: Altadena +0.246, p < 1e-4; Ian +0.078, p = 0.013; Milton +0.141, p = 0.006; margins over remote_only are larger still). Second, once single-view models are converged and calibrated, *simple probability averaging is a strong baseline*: on Altadena it exceeds vanilla crossview (0.954 vs 0.940 accuracy). Third, the reliability gate sits at the top of the wildfire columns by a clear margin and matches the best method elsewhere. Ordinal metrics tell the same story with an operational accent: on Altadena the fused methods cut two-step errors (no-damage ↔ destroyed confusions, the costliest triage failure) from ~1.0% (single views) to 0.4% (crossview), with QWK rising from 0.947–0.950 to 0.963–0.967.

### 5.2 The oracle gap and the calibration decomposition

**Table 3.** Conflict-subset oracle analysis (five-seed means). Gap = oracle − best single view. Closure per Eq. (1).

| Dataset | Conflict rate | Best single view | Oracle | Gap | Closure: crossview | Closure: calib. avg. | Closure: gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Altadena | 9.6% | 0.551 | 0.961 | 0.410 | 0.346 | 0.440 | **0.524** |
| Ian | 33.2% | 0.529 | 0.896 | 0.367 | 0.252 | 0.239 | 0.234 |
| Milton | 25.7% | 0.557 | 0.964 | 0.407 | 0.222 | 0.170 | 0.181 |

**Figure 2.** The oracle single-view gap. For each dataset, the bar spans from the best single view to the oracle on conflict cases; dots place each method within that span. No method reaches the midpoint of the span on the panoramic datasets; the gate comes closest on the wildfire.

The oracle gap is the paper's motivating quantity (Figure 2): on every dataset, simply knowing *which existing model to trust per sample* would add 0.37–0.41 conflict accuracy — several times the improvement any fusion method achieves. The calibration decomposition rules out the mundane explanation. Per-view temperatures are moderate (1.0–1.75) and expected calibration errors drop from 0.02–0.14 to 0.01–0.06 after scaling, but calibrated averaging closes at most 44% of the gap (Altadena) and under 24% elsewhere. The unexploited information is not a calibration artifact; it requires knowing *when* each view is reliable, which is what the gate supplies.

This analysis also surfaces a negative result the field should absorb: under converged training, end-to-end learned fusion (crossview) does not reliably beat calibrated probability averaging (Altadena conflicts: +0.034 in the pooled test, p = 0.073; Ian and Milton: no significant difference). Claims that learned fusion beats simple aggregation, common in the literature, may be artifacts of undertrained or miscalibrated single-view baselines — ours were, in our own three-epoch pilot.

### 5.3 The reliability gate

**Table 4.** Pooled paired tests (sign-flip permutation; sample as unit, seeds as repeated measures), conflict scope unless noted.

| Dataset | Comparison | Mean delta | 95% CI | p |
| --- | --- | --- | --- | --- |
| Altadena | gate − crossview (conflicts) | +0.072 | [+0.046, +0.097] | < 1e-4 |
| Altadena | gate − crossview (full test) | +0.018 | [+0.014, +0.023] | < 1e-4 |
| Altadena | gate − calibrated prob. avg. (conflicts) | +0.051 | [+0.026, +0.077] | 0.0001 |
| Ian | gate − crossview (conflicts) | +0.005 | [−0.021, +0.033] | 0.70 |
| Ian | gate − calibrated prob. avg. (conflicts) | −0.012 | [−0.060, +0.036] | 0.64 |
| Milton | gate − crossview (conflicts) | −0.015 | [−0.068, +0.036] | 0.57 |
| Milton | gate − calibrated prob. avg. (conflicts) | −0.011 | [−0.086, +0.062] | 0.76 |

**Figure 3.** (a) Conflict-case accuracy by method (mean ± std over five seeds; dashed line = oracle). (b) Linear gate coefficients on the wildfire dataset (mean ± std over seeds); positive coefficients shift trust toward the street view.

On the property-centric dataset the gate delivers exactly what the mechanism predicts (Figure 3a): +0.072 conflict accuracy over end-to-end fusion and +0.051 over the strongest baseline, both decisively significant, and a 52% oracle-gap closure — the only method in our study above one half. On the panoramic datasets, where street views rarely see the target structure (mean building pixel ratio 0.027 on Ian vs 0.119 on Altadena), the gate has little visibility signal to exploit and performs at parity — never significantly worse. The asymmetry is not a weakness of the method; it *is* the finding: reliability gating pays where ground-level evidence is target-aligned, and degrades gracefully to baseline behavior where it is not.

The linear gate's coefficients (Figure 3b) are interpretable and stable in sign across seeds on Altadena: street-view entropy carries the largest negative weight (an uncertain street view cedes to the overhead view), the centered-building-ratio feature is positive (a building filling the center of the frame earns trust), and the confidence gap is positive. The learned rule is the one a human analyst would state: *trust the street view when it is confident and actually looking at the building.*

**Cross-disaster transfer.** A linear gate fit on Ian transfers zero-shot to Altadena with 0.27 oracle-gap closure (78% of crossview's in-domain closure there), and an Altadena-fit gate retains just over half of the in-domain gate's advantage on Ian; transfers involving Milton, the smallest dataset, are weaker, and naive pooled training across disasters underperforms due to the order-of-magnitude difference in visibility-feature scales between regimes (per-dataset standardization recovers part of it). The confidence-side features transfer; the visibility features are regime-scaled. A universal gate is therefore plausible but not yet established — we state this as an open problem.

### 5.4 Causal field-of-view intervention

**Table 5.** FOV intervention (three seeds; remote models unchanged). Conflict gain = crossview − best single view on conflicts; closure per Eq. (1).

| Dataset | Street-view variant | Street acc. | Conflict gain | Gap closure |
| --- | --- | --- | --- | --- |
| Ian | original panorama | 0.709 | +0.064 | 0.166 |
| Ian | building-centered crop | 0.648 | +0.109 | 0.269 |
| Ian | random crop (control) | 0.574 | +0.042 | 0.072 |
| Milton | original panorama | 0.740 | +0.067 | 0.177 |
| Milton | building-centered crop | 0.693 | +0.149 | 0.365 |
| Milton | random crop (control) | 0.654 | +0.055 | 0.128 |

The intervention delivers a clean causal verdict. Building-centered cropping raises the conflict gain on both hurricanes — on Milton the gap closure doubles (0.177 → 0.365), reaching two-thirds of the wildfire reference level — while random cropping with identical geometry lowers it. The building-minus-random contrast is positive in 6/6 dataset-seed pairs. Two details sharpen the attribution. First, building-centered cropping *lowers* street-only accuracy (panoramic context is genuinely informative for the overall task), yet *raises* the fusion benefit: the gain does not come from a stronger street model but from the two views becoming alignable — they now attend to the same object. Second, the random-crop control rules out resolution, field of view, and cropping artifacts as explanations. Street-view target alignment is the causal variable behind the regime effect, which retroactively grounds the gate's visibility features as causal rather than merely correlated.

**Figure 4.** The field-of-view intervention. (a) One Hurricane Ian panorama with the two geometrically identical 90° crop windows: building-centered and random-azimuth; the resulting crops are shown at right — the building is present in one and absent in the other. (b) Conflict gain by crop variant (bars: three-seed means; dots: individual seeds). Building-centered cropping raises the fusion benefit on both hurricanes; the random control does not.

### 5.5 Conflict density as an unsupervised damage map

**Table 6.** Tile-level Spearman correlation with mean ordinal damage (five-seed averaged probabilities).

| Dataset | Tiling | Tiles | Conflict density r (p) | Uncertainty control r (p) |
| --- | --- | --- | --- | --- |
| Altadena | overhead tiles | 25 | **0.615 (0.001)** (hard) | −0.373 to −0.521 (negative) |
| Altadena | 0.01° grid | 39 | 0.473 (0.002) (hard) | −0.482 (0.002) |
| Milton | 0.0015° grid | 27 | 0.289 (0.14) (soft) | ~0 (n.s.) |
| Ian | — | — | not possible: release has no georeference | — |

**Figure 5.** Conflict density as a label-free damage map (Eaton wildfire, 0.01° grid). (a) Ground-truth tile damage. (b) Tile conflict density computed without labels. (c) Their rank correlation. (d) The single-view uncertainty control, which anti-correlates with damage. Point size in (c, d) reflects tile sample count.

On the wildfire, hard conflict density is a strong label-free predictor of tile damage (Figure 5), and it strengthens under converged models (r = 0.545 → 0.615). The uncertainty control fails in the most instructive way possible: it *anti-correlates* with damage, because burned-to-the-ground parcels are easy, confident classifications. Whatever conflict density measures, it is not model uncertainty — it is the spatial signature of the two sensors disagreeing about the world, which concentrates where damage disrupts the normal correspondence between facade and roof evidence. On Milton the soft-conflict signal was significant under the three-epoch pilot (r = 0.485, p = 0.010) but weakens below significance under converged models (r = 0.289), as better single-view models disagree less; we therefore present the wildfire case as the robust result and the hurricane case as regime-sensitive supporting evidence. Damage on Altadena is spatially autocorrelated (Moran's I = 0.27, p = 0.0001), so tile correlations partially reflect spatial structure; conflict density itself shows no significant autocorrelation.

### 5.6 External comparison

Our Ian dataset is the released CVIAN pairing, enabling a direct reading against CVDisaster (Li et al., 2025): their CGCViT cross-view model reports 77.96 overall accuracy (street-only 74.50, satellite-only 67.07) under a random 5:5 split with a 20M-parameter encoder trained for 100 epochs; our crossview reaches 73.5 ± 1.5 (street 70.3, remote 64.8) under an object-grouped split with ResNet-18 and early stopping. The single-view ranking (street > satellite) and the cross-view advantage replicate exactly across independent codebases; the absolute offset is consistent with our stricter anti-leakage split and smaller encoder, and the release's lack of coordinates prevents ruling out spatial leakage in random splits of this data. The contributions are orthogonal: CVDisaster contributes geolocalization and a fusion architecture; conflict-aware evaluation, the oracle gap, reliability gating, the causal intervention, and conflict-density mapping appear in neither that work nor, to our knowledge, elsewhere in the cross-view disaster literature.

**Figure 6.** Qualitative conflict cases. Rows: street view, overhead patch, per-model predictions with ground truth, and building-segmentation overlay, for wildfire and hurricane conflicts resolved by fusion and one failure case.

## 6. Discussion

**What the oracle gap means for the field.** The 0.37–0.41 oracle gap is, in effect, a measured upper bound on the value of solving per-sample view selection — and it dwarfs the gains that architectural innovation in symmetric fusion has produced. We read this as a re-prioritization: the productive question in cross-view disaster assessment is not "how should features be fused?" but "when should each view be believed?". The gate closes half of the gap on property-centric imagery with eleven interpretable features and a linear model; the remaining half is a concrete, quantified target for future work — richer visibility descriptors (occlusion, viewing distance, image quality), pre-event reference imagery, and structured metadata are the obvious candidates.

**Deployment.** The gate adds negligible cost to an operational pipeline: one frozen segmentation pass per street image, a temperature fit, and a linear model fit on validation predictions. It requires no retraining of existing models and no architectural access to them — only their output probabilities — making it applicable retroactively to deployed systems. Its interpretability matters in emergency-management contexts where automated triage must be auditable: the decision rule can be stated in one sentence, and per-sample weights explain each arbitration.

**When simple fusion is enough.** Our calibration decomposition offers practical guidance: where ground imagery is panoramic and rarely target-aligned (crowdsourced street view in vegetated or flooded areas), calibrated probability averaging captures most of the achievable benefit, and investment in learned fusion is hard to justify. Where ground imagery is property-centric (inspection programs, insurance photography, parcel-aimed capture), reliability gating pays substantially. Because the causal driver is alignment rather than disaster type, capture policy is itself an intervention: aiming cameras at structures — or cropping panoramas toward detected buildings, as in our intervention — increases the value of every downstream fusion component.

**Limitations.** (i) The study covers three datasets, two disaster types, and one wildfire; the property-centric regime is represented by a single (large) dataset. (ii) Milton is small (2,301 pairs), which limits gate training there and widens its intervals. (iii) The CVIAN release lacks coordinates, precluding spatial analysis and forcing group-based leakage control on that dataset. (iv) Visibility features come from a generic ADE20K segmenter; it does not identify *the* target building, only built-structure visibility, and our earlier analyses show the per-sample correlation between raw building ratio and fusion gain is weak — the gate's value comes from combining visibility with confidence features. (v) The conflict-density signal weakens on Milton under converged models; its robustness outside the wildfire case needs replication on additional georeferenced disasters. (vi) Cross-disaster gate transfer is partial; a universal gate remains open.

## 7. Conclusion

Cross-view fusion for disaster damage assessment has been evaluated, and built, as if the two views deserved equal trust everywhere. Measured on the cases where the views disagree, that assumption leaves 0.37–0.41 accuracy on the table across three real disasters. We showed that a large fraction of this information is recoverable with a linear, interpretable, visibility-conditioned reliability gate; that the mechanism — street-view target alignment — is causal, via a controlled field-of-view intervention; and that the disagreement signal itself, aggregated spatially, constitutes a label-free damage map that plain uncertainty cannot replicate. The framework converts "does fusion help?" into "which view should be trusted, where, and why?" — a question that our results show is both answerable and worth answering.

## Acknowledgements

The authors used AI-based assistance for experiment automation, grammar checking, and editorial compression, reviewed all content, and take full responsibility for the work. Supported by the Texas A&M University Environment and Sustainability Initiative (ESI) through the Environment and Sustainability Graduate Fellow Award.

## Data and code availability

Code, evaluation protocols, derived features, and result tables are available at https://github.com/rayford295/CrossViewGate. The Eaton wildfire imagery derives from the CAL FIRE DINS program; CVIAN is distributed by Li et al. (2025); Milton pairing details are documented in the repository.

## References

Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q., 2017. On calibration of modern neural networks. In: Proceedings of the 34th International Conference on Machine Learning, pp. 1321–1330.

Gupta, R., Goodman, B., Patel, N., Hosfelt, R., Sajeev, S., Heim, E., Doshi, J., Lucas, K., Choset, H., Gaston, M., 2019. Creating xBD: A dataset for assessing building damage from satellite imagery. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops.

He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep residual learning for image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778.

Jacobs, R.A., Jordan, M.I., Nowlan, S.J., Hinton, G.E., 1991. Adaptive mixtures of local experts. Neural Computation 3(1), 79–87.

Li, H., Deuser, F., Yin, W., Luo, X., et al., 2025. Cross-view geolocalization and disaster mapping with street-view and VHR satellite imagery: A case study of Hurricane Ian. ISPRS Journal of Photogrammetry and Remote Sensing.

Li, H., Deuser, F., Yin, W., Knoblauch, S., Zhao, W., Biljecki, F., Xue, Y., Huang, W., 2026. Towards generative location awareness for disaster response: A probabilistic cross-view approach. ISPRS Journal of Photogrammetry and Remote Sensing 237, 130–145.

Mabon, L., 2016. Charting disaster recovery via Google Street View: A social science perspective on challenges raised by the Fukushima nuclear disaster. International Journal of Disaster Risk Science 7(2), 175–185.

McNemar, Q., 1947. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika 12(2), 153–157.

Moran, P.A.P., 1950. Notes on continuous stochastic phenomena. Biometrika 37(1/2), 17–23.

Oquab, M., Darcet, T., Moutakanni, T., et al., 2023. DINOv2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193.

Radford, A., Kim, J.W., Hallacy, C., et al., 2021. Learning transferable visual models from natural language supervision. In: Proceedings of the 38th International Conference on Machine Learning, pp. 8748–8763.

Raj, A., Shetgaonkar, A., Arora, L., Pradhan, D., et al., 2025. AI and generative AI transforming disaster management: A survey of damage assessment and response techniques. arXiv preprint.

Weber, E., Kané, H., 2020. Building disaster damage assessment in satellite imagery with multi-temporal fusion. arXiv preprint arXiv:2004.05525.

Xiao, W., Su, J., Chen, Y., Cao, G., 2023. Cross-scale-guided fusion transformer for disaster assessment using satellite imagery and social media text. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.

Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M., Luo, P., 2021. SegFormer: Simple and efficient design for semantic segmentation with transformers. Advances in Neural Information Processing Systems 34.

Xing, Z., Yang, S., Zan, X., Dong, X., et al., 2023. Flood vulnerability assessment of urban buildings based on integrating high-resolution remote sensing and street view images. Sustainable Cities and Society.

Xue, Z., Zhang, X., Prevatt, D.O., Bridge, J., Xu, S., Zhao, X., 2024. Post-hurricane building damage assessment using street-view imagery and structured data. arXiv preprint arXiv:2404.07399.

Yang, Y., Zou, L., Zhou, B., Li, D., et al., 2025. Hyperlocal disaster damage assessment using bi-temporal street-view imagery and pre-trained vision models. International Journal of Applied Earth Observation and Geoinformation.

Yang, Y., Zou, L., Gong, W., Fu, K., et al., 2026a. DamageArbiter: A CLIP-enhanced multimodal arbitration framework for hurricane damage assessment. [Journal to be confirmed].

Yang, Y., Zou, L., Jepson, W., 2026b. Satellite-to-street: Synthesizing post-disaster views from satellite imagery via generative models. [Journal to be confirmed].

Yin, W., Deuser, F., Liu, Z., Wei, J., et al., 2026. Triple-objective cross-view geolocalization of disaster-related VGI: The case of Hurricane Ian. ISPRS Journal of Photogrammetry and Remote Sensing.

Zhai, W., Peng, Z.-R., 2020. Damage assessment using Google Street View: Evidence from Hurricane Michael in Mexico Beach, Florida. Applied Geography 123, 102252.

Zou, L., Mostafavi, A., Zhou, B., Lin, B., et al., 2023. GeoAI for disaster response. In: Handbook of Geospatial Artificial Intelligence. CRC Press.
