# Speaker Script: When Satellite and Street Views Disagree

GeoSearch '26 (5th ACM SIGSPATIAL Workshop on Searching and Mining Large Collections of Geospatial Data), Riverside, CA, 3 November 2026. Lightning talk.

The same text is stored in the speaker notes of `CrossViewGate_GeoSearch2026_slides.pptx`. Timings are cumulative at a normal speaking pace (about 9.5 minutes for all 17 slides).

## Five-minute cut

For a strict five-minute slot, present slides 1, 5, 6, 7, 9, 11, 12, 13, 14, 16 and 17 and skip the rest (background slides 2 to 4 are folded into slide 5; the definitions on slide 8 and the table on slide 10 are summarized by slides 11 and 12; slide 15 is a backup for questions). Trim each remaining note to its first paragraph.

## Full script

### Slide 1: Title  (0:00)

Good afternoon. I am Yifan Yang from the Department of Geography at Texas A&M, and this is joint work with my advisor Lei Zou.

The title is a question we kept running into while building damage-assessment pipelines: what should we do when the satellite view and the street view disagree about the same building? This talk argues that those disagreements are not noise to average away. They are the most informative records we have.

### Slide 2: Background: satellite imagery  (0:30)

After a disaster, satellite imagery is the first thing we get. It covers a whole region within hours, and it is the basis of standardized damage mapping such as xBD.

But from above you mostly see roofs. A roof that looks intact does not tell you what happened to the walls underneath.

### Slide 3: Background: street-view imagery  (0:50)

Ground-level imagery, whether crowdsourced street-view panoramas or official inspection photographs, shows the facades and the debris that overhead sensors cannot fully detect.

These four cases from our earlier hurricane work make the point: for the same structure, the street-view model and the satellite model give very different damage scores. Case A is level-5 damage from the ground and reads as zero from above. The views are complementary, and sometimes only one of them sees the damage.

### Slide 4: Background: Eaton fire DINS map  (1:15)

One of our settings is the January 2025 Eaton fire in Altadena, California. CAL FIRE's Damage Inspection program visited every structure, photographed it, and assigned a six-level damage label.

That gives us something rare: a ground photograph that is deliberately aimed at one building, paired with a post-event satellite tile of the same parcel. Keep that word, "aimed", in mind; it becomes the key variable later.

### Slide 5: The problem and research questions  (1:40)

Here is the problem. Fusion papers usually report that combining the two views raises average accuracy. That is true, but it says little about whether the complementary evidence is actually used, because for most buildings the two views simply agree.

So we focus on the conflict cases: the test samples on which two independently trained single-view models disagree. On a conflict, at least one view carries evidence the other lacks, whether damage is visible from only one side, vegetation hides the building, or the views are misaligned.

We ask four questions. One, how much complementary headroom do fusion methods leave on conflicts? Two, can a simple, readable gate recover part of it? Three, does aiming the ground view at the building change the value of fusion? And four, does the spatial density of conflicts track damage, with no labels at all?

### Slide 6: Data  (2:30)

We use three paired street and overhead collections. They span two hazards and, more importantly, two ground-view regimes.

Eaton is property-centric: the DINS photographs are aimed at a building. Hurricane Ian on Sanibel Island and Hurricane Milton in Florida use 360-degree panoramas from a vehicle, which are not aimed at anything in particular.

Conflict rates on the test sets are 9.6 percent on Eaton, 33 percent on Ian and 26 percent on Milton. So conflicts are a minority, but a large one on the panoramic data.

One caveat for the record: the released Ian split has sequence overlap between train and test. A spatially blocked protocol lowers macro-F1 for every model by 5 to 10 points, so the Ian numbers should be read as in-distribution results.

### Slide 7: Method overview  (3:10)

The pipeline has four parts. Independently trained street and remote models define the conflict cases. An oracle that always picks the correct single view tells us the headroom. A linear fusion gate decides, per sample, how much weight each view gets, from building-visibility, confidence and disagreement features. And aggregating conflicts over space gives a conflict-density map that we test as a damage indicator.

### Slide 8: Method: conflict cases and the oracle gap  (3:30)

Some definitions, quickly. For each dataset we train four models under one protocol and five seeds: street-only and remote-only ResNet-18s, a concatenation model, and a cross-view model with a learned interaction head, which is our end-to-end fusion reference.

The conflict subset is simply the samples where the two single-view predictions differ.

The oracle counts a conflict as correct if either single view is correct. It measures how much you would gain from perfect hard selection between two fixed models. We report each method's conflict accuracy and its closure of that oracle gap, from zero at the best single view to one at the oracle.

All training-free fusion baselines run on temperature-calibrated probabilities, so miscalibration does not distort the averaging comparisons.

### Slide 9: Method: visibility-conditioned fusion gate  (4:10)

The gate uses eleven features per sample. Four describe building visibility in the street image, from a frozen SegFormer: the building pixel ratio, the ratio inside a centered crop, their difference as a centering signal, and the distance of the building centroid from the image center. Seven describe the predictions: calibrated confidence and entropy per view, the confidence gap, the Jensen-Shannon divergence, and a disagreement flag.

The gate itself is a mixture of experts with linear logits, fit on the validation split. In the two-expert version there is a single logit, so each coefficient reads directly as "shift weight toward the street view" or "toward the overhead view". A small MLP in place of the linear logits did not help, so we kept the readable version.

### Slide 10: Results table  (4:50)

This is the main table. Three things to read off it.

First, end-to-end fusion beats the street model on conflicts on every dataset, and the gains are significant.

Second, once the single-view models are converged and calibrated, plain probability averaging is a strong baseline. On Eaton it beats end-to-end fusion on both overall and conflict accuracy.

Third, and this is the point of the paper: the oracle dwarfs all of these differences. Picking the right single view per conflict would add 0.37 to 0.48 over the better single view, and still 0.19 to 0.32 over the best fusion method we evaluated.

### Slide 11: RQ1: the oracle gap  (5:30)

The same result as a picture. Each grey bar spans from the best single view to the oracle on conflict cases, and the dots place each method. On all three datasets the dots sit in the left third of the bar. Current fusion, including the learned one, leaves most of the view-specific evidence unused.

### Slide 12: RQ2: the fusion gate  (5:50)

Can a simple gate recover part of that? On Eaton, where photographs are aimed at buildings, yes. The gate gains 7 points of conflict accuracy over end-to-end fusion and 5 points over calibrated averaging, both highly significant, with a closure of 0.59, the highest in the table.

On the panoramic hurricane data the gate is statistically indistinguishable from end-to-end fusion. The reason is on the right: the building is rarely prominent in a panorama, with a mean building pixel ratio of under 3 percent versus 12 percent on Eaton, so the visibility features carry little signal.

The two-expert coefficients are the interpretable part. The largest weight is building centering, followed by the confidence gap and street confidence, while the raw building pixel ratio is negative. In words: trust the street view when it is confident and the building sits in the middle of the frame, not merely when the image is full of built-up area.

### Slide 13: RQ3: field-of-view intervention  (6:40)

If framing matters, we should be able to change it. We take each hurricane panorama and cut one 90-degree window: either aimed at the building azimuth from the segmenter, or at a reproducible random azimuth. Same geometry, different pointing. Then we retrain the street and fusion models.

Building-centered cropping raises the conflict gain of fusion on both hurricanes: from 0.06 to 0.11 on Ian and from 0.07 to 0.15 on Milton. Random crops of identical size do not, and the building-minus-random contrast is positive in all six dataset-seed pairs.

Note that cropping actually lowers street-only accuracy. So the gain is not a stronger street model; it is that the street view now carries evidence the overhead view lacks. Building-oriented framing is a driver of which view carries the evidence.

### Slide 14: RQ4: conflict density  (7:30)

Finally, a use for conflicts that needs no labels once the models are trained. We aggregate predictions to 0.01-degree tiles over the Eaton fire and compute the fraction of samples on which the two views disagree.

Panel (a) is labeled damage, panel (b) is conflict density, and they track each other with a Spearman correlation of 0.6 over 25 tiles. The control in panel (d) is single-view uncertainty, and it anti-correlates with damage, because parcels burned to the ground are easy, confident classifications. Disagreement between views is a different signal from uncertainty within a view.

Two caveats: damage is spatially autocorrelated, so the nominal p-value overstates the evidence, and on Milton the signal is weaker and not significant, because better single-view models disagree less.

### Slide 15: What conflict cases look like  (8:15)

A few actual conflicts. In the wildfire rows a garage or facade is gutted while the roof still reads intact from above, so the remote model says "no damage" with near-certainty. In row three, end-to-end fusion followed the overhead view and missed as well. That is exactly the headroom the oracle measures.

In the hurricane rows the building is a small patch of the panorama and debris along the road drives the street prediction. There is little for a visibility feature to work with.

One honest note: we measure prediction complementarity, not which component was damaged. Some conflicts come from occlusion, misalignment or label ambiguity rather than from genuinely view-specific damage.

### Slide 16: Takeaways  (8:50)

Three takeaways. Conflict cases make the value of complementary views measurable, and by that measure current fusion leaves most view-specific evidence unused.

Which view carries the evidence depends on hazard and capture setting. On building-aimed photographs a cheap, auditable gate helps: one frozen segmentation pass and a linear fit. On panoramas, calibrated averaging or end-to-end fusion is enough, and pointing the camera at the structure raises the value of fusion.

And spatially aggregated disagreement is a label-free damage indicator.

So mining conflicts turns the question "does fusion help?" into "which view carries the evidence, where, and why?" Code, protocols and the full result tables are in the repository behind the QR code.

### Slide 17: Q&A  (9:20)

Thank you. I am happy to take questions, and the QR code goes to the GitHub repository with code, data pointers and the paper.
