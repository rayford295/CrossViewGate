# CVIAN sequence utility v1 — post-evaluation integrity addendum

## Final claim status

The frozen one-shot evaluation is a **hash-chain-verified, consumed
development/exploratory NO-GO**. It is not a confirmatory or external-event
result. The machine-generated primary report remains byte-for-byte unchanged
at [`cvian_sequence_utility_v1.md`](cvian_sequence_utility_v1.md); this addendum
records a completion-verifier defect discovered only after scoring.

No model, policy, threshold, decision row, metric, bootstrap result, or report
was recomputed by reloading the prospective cache. The prospective holdout was
not scored a second time.

## Primary result

At fixed `k=3`, positive contrasts would favor `utility_regression`. The
registered selector instead had higher dependency-component macro cost and a
higher severe-miss rate than both required baselines.

| Baseline | Baseline − utility cost | Paired component 95% CI | Five-seed requirement | Severe-miss requirement | Decision |
|---|---:|---:|:---:|:---:|:---:|
| `farthest` | −0.050935 | [−0.114026, −0.000229] | failed | failed | NO-GO |
| `max_building_privileged` | −0.025325 | [−0.054784, 0.001634] | failed | failed | NO-GO |

The learned policy beat farthest in only one of five seeds and the privileged
building heuristic in two of five. Its mean component-macro operational cost
was 1.803692, compared with 1.752756 for farthest and 1.778367 for the building
heuristic. The result therefore fails independently of the post-evaluation
verification issue.

## Verification defect and impact

The stored evaluation commitment was written on Windows with CRLF newlines.
Its actual raw SHA-256 is
`edb9044d231866ccd5f5b06a4d1bc0bd6f44d6d5ef925f131bfa7ca997d37009`.
The evaluated runner's completed-run branch re-rendered the same JSON with LF
newlines and incorrectly expected
`c3011406c3ff6c5fb3ed1da34dec73489f1101987d90ab018e2b60548a1c9b22`.
Consequently, a second invocation intended only to verify existing artifacts
returned a false commitment-mismatch error.

This defect is confined to the completed-run verifier:

- the parsed stored commitment and the freshly reconstructed commitment were
  semantically identical;
- both recomputed to evaluation fingerprint
  `0d5bddc735dd819dc379c212f7082745aeeceeadc2667c8fd5144f57fe7f9b42`;
- the start sentinel, all five seed completions, aggregate completion, and
  registry completion consistently reference the actual raw CRLF commitment
  hash;
- all 29 frozen input/source hashes matched;
- all metric, decision, aggregate JSON/CSV, and report hashes matched;
- independent recomputation from the saved decision tables reproduced the 45
  metric rows, both primary contrasts, and both 10,000-resample bootstrap
  intervals.

The commitment must not be newline-normalized or rewritten because that would
break every downstream raw-hash link. The frozen evaluated runner is likewise
left unchanged so its recorded source hash remains verifiable.

## Independent verification

Use the post-evaluation verifier, not the evaluated runner's `--phase
evaluate` branch:

```powershell
python scripts/verify_cvian_sequence_utility_completion.py
```

The verifier validates the semantic commitment fingerprint, actual-byte hash
links, frozen input/source hashes, the complete seed → aggregate → registry
DAG, and the primary analysis directly from saved decision tables. It never
loads a prospective NPZ, loads a model, or executes a policy.

## Scientific consequence

The registered rule states that a defect found after test scoring consumes the
result and downgrades it. Even though this defect did not affect scoring, the
conservative claim is therefore **exploratory within-CVIAN NO-GO**. The result
is sufficient for the project decision not to continue tuning the current
label-cost selector on CVIAN. Any future positive active-evidence claim requires
a new untouched event holdout and must not reuse this test for policy, budget,
threshold, or GO-rule selection.
