# Milton zero-shot active-view sensitivity v1: integrity verification

## Decision

The registered Milton sensitivity completed once and passed an independent
post-score verification with `verified=true`. No scoring defect was found and
no downgrade beyond the pre-registered **sensitivity-only, non-confirmatory**
claim is required. The run was not rescored.

The frozen result remains
[`active_view_milton_zero_shot_v1.md`](active_view_milton_zero_shot_v1.md).
That file is hash-bound to the completion ledger and was not edited after
scoring.

## Frozen chain

| Artifact | SHA-256 |
| --- | --- |
| sensitivity config | `c9457127d156aae00770d36196bdc1526934ca30bf5269978f75cadf088df463` |
| scorer | `142e0dc167482e33df7f34868bdbe265ce578b20f5cad7aeab9b59fa73e886b8` |
| CVIAN relative-fit summary | `7b6355f12527fae6086211d2c0e42ab8e06259053d82c3f957b437a560b6f851` |
| frozen result report | `40b3d143a727c923da6f2b89269d890040f854e255ca7bc2c1bb6ebee41c6262` |
| evaluation completion | `6049ef7e95f478d1a549fd3fcbf01ac556d97c76a25d8b49ecbe8ac62e0e34f7` |
| global completion registry | `e339e550d9cbd4d787e30547f06f55995d2de7c6f851574d97cb0e33a540aa19` |
| independent verification JSON | `1800510564a59ea0f0f48667c77ea22352fb7c7c37df749989762cca45fd0ae9` |

The commitment SHA-256 is
`efcd277beec87fefa729f7fb4d66dd3d04b3edfe51229853da33d1f84185d41c`.
Both local and global started sentinels have SHA-256
`ab92aff93fa109b3a34db5baa4b3446cb97f84210a454bb05143774637f1eebb`.

## Independent recomputation

[`verify_milton_zero_shot_active_view_sensitivity.py`](../../scripts/verify_milton_zero_shot_active_view_sensitivity.py)
does not import the scorer or PyTorch. It treats model artifacts and Milton
forward-cache NPZ files as opaque bytes and verifies their hashes without
deserializing them. Starting from the registered transfer manifest and emitted
decision CSVs, it independently verified:

- 39 opaque input artifacts and all commitment/completion/registry edges;
- 1,707 manifest rows, 259 dependency groups, 57 spatial blocks, and 11 joint
  dependency--spatial connected components;
- 518,928 decision rows per seed and 2,594,640 in total;
- local-to-physical action mapping, fixed farthest/clockwise paths, and
  2,184,960 exact `SeedSequence` random-trajectory replays;
- probability normalization, cost-sensitive Bayes decisions, realized class
  cost, the fixed two-view acquisition cost, and severe-miss indicators;
- the registered averaging chain
  `2,594,640 -> 477,960 -> 59,745 -> 11,949`; and
- both dependency-group and spatial-block point estimands, plus all 10,000
  primary joint-component and secondary direct-unit bootstrap intervals.

The independently recomputed aggregate JSON and CSV were semantically
identical to the registered outputs. The verifier performed no model inference,
no PT/NPZ semantic load, and no scoring rerun.

## Verification command

The JSON ledger is written exclusively. The command below is for a completed
run before that verification path exists; it intentionally refuses to
overwrite the verified artifact already recorded above.

```powershell
python scripts/verify_milton_zero_shot_active_view_sensitivity.py `
  --json-output outputs/cvian_sequence_active_v2/milton_zero_shot_sensitivity/independent_completion_verification.json
python -m pytest -q tests/test_milton_zero_shot_completion_verifier.py
```

The directed verifier suite reports **8 passed**.
