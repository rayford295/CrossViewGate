# CVIAN legacy vs repaired spatial split

The converged 5-seed comparison is now complete. The protocols use different
held-out samples (legacy n=300; repaired n=415), so seed-matched differences
measure stability of the protocol change rather than paired per-sample test
effects.

| mode | protocol | accuracy | macro-F1 | severe recall |
| --- | --- | ---: | ---: | ---: |
| concat | legacy | 0.7093 ± 0.0283 | 0.7092 ± 0.0283 | 0.7500 ± 0.0339 |
| concat | repaired spatial | 0.6578 ± 0.0184 | 0.6599 ± 0.0219 | 0.7793 ± 0.0895 |
| crossview | legacy | 0.7347 ± 0.0145 | 0.7391 ± 0.0151 | 0.7440 ± 0.0261 |
| crossview | repaired spatial | 0.6366 ± 0.0383 | 0.6416 ± 0.0401 | 0.7966 ± 0.0505 |
| remote only | legacy | 0.6480 ± 0.0146 | 0.6497 ± 0.0174 | 0.6980 ± 0.0444 |
| remote only | repaired spatial | 0.5802 ± 0.0094 | 0.5825 ± 0.0079 | 0.6879 ± 0.0526 |
| street only | legacy | 0.7027 ± 0.0202 | 0.7060 ± 0.0216 | 0.7020 ± 0.0415 |
| street only | repaired spatial | 0.6467 ± 0.0121 | 0.6500 ± 0.0131 | 0.8000 ± 0.0522 |

## Repaired minus legacy

Intervals are deterministic seed-level bootstrap intervals over the five
matched training seeds. With only five seeds they are descriptive, not a
substitute for event-level replication.

| mode | metric | mean difference | seed-bootstrap 95% interval |
| --- | --- | ---: | ---: |
| concat | accuracy | -0.0515 | [-0.0818, -0.0323] |
| concat | macro-F1 | -0.0493 | [-0.0825, -0.0282] |
| concat | severe recall | 0.0293 | [-0.0519, 0.1317] |
| crossview | accuracy | -0.0980 | [-0.1217, -0.0787] |
| crossview | macro-F1 | -0.0974 | [-0.1227, -0.0764] |
| crossview | severe recall | 0.0526 | [0.0134, 0.0896] |
| remote only | accuracy | -0.0678 | [-0.0815, -0.0540] |
| remote only | macro-F1 | -0.0672 | [-0.0834, -0.0510] |
| remote only | severe recall | -0.0101 | [-0.0417, 0.0190] |
| street only | accuracy | -0.0559 | [-0.0670, -0.0430] |
| street only | macro-F1 | -0.0561 | [-0.0684, -0.0418] |
| street only | severe recall | 0.0980 | [0.0361, 0.1650] |

The leaked split overstated generalization for every input mode. The largest
macro-F1 correction is crossview (-0.0974), and repaired crossview has much
higher seed variability. Severe recall did not fall in the same way because
balanced training shifted the error profile toward fewer severe misses. The
result supports keeping the repaired spatial protocol and treating the legacy
headline as historical only; it does not support claiming that the current
crossview architecture is robust to spatial holdout.

Machine-readable rows, summaries, and differences are generated under
`outputs/analysis/ian_split_protocol_comparison/` by
`scripts/compare_ian_split_protocols.py`.
