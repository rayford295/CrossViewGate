# Per-Sample Evidence Schema (P0.2)

`scripts/train_reliability_gate.py` now writes an auditable `p0.2-v1` evidence
CSV for every in-domain linear-gate dataset/seed. The file preserves the legacy
prediction columns and adds:

- calibrated class probabilities for street, remote, and crossview models;
- complete two-view and three-view gate weights and mixture probabilities;
- building visibility/centering features;
- per-view confidence and entropy;
- street/remote, street/crossview, and remote/crossview JS divergence;
- hard conflict and each source/gate selected prediction;
- target, sample id, object id, latitude, longitude, WKT location, tile id,
  group id, panorama sequence, spatial block, and protocol role.

Strict-role evidence also carries `event_id` and `gate_artifact_id`. The latter
fingerprints temperatures, gate-fit arrays and sample ids, normalization
statistics, and both gate models; all three roles must contain the same value.

Probability columns are generated dynamically as
`{street|remote|crossview|gate2|gate3}_probability_<class_index>`, so the schema
supports native class counts rather than assuming binary output. Manifest
metadata is joined one-to-one after canonicalizing sample ids; duplicate ids
fail instead of multiplying evidence rows.

The default dataset-to-manifest mapping points to the main split directories.
It can be overridden explicitly:

```powershell
python scripts/train_reliability_gate.py `
  --multiseed-root outputs/multiseed_v2 `
  --split-manifests "ian_original=data/splits/ian_hurricane_original"
```

For a strict P0.7 chain, provide role manifests created by
`build_risk_protocol_manifests.py`:

```powershell
python scripts/train_reliability_gate.py `
  --multiseed-root outputs/multiseed_ian_spatial_v1 `
  --datasets ian_original `
  --risk-protocol-manifests "ian_original=data/splits/ian_hurricane_risk_protocol"
```

Temperature fitting, feature normalization, and gate fitting then use
`gate_fit` only. The same frozen two-/three-view gates write
`<dataset>_seed<seed>_{gate_fit|risk_calibration|final_test}.csv`; the final-test
file is also copied to the historical `<dataset>_seed<seed>.csv` alias.

Files are written under `<output-dir>/predictions/`. Old evidence remains tied
to the historical split and cannot be mixed with repaired-split roles.
