# P0.4 DINS Field Join and Provenance

Status: implemented, dry run completed against the official service, and a new derived artifact directory produced under an explicit `--write`; the original manifest was neither modified in place nor overwritten.

## Data sources and snapshot boundary

Fields come from the CAL FIRE [Eaton/Palisades expanded DINS layer](https://services.arcgis.com/HdtZMT2FmI4wPzTM/arcgis/rest/services/Eaton_Palisades_DINS/FeatureServer/2), queried with the fixed condition `INCIDENTNAME = 'Eaton'`. The URLs in the existing attachment manifest come from a different CAL FIRE service, the [Eaton public attachment layer](https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/DINS_2025_Eaton_Public_View/FeatureServer/0). The two are distinct ArcGIS services.

The 2026-07-10 dry run observed:

| Item | Expanded field layer | Public attachment layer |
| --- | ---: | ---: |
| Eaton records | 18,422 | 18,428 |
| Official metadata `lastEditDate` | 2025-02-20 20:13:40.705 UTC | 2025-06-23 19:24:09.658 UTC |
| Number of fields | 44 | 4 |
| Fields with a published coded-value domain | 0 | 2 |

The expanded layer publishes no field-level coded-value domain for its 44 fields; the program records `domain: null` faithfully and does not promote observed values into an "official domain". The public layer publishes coded-value domains for `DAMAGE` and `STRUCTURETYPE` with 6 and 16 codes respectively; that reference metadata is kept separately.

## Join strategy

The current Eaton manifest carries only the public-layer `objectid`, attachment id, category, and coordinates, and shares no stable business key with the expanded layer. The dry run therefore uses an explicit spatial join and never compares `OBJECTID` values across the two services.

The program tries unique, non-empty stable keys in this order:

1. `GLOBALID`;
2. `INCIDENTNUM + APN + STRUCTURETYPE`;
3. `APN + STRUCTURETYPE`;
4. `SITEADDRESS + STRUCTURETYPE`.

A business key counts only when the source side has exactly one candidate. `OBJECTID`, `OID`, and `FID` are rejected even when specified explicitly on the command line. Only without a unique business key does the program fall back to haversine nearest-neighbour matching: default tolerance 10 m, ambiguity margin 1 m; when the second-nearest candidate lies within 1 m of the nearest, the row is flagged `spatial_ambiguous` and no DINS fields are filled.

Every output row carries the following audit information: join status/method/key, nearest and second-nearest distances, tolerance, ambiguity margin, candidate count, business-key ambiguity, distance warnings, field-service URL, and service update time. The expanded-layer `OBJECTID` is retained only as `dins_source_objectid` provenance; the snapshot stores a canonical-JSON SHA-256 and the input manifest a file SHA-256.

## Official-data dry-run result

The input is the 19,780 attachment rows of the local `dataset_index.csv`. The official query paged by `OBJECTID ASC` at 2,000 records per page over 10 pages; the pre-query count and the final GeoJSON feature count were both 18,422.

| Metric | Result |
| --- | ---: |
| Manifest attachment rows | 19,780 |
| Unambiguously matched rows | 19,776 (99.9798%) |
| Unmatched rows | 4 |
| Spatially ambiguous rows | 0 |
| Unique expanded records used by attachments | 18,411 |
| Expanded records with no attachment | 11 |
| Expanded records matched to several attachment rows | 1,177 |
| Match distance p50 / p95 / p99 | <0.001 m / <0.001 m / <0.001 m |
| Maximum match distance | 3.992 m |

The 4 unmatched rows stay empty; they are not forced through a relaxed tolerance or an assumed same-number `OBJECTID`. Before any formal write they should be reviewed as a manual provenance queue.

An explicit write run on the same day reproduced identical counts, join statuses, and distance statistics, and saved six derived products under the local `Eaton_Fire_attachments_index_output/dins_field_join/` directory: the official GeoJSON snapshot, the full service metadata, field domains, field statistics, `eaton_manifest_with_dins.csv`, and `join_provenance.json`. That run used no `--overwrite` and did not change the input `dataset_index.csv`.

Selected field-quality statistics show why null and `Unknown` must be reported separately:

| Field | Null/blank | `Unknown` / all rows | Semantics |
| --- | ---: | ---: | --- |
| `EAVES` | 0.00% | 37.40% | visually observable candidate |
| `WINDOWPANE` | 0.00% | 29.84% | visually observable candidate |
| `VENTSCREEN` | 0.00% | 27.66% | visually observable candidate |
| `PATIOCOVERCARPORT` | 0.00% | 13.57% | visually observable candidate |
| `DEFENSIVEACTIONS` | 58.87% | 37.25% | inspector assessment/observation |
| `WHEREFIRESTARTEDONSTRUCTURE` | 95.34% | 0.51% | conditional inspector observation |
| `WHATDIDFIRESTARTFROM` | 95.34% | 0.48% | conditional inspector observation |
| `BATTALION` / `FIRENAME` | 100.00% | 0.00% | administrative |

The definitions are fixed: JSON null, NaN, empty strings, and whitespace-only strings count as null/blank; a value counts as Unknown only when, after trimming, it equals `Unknown` case-insensitively.

## Semantic boundary

Every added value originates from the DINS inspector/service record. The code groups fields as:

- `visually_observable_candidate`: structure type, roof, eaves, vent screens, exterior walls, windows, patio covers, and similar; usable for visual fact assessment only when the relevant part is visible in the given image;
- `inspector_assessment_or_observation`: damage, fire origin/source, defensive actions, and count fields;
- `administrative_or_parcel`: incident, address, APN, valuation, year built, and location fields;
- `source_identifier`: ArcGIS source identifiers.

A "visually observable candidate" is not image-level visibility ground truth. The pipeline creates no `recommended_action` or `rationale` fields and never presents such fields as existing ground truth.

## Usage

The default command reads the official service and prints a summary without creating an output directory:

```powershell
python scripts/fetch_dins_fields.py `
  --manifest-csv <path>\dataset_index.csv
```

A fully offline replay uses a local GeoJSON and metadata; this mode makes no network requests:

```powershell
python scripts/fetch_dins_fields.py `
  --manifest-csv <path>\dataset_index.csv `
  --source-geojson <snapshot>\dins_fields.geojson `
  --source-metadata <snapshot>\service_metadata.json `
  --reference-metadata <snapshot>\public_service_metadata.json
```

Only an explicit `--write` creates a new derived directory; an existing artifact is refused by default and requires an additional explicit `--overwrite`:

```powershell
python scripts/fetch_dins_fields.py `
  --manifest-csv <path>\dataset_index.csv `
  --output-dir data\dins\eaton_field_join `
  --write
```

Write mode produces `dins_fields.geojson`, the full `service_metadata.json`, `field_domains.json`, `field_statistics.json`, the derived `eaton_manifest_with_dins.csv`, and `join_provenance.json`. The original manifest is never modified in place.

## Verification

```text
python -m pytest tests/test_dins_join.py -q
.......                                                                  [100%]
7 passed
```

The tests use local fixtures only and cover stable-key precedence, refusal to join on same-number `OBJECTID`, spatial distance and ambiguity, null versus Unknown, pagination count integrity, the offline CLI dry run, and the default refusal to overwrite.
