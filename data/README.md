# Auxiliary input subset

`auxiliary_subset/` contains 3,675 real auxiliary records from 12 anonymized sites. All sample-level tables share the unique key `sample_id` and anonymous station key `site_id`.

The released tables provide a subset of the auxiliary inputs used in the study for data inspection and verification; they are not intended to constitute a complete inference input package.

| File | Rows | Fields | Content |
| --- | ---: | ---: | --- |
| `sar_features.csv` | 3,675 | 17 | Backscatter, incidence angles, SAR validity and quality flags |
| `meteorological_features.csv` | 3,675 | 13 | Temperature, radiation, evaporation, wind, snow, freeze-thaw flag |
| `precipitation_features.csv` | 3,675 | 9 | Six accumulation windows and elapsed time since rain |
| `vegetation_features.csv` | 3,675 | 6 | NDVI, EVI, LAI, FPAR |
| `terrain_features.csv` | 3,675 | 3 | Terrain elevation |
| `temporal_matching.csv` | 3,675 | 7 | Reference and SAR timestamps, SAR and vegetation matching intervals |
| `site_summary.csv` | 12 | 4 | Auxiliary-record count and anonymous time span per site |
| `combined/sc_kan_moe_auxiliary_features.csv` | 3,675 | 45 | One-to-one join of all sample-level tables |

Read the combined CSV directly or join the individual sample tables on `sample_id`. The site summary joins on `site_id`. The [field dictionary](../metadata/feature_dictionary.csv) defines every released column, and the [manifest](../metadata/release_manifest.json) records row counts and SHA256 hashes.

Scientific values retain their original numerical representation. VV/VH are in dB, while single-pixel and 3 × 3 neighborhood backscatter use linear power. Time translation preserves differences within each site. Anonymous dates do not represent the original calendar or season.

The general evaluation utility uses a separate, user-supplied observation–prediction CSV. The auxiliary files are input examples, not an evaluation-pair dataset.
