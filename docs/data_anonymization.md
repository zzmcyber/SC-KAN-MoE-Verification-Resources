# Data anonymization

Sites are selected after data-completeness and consistency checks. A fixed random draw selects 12 sites. The public IDs are `SITE_001` through `SITE_012`, with sequential sample IDs such as `SITE_001_S0001`. Original identities, row mappings, and time offsets remain outside version control.

Public records omit coordinates, network and location categories, original identifiers, image and orbit identifiers, original calendars, in-situ observation values and histories, observation-provider metadata, and record-level predictions.

For each site, all absolute auxiliary timestamps use the same translation:

`anonymous_time = site_anchor + (original_time - original_first_reference_time)`.

`site_anchor_mapping` assigns distinct anchors from 2001–2010 to the sorted public namespace `SITE_001` through `SITE_012`. The mapping is independent of source-file order, input iteration order, or whether a subset of public IDs is requested. A series can extend beyond the anchor years. Scientific values, sample order, and within-site time intervals are preserved.

`anonymous_reference_time` is the auxiliary-record reference time. `anonymous_sar_time` is the matched SAR acquisition time. `s1_time_diff_minutes` is the signed difference, SAR minus reference. The duplicate SAR-difference column and its alternative unit/absolute-value forms are omitted. NDVI and LAI matching intervals and precipitation durations retain their source values.

These measures remove direct identifiers. They do not establish immunity to linkage with external datasets.
