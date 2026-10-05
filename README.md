# SC-KAN-MoE Verification Resources

**Paper:** Site-State-Constrained KAN-MoE Residual Fusion Network for Soil Moisture Estimation from Sentinel-1 and Multi-Source Environmental Observations

This repository provides verification resources for SC-KAN-MoE, including anonymized remote-sensing and environmental input examples, model-reference code, documented parameter settings, data-field definitions, and independent evaluation utilities.

The manuscript study uses 158 stations; this repository releases auxiliary records from 12 anonymized study sites as a verification subset. The subset contains 3,675 real project records with anonymized identifiers and privacy-shifted timestamps. Scientific values and relative time intervals are preserved.

In-situ soil-moisture observations and associated provider metadata are not redistributed because the applicable source-data terms restrict onward distribution.

Record-level model predictions and trained weights are not included in this release.

## Auxiliary inputs

The [data directory](data/README.md) contains project-processed auxiliary inputs for SAR, meteorology, precipitation, vegetation, terrain, and temporal matching. The [combined CSV](data/auxiliary_subset/combined/sc_kan_moe_auxiliary_features.csv) joins all 45 sample-level fields. See the [field dictionary](metadata/feature_dictionary.csv) and [data sources](metadata/data_sources.md).

## Model and parameters

The [reference implementation](model/README.md) includes KAN layers, contextual and time embeddings, top-k expert aggregation, and residual prediction. The reported configuration uses six experts and top-3 routing. [Parameter resources](docs/model_and_parameters.md) distinguish Reported settings, Reference settings, and Candidate ranges.

## General evaluation utility

The evaluation utility accepts user-supplied observation–prediction pairs and reports RMSE, R², Pearson's r, KGE, and Bias.

```sh
python -m pip install -r requirements.txt
python evaluation/evaluate_predictions.py YOUR_PAIRS.csv
python evaluation/evaluate_predictions.py YOUR_PAIRS.csv --by-site
```

Supply a separate CSV with `sample_id`, `site_id`, `observed_sm`, and `predicted_sm`, with soil moisture in m3/m3. The utility rejects duplicate or non-finite pairs, handles undefined metrics, and preserves predictions without clipping. See [metric definitions](docs/evaluation.md).

## Integrity checks

```sh
python -m pip install -r requirements-model.txt
python tests/run_checks.py
```

Checks validate the released tables, numeric ranges, metadata, hashes, model parameters, and evaluation behavior. GitHub Actions runs the same checks and rejects skipped tests.

Repository-authored code and documentation use the MIT license. Third-party auxiliary inputs retain their provider terms. See [data usage](DATA_USAGE.md) and the [resource guide](docs/resource_guide.md).
