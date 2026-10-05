# Resource guide

1. Open `data/auxiliary_subset/combined/sc_kan_moe_auxiliary_features.csv` to inspect all released sample-level inputs, or use the individual category tables.
2. Read `metadata/feature_dictionary.csv` for field definitions and `metadata/data_sources.md` for product attribution.
3. Run `python tests/run_checks.py` to check the public files, model reference, and metric implementation.
4. Read `model/reference_model.py` and the configuration files to inspect SC-KAN-MoE components and parameter categories.
5. Evaluate separate, user-supplied observation–prediction pairs with `evaluation/evaluate_predictions.py`; use `--by-site` for station metrics.

The model reference predicts a transformed residual. Apply the fitted inverse target transformation before adding the physical residual to antecedent soil moisture with `reconstruct_sm`.

The architecture retains climate, land-cover, and calendar embeddings. Public auxiliary tables omit identifying categories and original calendars. Privacy-shifted timestamps preserve relative timing and must not be used to reconstruct the model's original month or season inputs. The released tables support inspection of auxiliary inputs and data handling.
