# Model reference

`reference_model.py` contains the KAN layers, gated residual blocks, expert aggregation, climate and land-cover embeddings, time embeddings, and residual head. It supplies model-reference code without trained weights.

Install `requirements-model.txt` to use the reference components. Construct `ModelConfig` with the architecture constants and input dimensions appropriate to the source experiment. See [model and parameters](../docs/model_and_parameters.md).

`reconstruct_sm` expects a residual in physical soil-moisture units. Invert the fitted target transform first. Its optional clipping argument defaults to no clipping; the independent evaluation utility never clips predictions.
