# Model and parameters

The reference implementation provides KAN layers, contextual embeddings, a residual prediction head, and top-k expert aggregation. The residual head predicts a transformed soil-moisture increment. Apply the fitted inverse target transformation before adding that increment to antecedent soil moisture with `reconstruct_sm`.

## Reported settings

The manuscript reports six experts with three selected experts per sample. These settings are recorded in `configs/documented_settings.yaml`.

## Reference settings

The architecture uses cubic B-splines, a grid interval of [-3, 3], a spline initialization scale of 0.1 divided by the grid size, and a 64-unit regression hidden layer. Month indices use 1 through 12 with a 13-entry embedding; calendar quarters use indices 0 through 3. These constants are listed in `configs/reference_architecture_settings.yaml`.

The complete `ModelConfig` also takes continuous-input width, category vocabulary sizes, context and time embedding widths, expert hidden width, grid size and dropout. These are explicit configuration inputs. Category vocabularies and fitted preprocessing belong to their corresponding training fold. The model structure retains contextual embeddings; public auxiliary tables omit the identifying category values.

## Candidate ranges

`configs/hyperparameter_search_space.yaml` lists the candidate ranges for antecedent history, experts, top-k aggregation, embedding widths, expert hidden width, grid size, dropout, learning rate, weight decay and batch size. The embedding and hidden-width steps preserve the documented candidate grids. Candidate ranges are configuration guidance rather than selected per-fold parameters.

The reference permits `top_k` values 2 and 3 in its candidate list. Its validation rejects `top_k=1` and values exceeding the expert count. The reference implementation applies top-k weighted aggregation over expert outputs: it computes the expert outputs, selects the highest routing weights, normalizes the selected weights and aggregates their corresponding outputs.

Antecedent-history candidates span one to five earlier valid observations from the same station. Every antecedent must precede both the target observation and the SAR acquisition used in that sample. Sample identity must remain distinct from its target. Continuous transforms and categorical vocabularies fit the training split only.
