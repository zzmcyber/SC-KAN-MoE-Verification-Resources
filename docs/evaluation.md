# Evaluation definitions

Let y be observed soil moisture, p its paired prediction, and n the number of pairs. Bias is mean(p − y), RMSE is sqrt(mean((p − y)²)), and R² is 1 − sum((p − y)²) / sum((y − mean(y))²).

Pearson's r is the centered cross-product divided by the product of the two centered Euclidean norms. KGE is 1 − sqrt((r − 1)² + (alpha − 1)² + (beta − 1)²), where alpha is the prediction standard deviation divided by the observation standard deviation, and beta is their mean ratio. Both standard deviations use the same sample convention.

The utility supports pooled and per-site evaluation. It preserves every supplied pair, applies no prediction clipping, and rejects missing or non-finite values and repeated sample IDs. A zero observation mean makes KGE undefined. Constant observed series make R², Pearson's r and KGE undefined; constant predicted series make Pearson's r and KGE undefined. For a single pair, only RMSE and Bias are defined. Undefined values are emitted as JSON `null`.

Numerically constant series are detected with a range tolerance of 64 times double-precision machine epsilon multiplied by max(1, the largest absolute value). This prevents roundoff from producing enormous finite scores for constant observations.
