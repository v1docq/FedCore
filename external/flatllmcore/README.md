# Local calibration absorption contracts

The MLP path implements `nystrom_least_squares_pseudoinverse`: for actual
intermediate activations Z entering `down_proj`, C = Z.T @ Z and retained
channel indices I, the new row-weight matrix is
`W_down @ C[:, I] @ pinv(C[I, I])`. This follows the Nyström reconstruction
orientation in [FLAT-LLM Appendix C](https://arxiv.org/html/2505.23966v4#A3),
with a pseudoinverse for dependent/zero channels instead of the paper's invertible
selected matrix. Ridge leverage scores are measured from C, never random.
Gate/up weights and biases select the same channels; down bias is preserved.

`tolerance` is a hard lower bound on retained calibration activation squared
norm. `sparsity_ratio` is the maximum retained fraction. An incompatible request
raises before modifying the affected weights. Metadata reports actual method,
rank, tolerance and measured retained energy. These are calibration contracts;
model accuracy, speed and physical energy require separate measurements.

The attention path absorbs a headwise PCA basis into V and O weights and V bias.
Calibration uses actual activations entering `o_proj`, aggregating query heads
that share a KV head. The selected uniform rank must meet every KV-group tolerance.
The existing transformer forward/cache patch is experimental and does not establish
parity for every Transformers family/version or cache implementation. The standalone
random importance selector is blocked by the capability registry.

Use the independent model returned by `compressor.model` after compression.
The constructor preserves the supplied model. Tests use local duplicate channels,
identity/zero Gram matrices and actual CPU operations; no large-model benchmark or
model-download result is claimed.
