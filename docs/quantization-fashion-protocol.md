# Fashion-MNIST local CPU experiment

This protocol is separate from the synthetic teacher-fidelity pilot. The old pilot files remain unchanged.
All numerical commands, including tests, must be launched by the root orchestrator beneath
`fedcore/experiments/quantization_pilot_guard.py`, with the original shared absolute `LIMITS.json`.
No command starts another process, downloads data, installs packages, uses CUDA or contacts a cluster.

The root supplies the four canonical Fashion-MNIST gzip files in `<run-root>/data/raw` and a flat
`SHA256.json` mapping their standard basenames to SHA256. The worker verifies SHA256 and the
official MD5 before parsing. Preparation never opens either `t10k` file.

The finite manifest freezes M0 (784→256→128→10, ReLU, seed 20261027), preprocessing,
20 Adam epochs, checkpoint selection by stop CE, the minimum stop accuracy of 85%, all partitions,
search domains, budgets, selection rule and statistics before training. Code identities are also locked.
The five exhaustive stratified roles have 40000/5000/5000/2000/8000 rows; actual C and Vsearch
contain 256 and 512 stratified examples. Images remain uint8; float normalization happens per batch.
Official labels define CE and accuracy. Teacher agreement never substitutes for task quality.

M0 is saved and reloaded through CheckpointManager. Its unique physical tensor storage defines S0
and B=0.5 S0. Every method uses the same loaded checkpoint. Controls are Dense32, fixed SVD ranks
64/32/4, weight-only PTQ4, native static INT8, and the declared PTQ4/SVD32/LPLR8 manual assignment.

Both search arms call the same run_grammar_search with QUANTIZATION_GRAMMAR_ADAPTER. Each pair
uses search seed 17, 29 or 43. Random draws 24 unique initial genotypes, including the exact four
used to initialize Evo; Evo has population four and 24 generation cap. Both charge 24 assessments
including failures and cap proposals at 128. Search time is at most 600 seconds, including preparation;
Vpick has a separate common 300-second cap and remains inside the global deadline.
Search must start only when the orchestrator has enough remaining time for a complete pair and reporting.

Each path offers Dense/PTQ4 and SVD/DSVD/LPLR/LPLR-SVD/CALDERA at ranks 32/64/96,
16/32/48, or 2/4/6. TensorTender W4A8 is available only for the 128→10 head in both arms.
Non-dense compatible prefixes cover None/2/4. LPLR uses stochastic endpoint four-bit factors,
with comparison rank equal to factor rank. CALDERA has one outer and inner iteration and explicit
ridge 1e-4, so its stabilized metric is predeclared. Execution randomness derives from the complete
canonical genotype and does not depend on visit order. No recovery or QAT enters this primary space.

Every assessed artifact is publicly quantized, saved, reloaded, checked for unique physical bytes,
and evaluated against Vsearch labels. Failed proposals, traces, context identities, the engine's
entire finite archive and intermediate Vpick records are retained. A failed engine with a preserved
artifact is `partial_artifact`. The fixed shortlist is the top four feasible finite archive entries
by Vsearch CE, storage, canonical ID. Vpick CE, storage and canonical ID select the final artifact.

The `freeze` command requires terminal records for every declared job and preflights the full-test
inference cost on calibration inputs. A predicted full pass with the fixed safety factor above the
300-second cap makes that artifact ineligible; the test sample size is never reduced. The exclusive
`CLOSED_TEST_OPENED.json` marker is created only after the selection lock. `closed-test` then decodes
all 10000 official test examples once, evaluates each eligible model in batches of 128, and preserves
partial results after each model. It cannot be resumed by reopening the test.

Example command shapes (the guardian is intentionally omitted here; the orchestrator supplies it):

```
python -m fedcore.experiments.fashion_worker manifest --manifest MANIFEST --run-root ROOT --limits LIMITS --source-commit COMMIT
python -m fedcore.experiments.fashion_worker prepare --manifest MANIFEST --run-root ROOT --limits LIMITS
python -m fedcore.experiments.fashion_worker train --manifest MANIFEST --run-root ROOT --limits LIMITS
python -m fedcore.experiments.fashion_worker job --job evo_17 --manifest MANIFEST --run-root ROOT --limits LIMITS
python -m fedcore.experiments.fashion_worker freeze --manifest MANIFEST --run-root ROOT --limits LIMITS
python -m fedcore.experiments.fashion_worker closed-test --manifest MANIFEST --run-root ROOT --limits LIMITS
python -m fedcore.experiments.fashion_worker analyze --manifest MANIFEST --run-root ROOT --limits LIMITS
```

Analysis includes CE, accuracy, class counts, confusion matrices, M0 agreement, paired correctness,
exploratory exact McNemar values and a paired-example bootstrap. Three search seeds conditional on
one M0 are descriptive and do not establish independent training replication or optimizer superiority.
Fashion-MNIST remains a historical public benchmark; a bounded prior-use audit cannot establish
global test-set pristineness. Reference kernels and compression ratios imply no hardware speed claim.
