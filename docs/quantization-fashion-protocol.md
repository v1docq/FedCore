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

## S2 amendment: independent proposal streams and a new closed holdout

S1 is preserved at commit `9295f6b36151b1d163b06abc27f3d3ab182a624c`, including all raw records.
Its inherited `search_seed + attempt` initializer produced overlapping shifted windows across seeds:
12/24 initial Random genotypes overlapped for seeds 17/29 and 10/24 for 29/43. This does not break
within-pair fairness, but those S1 search repetitions are correlated by construction.

The corrected Fashion initializer owns one `random.Random(search_seed)` stream per run, draws a
63-bit seed for every proposal, and deduplicates canonical genotypes. Evo and Random within a pair
still share the exact first four proposals. Naturally recurring genotypes across independently seeded
streams are allowed; there is no artificial exclusion between replicates. The old synthetic pilot
initializer remains unchanged.

S2 has its own finite schema, `pr166_fashion_s2_v1`, and module
`fedcore.experiments.fashion_independent_worker`. It reuses the exact M0 checkpoint SHA256
`b333cdaa33f29d07e9be27249145aa95ffba6459969c66ce481bf8ae15282429`, without training.
All original fit, stop, C, Vsearch and Vpick identities remain unchanged. The five unchanged control
artifacts are imported with byte-for-byte SHA256 verification; only the six search jobs are rerun.

The official S1 10000-row test is never reopened. S2 instead freezes a 6000-row holdout, 600 per
class, sampled with seed 20261031 from the sorted union of unused calibration and search pool rows.
There are 6232 eligible rows (4744 calibration-pool remainder and 1488 search-pool remainder), and
none belong to fit, stop, actual C, actual Vsearch or Vpick. Holdout metrics are computed once after
S2 selection lock through the same exclusive-marker mechanism, with full 6000-row passes.

S2 is authorized within the original shared deadline; it creates no new time allowance. The root
orchestrator owns all tests, data preparation and computations. Its command sequence is:

```
python -m fedcore.experiments.fashion_independent_worker manifest --manifest S2/manifest.json --run-root S2 --limits ORIGINAL_LIMITS --source-commit COMMIT --parent-root S1 --parent-manifest S1/manifest.json
python -m fedcore.experiments.fashion_independent_worker reuse-m0 --manifest S2/manifest.json --run-root S2 --limits ORIGINAL_LIMITS --parent-root S1 --parent-manifest S1/manifest.json
python -m fedcore.experiments.fashion_independent_worker job --job random_17 --manifest S2/manifest.json --run-root S2 --limits ORIGINAL_LIMITS
python -m fedcore.experiments.fashion_independent_worker freeze --manifest S2/manifest.json --run-root S2 --limits ORIGINAL_LIMITS
python -m fedcore.experiments.fashion_independent_worker closed-test --manifest S2/manifest.json --run-root S2 --limits ORIGINAL_LIMITS
python -m fedcore.experiments.fashion_independent_worker analyze --manifest S2/manifest.json --run-root S2 --limits ORIGINAL_LIMITS
```

`reuse-m0` also creates the five terminal imported-control records. Run only `random_17`, `evo_17`,
`random_29`, `evo_29`, `random_43`, and `evo_43` as new jobs. Source roots must be separate siblings.
Reports must distinguish S1 from S2 and never pool their different holdouts. All inference remains
conditional on a single trained M0; independent proposal streams do not replicate training.
