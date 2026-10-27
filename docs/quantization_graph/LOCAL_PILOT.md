# PR166 bounded local CPU pilot

This experimental harness adds no production transformation. It measures technical
feasibility and fidelity to one frozen, untrained synthetic teacher. Its CE and
argmax agreement do not measure trained task quality. The optional slow-spectrum
regime is declared diagnostic and is not run by this protocol.

The caller must preserve the original `LIMITS.json`, its SHA256, source commit,
implementation commit, environment and manifest hashes before the first numeric
job. All preparations, tests, failures and profiling share its original absolute
three-hour deadline. Nothing resumes or resets that deadline.

Run every worker through the direct-file guardian:

```text
python fedcore/experiments/quantization_pilot_guard.py --limits LIMITS.json --run-root RUN --report RUN/guards/NAME.json --timeout-seconds 180 -- python -m fedcore.experiments.quantization_pilot_worker COMMAND --manifest RUN/manifest.json --run-root RUN --limits LIMITS.json
```

`RUN` and `LIMITS.json` above stand for absolute paths. Use the existing CPU Python
environment; do not install packages, change the global environment, download data,
use SSH, or use GPU. The worker directs the existing model registry into
`RUN/registry` before importing FedCore. The guardian imports only stdlib/psutil.

The Windows guardian atomically creates the child inside a kill-on-close Job
Object, starts it suspended and then resumes it. Unsupported job attributes refuse
launch. The additional kernel job cap is six GiB of aggregate private committed
memory, separately labeled from RSS. Every 20 ms, the guardian records identities
and RSS for itself and all live Job Object members. It aborts at seven GiB sampled
family RSS, within the eight GiB authorization. It also checks the five GiB artifact
cap, both absolute clocks, per-job deadline and an exclusive worker lock. A
disappeared child is skipped; access denied or unobservable membership aborts.
Raw `.rss.jsonl` samples permit independent recomputation of the sampled family
maximum; samples may miss shorter peaks. Windows job membership and kernel limits
add protection between samples. A surviving child on root exit is explicitly
recorded as guardian cleanup.

An external experiment controller can register its identity before guardian launch:
`FEDCORE_PILOT_CONTROLLER_IDENTITY` is JSON with the exact keys `pid` and
`create_time`. The guardian requires a live ancestor with that creation time and
includes its RSS in every family sample. Arbitrary app processes are not admitted.
When this variable is absent, the receipt explicitly says that parent/controller
RSS is excluded. Historical receipts created without registration retain that
limitation; their measurements must not be relabeled as including the controller.

The finite command sequence is:

1. `manifest --source-commit c70d37914b152fd6d8886c4e692e4c8355ed05c2`
   writes the exact 16-job finite protocol. `--include-recovery` declares an optional
   two-job matched learning panel before the manifest is frozen.
2. `prepare` constructs the `64→128→64→10` Sequential/ReLU teacher with fast singular
   decay, serializes and reloads M0, and independently generates training 4096,
   calibration 256 and selection 256 rows. The 2048-row closed test remains unopened.
   IDs, seeds, byte hashes and logits provenance are recorded. Training rows are
   unused in the primary 16-job panel. The storage limit is half M0 unique physical
   tensor bytes.
3. `job --job ID` runs one declared job. Required IDs are `dense32`, `dense64`,
   `static_int8`, `ptq4`, `svd`, `dsvd4`, `lplr4`, `lplr_svd4`, `caldera4`,
   `tender_w4a8`, `manual_hetero`, `exact_catalogue`, `random_17`, `evo_17`,
   `random_29`, `evo_29`. Every job is terminal and cannot be replaced. Use a
   180-second guardian timeout for construction and 600 seconds for each search,
   narrowed further by the shared deadline. Failed jobs and partial logs count.
4. If the guardian kills a job, run `record-abort --job ID --guard-report RECEIPT`.
   This finalizes only an unfinished record and preserves the original guard
   receipt hash, failure and cost. It does not retry the numerical work.
5. `freeze` requires all declared terminal records and locks their hashes and
   artifact selections. No jobs can subsequently be added or replaced.
6. `closed-test` opens the held-out role once, verifies frozen record hashes and
   measures all loaded artifacts. Its exclusive opening marker prevents reopening
   after interruption; retain any partial report.
7. `timing --job ID --batch 1 --threads 1 --repeat 0 --profile-mode single --seconds 120`
   profiles one loaded artifact in a fresh worker. Declare the fixed subset and
   repeats before timing. Modes `aa` and `ab` load two identical copies or a dense
   control/candidate, use randomized paired AB/BA order and preserve individual
   observations. Full protocol: batches 1/128, threads 1/8, 20 warmups, 200 requests
   per model and five fresh-process repeats. Each timing job is capped at 180
   seconds. Short profiles have no eligible p95. `single` mode supplies an isolated
   artifact RSS profile; paired timing RSS includes both models.
8. `analyze` writes `REPORT.json` and `REPORT.md`. A speedup is eligible only after
   five complete A/A repeats pass the median and order-group 0.95–1.05 checks and
   five complete paired candidate repeats exist in the same regime. Missing
   coverage remains unknown. No significance or search-superiority claim is made.

All methods use explicit `StageBudgets` and request `max_peak_bytes=8 GiB`; the
external family guardian is stricter. The no-learning panel permits zero training
updates. Homogeneous SVD ranks are 8/8/4; DSVD/LPLR/LPLR-SVD widths are 16/16/8.
The primary LPLR comparison rank equals its actual factor width. Both factor grids
use endpoint stochastic four-bit quantization. CALDERA uses one outer/inner
iteration; Tender uses W4A8 and its existing Python integer reference. Portable
decode uses its existing FP64 execution, and native backend `prepack` remains in
each forward. No decoded or packed forward cache is introduced.

Dense controls are explicitly outside the final storage limit. Native static
conversion omits an intermediate Dense storage cap, then checks the loaded final
artifact against the same external limit. All other homogeneous configurations
are fixed before test; infeasibility remains a result. The manual mixture uses
PTQ/SVD/LPLR across its three operators. The exact catalogue builds nine local
Dense/PTQ/SVD alternatives, enforces 256 options and 128 MiB cumulative built
storage, and records additive frozen local output SSE separately from whole-model
selection CE.

Random and Evo share `run_grammar_search`, the same typed GrammarAdapter, finite
domain, prequantization prefix grid `(None, 2, 4)`, no training stages, eight charged
evaluations, provenance checks, artifact cache and loaded-model assessment. Random
uses eight unique seeded initial genotypes with no adaptive variation. Evo begins
with their first four and uses GOLEM variation until the common cap. Every proposal,
failure, genotype, trace and candidate artifact is preserved. Cache bytes and full
catalogue/search costs are reported.

All artifacts pass independent `CheckpointManager` save/load, finite forward,
round-trip output and full unique physical-storage checks. Checkpoints live in a
SHA256 content-addressed store; serialized file bytes are a separate quantity.
The optional recovery pair starts from identical decoded LPLR factors and uses
the same training rows/logits, LoRA factor stages and eight SGD updates. One member
is re-encoded to packed factors; the real-factor control retains its float factors.
Both final storage costs are measured; neither is silently retuned.

The first frozen series exposed an optimizer integration bug: failed candidates
produced GOLEM `null_fitness`, but `FedcoreDispatcher` admitted those evaluation
results into the population. GOLEM then failed while comparing invalid fitness.
The narrow dispatcher repair excludes invalid fitness before population admission;
the original failed assessments remain charged and recorded. Numerical forward
implementations are unchanged. Historical job records and closed-test results are
never replaced after this repair. An engineering diagnostic must use a separate
directory, the remaining original deadline and selection data only.

A valid finite artifact from an incomplete search is reported as `partial_artifact`.
`recorded_job_status` preserves the historical job field. The report separately
shows the nested engine status/reason, charged evaluations, requested evaluations
and number of assessed variation proposals. Successful loading of an initial
population artifact cannot establish that evolution completed. If a later
profiling harness commit differs from the artifact construction commit, preserve
both source identities explicitly before profiling.

Run meaningful contract tests under the same guardian:

```text
python -m pytest tests/refactoring/test_quantization_pilot.py -q --basetemp RUN/pytest-tmp
```

They cover both deadline clocks, authorization narrowing, PID reuse and disappearance,
fail-closed observation, finite manifest parsing, generated quantile invariants,
source and split reproducibility, prefix support/normalization, public quantize and
checkpoint storage round-trip, Dense64 dtype, and the one-way selection/test lock.
