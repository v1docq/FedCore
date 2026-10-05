# Measured PETRA pilot

Run status: succeeded.
Selection uses validation only. Final test scores are recorded after the archive freezes.
All rows refer to one configuration and one artifact. Missing measurements keep an explicit status.
Latency: ms/batch; throughput: samples/s; file and tensor contents: bytes. RSS is a process snapshot; CUDA is a process peak.
p50/p95 describe empirical repetitions of calls, not uncertainty of cross-seed means.

| Configuration ID | Method | Status | Validation | Test | File bytes | p50 ms/batch |
|---|---|---|---:|---:|---:|---:|
| a8557c4fdd57f193e76e | baseline | succeeded | 0.9739776951672863 | 0.9518518518518518 | 290193 | 0.7716 |
| fadf8b12d4e146bd4fec | train | succeeded | 0.9739776951672863 | 0.9518518518518518 | 290193 | 1.01965 |
| f71db8df54de63e32301 | svd | succeeded | 0.9330855018587361 | unavailable | 169498 | 0.9292499999999999 |
| a523201eb1a1e0361959 | structural_pruning | succeeded | 0.9702602230483272 | 0.9555555555555556 | 146479 | 1.1776499999999999 |

pilot only; superiority is not established

The manifest, prediction files and raw timing repetitions are the source of this table.
