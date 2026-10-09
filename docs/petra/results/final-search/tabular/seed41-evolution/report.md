# Measured PETRA pilot

Run status: succeeded.
Selection uses validation only. Final test scores are recorded after the archive freezes.
All rows refer to one configuration and one artifact. Missing measurements keep an explicit status.
Latency: ms/batch; throughput: samples/s; file and tensor contents: bytes. RSS is a process snapshot; CUDA is a process peak.
p50/p95 describe empirical repetitions of calls, not uncertainty of cross-seed means.

| Configuration ID | Method | Status | Validation | Test | File bytes | p50 ms/batch |
|---|---|---|---:|---:|---:|---:|
| a8557c4fdd57f193e76e | baseline | succeeded | 0.9764705882352941 | 0.9651162790697675 | 58703 | 0.07715 |
| 8eac4fa004b89d0acde4 | structural_pruning | succeeded | 0.9764705882352941 | 0.9651162790697675 | 30999 | 0.0567 |
| 112448f629999f366675 | train | succeeded | 0.9764705882352941 | unavailable | 58703 | 0.128 |
| 271046b594c30f3c86de | svd | succeeded | 0.9764705882352941 | unavailable | 63295 | 0.1351 |
| 9948d613124305535090 | structural_pruning | succeeded | 0.9764705882352941 | unavailable | 43743 | 0.06965 |
| 7aaee2ebd0513dd8aa15 | pruning | succeeded | 0.9764705882352941 | unavailable | 58903 | 0.07905000000000001 |

pilot only; superiority is not established

The manifest, prediction files and raw timing repetitions are the source of this table.
