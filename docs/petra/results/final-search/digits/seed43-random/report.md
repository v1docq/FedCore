# Measured PETRA pilot

Run status: succeeded.
Selection uses validation only. Final test scores are recorded after the archive freezes.
All rows refer to one configuration and one artifact. Missing measurements keep an explicit status.
Latency: ms/batch; throughput: samples/s; file and tensor contents: bytes. RSS is a process snapshot; CUDA is a process peak.
p50/p95 describe empirical repetitions of calls, not uncertainty of cross-seed means.

| Configuration ID | Method | Status | Validation | Test | File bytes | p50 ms/batch |
|---|---|---|---:|---:|---:|---:|
| a8557c4fdd57f193e76e | baseline | succeeded | 0.9702602230483272 | 0.9666666666666667 | 290245 | 0.83585 |
| 8eac4fa004b89d0acde4 | structural_pruning | succeeded | 0.9405204460966543 | 0.9592592592592593 | 108413 | 0.63965 |
| 7aaee2ebd0513dd8aa15 | pruning | succeeded | 0.9256505576208178 | unavailable | 290621 | 0.8337 |
| 9948d613124305535090 | structural_pruning | succeeded | 0.9330855018587361 | unavailable | 180413 | 0.6903999999999999 |
| 271046b594c30f3c86de | svd | succeeded | 0.9256505576208178 | unavailable | 225012 | 1.0263 |
| ee669b6494438e8cb1d2 | svd | succeeded | 0.6431226765799256 | unavailable | 110132 | 0.87575 |

pilot only; superiority is not established

The manifest, prediction files and raw timing repetitions are the source of this table.
