# Measured PETRA pilot

Run status: succeeded.
Selection uses validation only. Final test scores are recorded after the archive freezes.
All rows refer to one configuration and one artifact. Missing measurements keep an explicit status.
Latency: ms/batch; throughput: samples/s; file and tensor contents: bytes. RSS is a process snapshot; CUDA is a process peak.
p50/p95 describe empirical repetitions of calls, not uncertainty of cross-seed means.

| Configuration ID | Method | Status | Validation | Test | File bytes | p50 ms/batch |
|---|---|---|---:|---:|---:|---:|
| a8557c4fdd57f193e76e | baseline | succeeded | 0.9553903345724907 | 0.9407407407407408 | 290245 | 0.8238000000000001 |
| 5bed6694e8e40f9b97a4 | svd | succeeded | 0.9516728624535316 | unavailable | 170100 | 0.87575 |
| ee669b6494438e8cb1d2 | svd | succeeded | 0.6765799256505576 | unavailable | 110132 | 0.8878 |
| 9948d613124305535090 | structural_pruning | succeeded | 0.9702602230483272 | 0.9481481481481482 | 180413 | 0.78425 |
| 7aaee2ebd0513dd8aa15 | pruning | succeeded | 0.9739776951672863 | 0.9481481481481482 | 290621 | 0.80665 |
| 8eac4fa004b89d0acde4 | structural_pruning | succeeded | 0.966542750929368 | 0.9296296296296296 | 108413 | 0.6228 |

pilot only; superiority is not established

The manifest, prediction files and raw timing repetitions are the source of this table.
