# Measured PETRA pilot

Run status: succeeded.
Selection uses validation only. Final test scores are recorded after the archive freezes.
All rows refer to one configuration and one artifact. Missing measurements keep an explicit status.
Latency: ms/batch; throughput: samples/s; file and tensor contents: bytes. RSS is a process snapshot; CUDA is a process peak.
p50/p95 describe empirical repetitions of calls, not uncertainty of cross-seed means.

| Configuration ID | Method | Status | Validation | Test | File bytes | p50 ms/batch |
|---|---|---|---:|---:|---:|---:|
| a8557c4fdd57f193e76e | baseline | succeeded | 0.9739776951672863 | 0.9518518518518518 | 290245 | 0.8402000000000001 |
| ee669b6494438e8cb1d2 | svd | succeeded | 0.6728624535315985 | unavailable | 110132 | 0.8932 |
| 271046b594c30f3c86de | svd | succeeded | 0.9553903345724907 | unavailable | 225012 | 1.0040499999999999 |
| 9948d613124305535090 | structural_pruning | succeeded | 0.9628252788104089 | 0.9518518518518518 | 180413 | 0.70235 |
| 5bed6694e8e40f9b97a4 | svd | succeeded | 0.895910780669145 | 0.9185185185185185 | 170100 | 1.17275 |
| 7aaee2ebd0513dd8aa15 | pruning | succeeded | 0.9591078066914498 | unavailable | 290621 | 0.80335 |

pilot only; superiority is not established

The manifest, prediction files and raw timing repetitions are the source of this table.
