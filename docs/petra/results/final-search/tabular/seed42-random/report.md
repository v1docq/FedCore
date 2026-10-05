# Measured PETRA pilot

Run status: succeeded.
Selection uses validation only. Final test scores are recorded after the archive freezes.
All rows refer to one configuration and one artifact. Missing measurements keep an explicit status.
Latency: ms/batch; throughput: samples/s; file and tensor contents: bytes. RSS is a process snapshot; CUDA is a process peak.
p50/p95 describe empirical repetitions of calls, not uncertainty of cross-seed means.

| Configuration ID | Method | Status | Validation | Test | File bytes | p50 ms/batch |
|---|---|---|---:|---:|---:|---:|
| a8557c4fdd57f193e76e | baseline | succeeded | 0.9882352941176471 | 0.9534883720930233 | 58703 | 0.0797 |
| ee669b6494438e8cb1d2 | svd | succeeded | 0.9882352941176471 | 0.9534883720930233 | 34763 | 0.07705000000000001 |
| 271046b594c30f3c86de | svd | succeeded | 0.9882352941176471 | unavailable | 61579 | 0.09995000000000001 |
| 9948d613124305535090 | structural_pruning | succeeded | 0.9882352941176471 | unavailable | 43743 | 0.0695 |
| 5bed6694e8e40f9b97a4 | svd | succeeded | 0.9882352941176471 | unavailable | 48587 | 0.083 |
| 7aaee2ebd0513dd8aa15 | pruning | succeeded | 0.9882352941176471 | unavailable | 58903 | 0.1266 |

pilot only; superiority is not established

The manifest, prediction files and raw timing repetitions are the source of this table.
