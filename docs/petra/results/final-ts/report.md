# Measured PETRA pilot

Run status: succeeded.
Selection uses validation only. Final test scores are recorded after the archive freezes.
All rows refer to one configuration and one artifact. Missing measurements keep an explicit status.
Latency: ms/batch; throughput: samples/s; file and tensor contents: bytes. RSS is a process snapshot; CUDA is a process peak.
p50/p95 describe empirical repetitions of calls, not uncertainty of cross-seed means.

| Configuration ID | Method | Status | Validation | Test | File bytes | p50 ms/batch |
|---|---|---|---:|---:|---:|---:|
| a8557c4fdd57f193e76e | baseline | succeeded | 24.1302433013916 | 18.15318489074707 | 89281 | 0.45130000000000003 |
| fadf8b12d4e146bd4fec | train | succeeded | 26.10818099975586 | unavailable | 89281 | 0.5024 |
| f71db8df54de63e32301 | svd | succeeded | 23.913888931274414 | 15.61528491973877 | 69053 | 0.3837 |

pilot only; superiority is not established

The manifest, prediction files and raw timing repetitions are the source of this table.
