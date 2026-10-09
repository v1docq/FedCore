# Measured PETRA pilot

Run status: succeeded.
Selection uses validation only. Final test scores are recorded after the archive freezes.
All rows refer to one configuration and one artifact. Missing measurements keep an explicit status.
Latency: ms/batch; throughput: samples/s; file and tensor contents: bytes. RSS is a process snapshot; CUDA is a process peak.
p50/p95 describe empirical repetitions of calls, not uncertainty of cross-seed means.

| Configuration ID | Method | Status | Validation | Test | File bytes | p50 ms/batch |
|---|---|---|---:|---:|---:|---:|
| a8557c4fdd57f193e76e | baseline | succeeded | 3.8335819244384766 | 3.8855278491973877 | 192555 | 20.4719 |
| fadf8b12d4e146bd4fec | train | succeeded | 3.454167604446411 | 3.507242441177368 | 192555 | 20.58625 |
| f71db8df54de63e32301 | svd | succeeded | 3.4808008670806885 | 3.52359938621521 | 146457 | 17.83495 |

pilot only; superiority is not established

The manifest, prediction files and raw timing repetitions are the source of this table.
