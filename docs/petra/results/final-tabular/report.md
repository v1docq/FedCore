# Measured PETRA pilot

Run status: succeeded.
Selection uses validation only. Final test scores are recorded after the archive freezes.
All rows refer to one configuration and one artifact. Missing measurements keep an explicit status.
Latency: ms/batch; throughput: samples/s; file and tensor contents: bytes. RSS is a process snapshot; CUDA is a process peak.
p50/p95 describe empirical repetitions of calls, not uncertainty of cross-seed means.

| Configuration ID | Method | Status | Validation | Test | File bytes | p50 ms/batch |
|---|---|---|---:|---:|---:|---:|
| a8557c4fdd57f193e76e | baseline | succeeded | 0.9882352941176471 | 0.9534883720930233 | 58703 | 0.11355000000000001 |
| fadf8b12d4e146bd4fec | train | succeeded | 0.9882352941176471 | unavailable | 58703 | 0.11345 |
| f71db8df54de63e32301 | svd | succeeded | 0.9882352941176471 | unavailable | 48373 | 0.1085 |
| a523201eb1a1e0361959 | structural_pruning | succeeded | 0.9882352941176471 | unavailable | 36767 | 0.0612 |
| cc2b9661ea310ba236fc | ptq | succeeded | 0.9882352941176471 | 0.9534883720930233 | 26717 | 0.035250000000000004 |
| 716197c55bbadc90c911 | qat | succeeded | 0.9882352941176471 | unavailable | 26819 | 0.07569999999999999 |

pilot only; superiority is not established

The manifest, prediction files and raw timing repetitions are the source of this table.
