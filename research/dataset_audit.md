# Canonical dataset audit

Measured on 16 July 2026 from the files linked by the official DCRNN repository.
Hashes make these observations specific to the exact files, not generic dataset
claims.

| File | SHA-256 | Shape | Time range | NaN rate | Zero rate |
|---|---|---:|---|---:|---:|
| `metr-la.h5` | `64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05` | 34,272 x 207 | 2012-03-01 00:00 to 2012-06-27 23:55 | 0 | 0.0810935083 |
| `pems-bay.h5` | `65d69fb0a2323dba9867179eb7af47c8b814186bc459ff0a4937d21614153c8f` | 52,116 x 325 | 2017-01-01 00:00 to 2017-06-30 23:55 | 0 | 0.0000307598 |

| Graph file | SHA-256 | Shape | Positive entries (including self-loops if present) |
|---|---|---:|---:|
| `adj_mx.pkl` | `a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3` | 207 x 207 | 1,722 |
| `adj_mx_bay.pkl` | `116275f5704d0d492018e14c047f7ae5004385b450aa907f88424055a4a97370` | 325 x 325 | 2,694 |

The HDF5 values contain no NaNs. Following the canonical masked traffic metrics,
zero values are treated as unavailable observations; this convention explains
the METR-LA 8.109% missing ratio often reported in benchmark papers. Any other
missingness definition must be separately labelled.

