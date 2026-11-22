# From public source files to wallet–event edges

The published source-data tables can be used without downloading a fill archive. This directory is for readers who want to check an upstream shard or construct aggregates themselves. Commands below run from the repository root; replace `RAW`, `WORK`, and `CSR` with external directories of your choice. No API key, login, or blockchain RPC is needed. Python 3.11+ is required; the fill collector and Gamma collector use only the standard library. Aggregation was tested with DuckDB 1.5.5; the offline tests also use PyArrow.

## 1. Pinned fill archive (historical input)

`frozen_fill_inventory.json` is the original collector v1.1.0 download plan, saved on 1 September 2026. Although that collector requested the branch named `main`, each source has a **resolved immutable revision**. The release adapter ignores the branch and addresses only files at those two commits. Every file has its partition date, size and SHA-256. Only the cleaned `daily_aligned` and `daily_aligned_multi` Parquet layers enter this study.

| Public source | Frozen revision | Files |
|---|---|---:|
| [TimeSeventeen/Polymarket-v1](https://huggingface.co/datasets/TimeSeventeen/Polymarket-v1/tree/5aa1b9d52316a8b2e789e81c8ae42c7ed532e8aa) | `5aa1b9d52316a8b2e789e81c8ae42c7ed532e8aa` | 2,105 |
| [TimeSeventeen/Polymarket-v2](https://huggingface.co/datasets/TimeSeventeen/Polymarket-v2/tree/2a8dba6472f4c9ca558869e9cc1a65b9e1614264) | `2a8dba6472f4c9ca558869e9cc1a65b9e1614264` | 270 |

Inspect the plan offline; to try a real input, request just the first day:

```sh
python code/acquisition/archive.py plan
python code/acquisition/archive.py plan --start 2022-11-21 --end 2022-11-21
python code/acquisition/archive.py fetch --root RAW --start 2022-11-21 --end 2022-11-21
python code/acquisition/archive.py verify --root RAW --start 2022-11-21 --end 2022-11-21
```

The first day is **one 10,893-byte shard**. An unrestricted `fetch` requests **24,060,654,592 bytes** (2,375 files). Existing files are hash-checked, not overwritten; interrupted whole-file downloads can be retried. The two exchange generations are pooled where their calendars overlap; the adapter does not discard records merely because a date, wallet or amount occurs in both. It does not independently establish absence of duplicate transactions.

## 2. Direct Polymarket snapshots

**For replication use the dated extracts** in [data/gamma](../../data/gamma) and [data/leaderboard](../../data/leaderboard). Fresh API calls cannot reconstruct what either endpoint reported on 1 or 4 September 2026. The Gamma extracts contain the fields needed for this study, not full historic HTTP responses. The direct leaderboard data are platform-reported rankings, not profits inferred from the fill archive.

To collect a **new**, independent Gamma snapshot: `python code/acquisition/platform_snapshot.py --fetch-current-to NEW_GAMMA`. This release adapter follows the study's `/events/keyset` collector: page through open and then closed events, requesting ascending ID order, `limit=500`, and using the returned `next_cursor` as `after_cursor`. Take the first appearance of each event ID; emit event/title/slug/tags, nested market/condition/outcome mappings, and a unique tag dictionary. Empty pages/end cursors terminate pagination. Only a successful capture writes `complete.json` with timestamps, counts and output hashes. It refuses to overwrite a directory and may stop if the public API rejects a request.

The [original leaderboard collector](../leaderboard) covers the other direct endpoint: `/v1/leaderboard?category=OVERALL&timePeriod=ALL&orderBy=VOL` (or `PNL`), `limit=50`, offsets 0, 50, …, 1000. These final two pagesets provided up to 1,050 observed rank positions per ordering. The analysis selects reported ranks 1–1,000 **separately** for each ordering; it does not pool the lists.

## 3. Positive-fill and graph aggregation (offline)

First hash-verify the requested date slice with `archive.py verify`. `aggregate.py` checks the file inventory and sizes, applies inclusive UTC date bounds, and joins the fills to the **frozen** mapping by condition ID rather than market title. The compact mapping stores 32-byte binary identifiers; these are decoded as lowercase `0x`-prefixed hexadecimal to match the source. A nonunique mapping aborts the run. For example:

```sh
python code/acquisition/aggregate.py --root RAW --out WORK --start 2022-11-21 --end 2022-11-21
```

| Output | Unit and interpretation |
|---|---|
| `event_totals.parquet` | One row per mapped event with positive finite fills. Count and sum each fill once (`trade_count`, `usd_volume`). |
| `wallet_totals.parquet` | One row per wallet. Sum maker and taker incidences (`participant_fills`, `participant_volume`), **including unmapped fills**. This is gross participation, not wealth or P&L. |
| `participation.parquet` | One row per mapped `wallet_id,event_index`, ordered by these two integers. `participation_fills` and `usd` both credit maker and taker; if they coincide both incidences are counted. Checks require mapped incidences and amount to equal twice the mapped fill count and volume. |
| `wallets.parquet`, `events.parquet`, `dimensions.json` | Reversible zero-based indices of lowercased addresses and event IDs (both sorted as strings). Includes all observed wallets in the requested interval; wallets trading only unmapped conditions have no graph edges. The JSON contains scope, counts and conservation totals. |

Use **one run over the entire interval** (`2022-11-21` through `2026-08-15`) for comparable global integer IDs and an input to the [sparse network builder](../network/sparse_storage.py): `python code/network/sparse_storage.py --input WORK/participation.parquet --dimension-manifest WORK/dimensions.json --output CSR --scope full`. IDs from different partial runs **cannot** be concatenated. Using the full window entails repeated large reads, substantial temporary disk and additional output storage; the 24.06 GB download size is not a working-disk estimate. The aggregation defaults to 2 threads and a DuckDB 4 GB memory setting; `--temporary-disk-limit` applies only to spill files, not database files or outputs.

**Verification and scope.** This is a portable adaptation of the study's once-per-fill event aggregation and maker/taker edge builder, not the original checkpointed production run. Offline tests cover the frozen inventory, Gamma pagination/duplicate event IDs, unmapped conditions, self-trades, date and amount filters, and event-index order. On the real 21 November 2022 shard, joined to all 2,946,628 mappings, it returned 28 fills, one mapped event, 290.20 USDC event volume, two wallets and two edges (580.40 gross wallet USDC). That day's fill count and volume agree with the archived independent daily table. The billion-record rebuild was **not** rerun for this release; file bytes, floating-point sums, optimizer seeds and derived partition labels should not be presumed identical to historical outputs without checking. This adapter does not rebuild election cohorts, assign new event subjects, or estimate wallet profits.

```sh
python -B -m unittest discover -v -s code/acquisition
```
