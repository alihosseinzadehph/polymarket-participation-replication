# Polymarket: data and code for the arXiv manuscript

**Political Orientation and Participation in Prediction Markets: Large Stakes Converge, Small Stakes Diverge**

This archive separates three inputs that should not be confused: executed trades obtained from a public trade archive, metadata and leaderboard rankings collected directly from Polymarket, and data products calculated for the paper. No credentials are needed to inspect the deposited files.

## Data sources

| Source | Observation frame | Use in the paper |
| --- | --- | --- |
| [TimeSeventeen/Polymarket-v1, fixed revision](https://huggingface.co/datasets/TimeSeventeen/Polymarket-v1/tree/5aa1b9d52316a8b2e789e81c8ae42c7ed532e8aa) | 21 November 2022–28 April 2026 | Cleaned executed fills, first exchange generation. See [Qin and Yang](https://doi.org/10.48550/arXiv.2606.04217) for the v1 data paper. |
| [TimeSeventeen/Polymarket-v2, fixed revision](https://huggingface.co/datasets/TimeSeventeen/Polymarket-v2/tree/2a8dba6472f4c9ca558869e9cc1a65b9e1614264) | 3 April–15 August 2026 | Cleaned fills from the later exchange generation. Both generations are retained in overlapping calendar dates. |
| Polymarket Gamma API, `GET /events/keyset` | 1 September 2026, 11:40:20–12:54:17 UTC | Market-condition to event mapping and the event metadata used for subject classification. The snapshot includes open and closed events. |
| Polymarket Data API, `GET /v1/leaderboard` | 4 September 2026, 08:21–08:30 UTC | Two separately ordered cross-sections, `OVERALL` / `ALL`, ordered by `VOL` or `PNL`. The paper uses ranks 1–1,000 from each. This is not an all-wallet profit history. |

The published trade files are cited, not redistributed. Running the API collectors today would produce **new** metadata or rankings, not the snapshots used here. Use the deposited frozen fields to reproduce the paper. Attribution to the original sources is retained; this package does not grant new rights over third-party inputs.

## Where to start

1. To check that the deposited files are intact, run `python verify_release.py` from this directory (Python 3.9 or later; no additional packages). A checksum verifies a file's contents, not a scientific result.
2. [The data dictionary](data/README.md) describes the unit of observation and omitted fields for each file. Run `python check_source_data.py` (NumPy and DuckDB; tested with DuckDB 1.5.5) to check the condition mapping, join mapped event totals to the event dictionary, recalculate both Ginis and both top-1% shares, reconcile graph degrees, and check the three Fig. 2 samples against their sources.
3. To redraw the six main figures, follow the [code guide](code/README.md), install [the analysis requirements](code/requirements.txt), and run `python code/reproduce_figures.py --output rebuilt_figures`, using a new output directory. `python code/check_analysis.py` re-estimates additional reported statistics from compact inputs. Neither command requires the trade archive.
4. For a fresh full-network reconstruction, see [acquisition and preparation](code/acquisition/) and the [network pipeline](code/network/PIPELINE.md). The original 2,375 compressed trade files total 24,060,654,592 bytes (22.41 GiB). Decoding the two trade layers and building the sparse network require additional working storage; the 255,704,359 wallet–event edges and individual node assignments are not shipped in this archive. The original billion-fill analysis was not rerun when preparing this release.
5. To reconstruct the September election cohorts and October follow-up vectors directly from the relevant archive partitions, see [the election formation replay](code/election_formation/). The frozen vectors can also be used on their own for the reported statistical tests.

## Accounting boundaries

See the [release checks](validation/README.md) for what was run, what matched the manuscript, and what was not rerun.

An event can contain multiple markets. An event's fill amount counts each mapped fill once; gross wallet trading credits its amount once to the maker and once to the taker. Across the archived fills there are 3,106,517 observed wallet addresses. The graph has 3,106,514 wallets and 764,378 positive-volume mapped events. The 52,022 unmatched fills do not enter event volume or graph edges.

The 21 detailed subjects classify events before subject-network construction. They are not the 149 communities found by optimization in the amount-weighted network. Subject labels only annotate community membership after optimization. Similarly, the election groups describe observed contract purchases, not the personal political identity of wallet owners.
