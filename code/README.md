# Code and reproducibility

Run the commands below from the repository root in a Python 3.12 environment. Install the compact-analysis dependencies with `python -m pip install -r code/requirements.txt`. No API key or external download is required to analyze the included data. Outputs go to a new location; keep the released data unchanged.

| Question | Command | What is recomputed |
|---|---|---|
| Do the main numerical estimates follow from the released inputs? | `python code/check_analysis.py --output reproduced/estimates.json` | Event and wallet Ginis and top-1% shares; degree totals for both node sets; the three 1,000-observation tail MLEs, cutoffs and KS distances; six candidate-group divergences in each purchase stratum; the 210 expected subject overlaps from the binary configuration model. Validates the monetary observations against the complete event totals and the dated PnL/volume leaderboard lists. |
| Do the simulation results match? | `python code/check_analysis.py --tail-bootstrap --election-bootstrap --bicm-simulations --output reproduced/resampling.json` | For each monetary tail, 2,500 semiparametric goodness-of-fit draws and 1,000 nonparametric CI draws, refitting the cutoff each time. For the six-view election interval, 1,999 eligible-wallet resamples. For each of 210 subject pairs, 19,999 marginal overlap simulations, followed by Holm correction. Checks against archived estimates. |
| Does the direct randomization reproduce the election tests? | `python code/election/permutation_audit.py` | Permutes candidate labels within each purchase-size stratum; 9,999 draws. Writes the six-view results (including directional and two-sided tests), plus the older within-tier comparisons, to `reproduced/`. Uses the 971 active October wallets. The eligible-wallet bootstrap instead includes all 1,630 eligible rows in the original order, including inactive rows. |
| Are the published figures reproducible? | `python code/reproduce_figures.py --output reproduced/main_figures` | Six main PDFs from the released data: Lorenz curves, monetary tails, 21 subject shares, the 149-group atlas, two subject networks and the election networks (top row = top 5%). Original rendering logic; fonts available on the system can change the PDF bytes. Outputs only to a new directory; optional Poppler produces additional PNG previews. This command does not re-fit the models. |
| Does the frozen taxonomy classify the retained examples as expected? | `python code/taxonomy/check_frozen_cases.py` | Original versioned ordered rules, title/slug/tags classifier and 20 regression cases. This is a software check, not independent human validation. |

The event Gini uses 764,378 mapped events. The historical wallet concentration calculation includes 3,106,517 addresses; the graph degree tables include 3,106,514 mapped addresses. These are distinct denominators. The three Fig. 2 fits are conditional on the selected top 1,000 events or leaderboard entries; non-rejection is not a population-wide finding.

## What the graph code can and cannot recreate locally

`network/build_full_participation.py` is the historical quarter-by-quarter executed-fill aggregator. It requires external aligned fill partitions, a verification manifest, frozen wallet/event dimensions, a clean condition-to-event mapping and quarterly reconciliation inputs. Run `python code/network/build_full_participation.py --help` for its input/output flags. A download of contemporary platform metadata does not by itself recreate the historical universe. The compact summaries do not contain individual wallet–event edges.

The portable acquisition bridge in `acquisition/aggregate.py` writes a sorted `participation.parquet` and `dimensions.json` from user-supplied aligned fills and frozen mappings. To convert those external outputs to indexed CSR/CSC arrays:

```text
python code/network/sparse_storage.py --input WORK/participation.parquet --dimension-manifest WORK/dimensions.json --output CSR --scope full
```

Use a full-history input for `--scope full`; label a pilot with its actual scope. `network/pipeline.py` then runs original-node optimization, directed Leiden on the quotient, and original-node refinement. The package includes both weighting modes, synthetic objective tests, search budgets and stopping checks; see [the optimizer guide](network/PIPELINE.md). A complete market refit requires separately prepared full graph storage; it is **not** part of the compact checks. The frozen sparse pipeline uses Windows memory counters for its resource safety check. Its elapsed-time budgets can make results machine-dependent. The small-graph tests do not validate an optimum on the full graph. Core code for BiCM and its null simulations is in `network/subject_null.py`; `tails/estimators.py` contains the continuous CSN implementation used for the three main fits. Archived supplementary discrete results are not replayed by this command.

The archived *partition* can also be checked without refitting it. The compact checker uses the stored group strengths and internal weights to recompute the amount-weighted bipartite modularity at resolution one; it independently counts the 149 groups and the 1,289 distinct between-group pairs from the quotient Parquet file. For full membership data, `network/annotate_communities.py` and `network/annotate_wallet_share.py` contain the post-detection category accounting. They require a completed, converged external run and its event-statistics table; `python code/network/annotate_communities.py --self-test` exercises the alignment, ranking and category accounting without raw data.

## Original leaderboard collector

The actual standard-library collector and its offline tests are in `leaderboard/`. The analysis uses the dated snapshot distributed in `data/leaderboard/`; the live API cannot recreate September's ranks. To test the collector **without network access**, change into `code/leaderboard` and run `python -m unittest discover -s tests`. If a **new** snapshot is wanted, from that directory run:

```text
python -m polymarket_leaderboard collect --categories OVERALL --periods ALL --order-by PNL,VOL --max-offset 1000 --no-profiles --output-dir new-snapshot
```

This optional command queries the public API. Maximum offset 1,000 with a page size of 50 can return 1,050 rows per ordering; the analysis uses ranks 1–1,000, not all 1,050. The original collector stores public account/profile attributes in its local raw responses. Do not publish its unreviewed output; the released research tables omit wallet addresses, usernames and social handles. No fresh collection is necessary to use the paper's frozen numbers.
