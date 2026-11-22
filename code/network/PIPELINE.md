# Run one complete-population community analysis

`pipeline.py` runs one specified dataset, one weighting mode and one seed. It does not download data, select the largest events, remove low-activity wallets, or start additional seeds/modes. Existing output directories containing files are refused.

## Running the optimization on prepared sparse storage

From this directory, with the dependencies listed in `requirements.txt` and external storage prepared using `sparse_storage.py`, run:

```powershell
python .\pipeline.py --storage .\storage_all --output .\amount_pipeline_seed_20260913 --mode amount --seed 20260913
```

The input is the full indexed CSR/CSC store, **not** the small quotient graph used to draw the atlas. No credentials are required. Missing dependencies stop the run during preflight. Choose a new output directory for every run.

For a separate equal-wallet analysis, use `--mode wallet_share` and a different output directory. This is an explicit second run; it is never launched automatically. Raw mode weights participation by amount. Wallet-share mode gives each positive-strength wallet total weight one across its complete event history.

## Recorded sequence and budgets

| Stage | Operation on the same complete population | Default search budget |
|---|---|---|
| `stage1` | Sparse original-node optimization | 30 sweeps; 100 merger rounds; 600 seconds |
| `coarse` | Sum all links by current groups, retain loops, run directed Leiden, expand to all original nodes | 30 native iterations; 180 seconds for native search |
| `final` | Resume original-node optimization from expanded quotient labels, including connected-component refinement | 150 sweeps; 100 merger rounds; 600 seconds |

These are computational budgets, **not a target number of communities**. They are written to `pipeline_plan.json` before data processing. Limits are checked at the stages' documented boundaries: loading, compilation, complete objective scans, quotient aggregation, and a native iteration can extend elapsed time beyond the search budgets. They are not hard limits on total runtime.

Override them with `--stage1-sweeps`, `--stage1-seconds`, `--stage1-merge-rounds`, `--coarse-iterations`, `--coarse-seconds`, `--final-sweeps`, `--final-seconds`, and `--final-merge-rounds`. `--reserve-gib` defaults to 3 GiB free-memory reserve, checked by the stage runners. The existing quotient runner uses a 2 GiB DuckDB buffer-manager limit, a 4 GiB spill cap and a 3 GiB estimated graph-stage budget; these are recorded, not treated as guaranteed process peaks. A quotient that fails the graph-size gate is not filtered: its unchanged complete memberships continue to the final stage, and the skipped status is retained.

`--verify-storage-hashes` optionally rehashes all stored arrays in stage1, adding a complete sequential read. Every run records the storage manifest hash and expected array hashes, and checks sizes and modification timestamps between stages. Input storage must remain unchanged during the run.

## Outputs and interpretation

- `pipeline_plan.json`: immutable initial schedule, exact commands, input manifest, array identities, source hashes and declared scope.
- `code/`: frozen copies of the sources actually executed by all stages. Stage folders also keep their source snapshots.
- `software_versions.json`: selected interpreter and imported scientific package versions/locations. No full environment or API keys are recorded.
- `stage1/`, `coarse/`, `final/`: complete node memberships with original dimension IDs, checked objective values, community summaries and run-specific stopping details. The quotient stage also saves its complete aggregated graph and group mapping when it fits.
- `stage1.log`, `coarse.log`, `final.log`: relayed subprocess output; stages are strictly sequential.
- `pipeline_status.json`: current progress, failure or interruption state.
- `pipeline_result.json`: final summary with every stage-result hash and explicit convergence status.

Start interpretation from **`pipeline_result.json`**, not from a successful-looking figure or quotient result. `original_optimizer_converged=false` means the final original-node search stopped without establishing its stated local stopping condition. Increasing a budget and continuing can be a separate, documented decision; this pipeline never retries or starts a new full run by itself. Exit code 0 means the declared sequence finished and the final optimizer reported convergence; exit code 2 means all stages finished but the final optimizer did **not** converge. Other nonzero exits indicate interruption or failure; completed/partial outputs remain available and are not overwritten.

Even a converged result is a **preliminary descriptive partition**, not a publication-ready inference: one seed does not establish seed stability; quotient convergence is not full-node convergence; optimized modularity is not a p-value; and convergence is not proof of a global optimum. Null comparisons, seed sensitivity, substantive community annotations and temporal interpretation must be assessed separately. See `METHOD.md` and `COARSE_REFINEMENT.md` for the exact objective and algorithms.

The script does not modify the manuscript. No full pipeline is launched by `--help` or by importing the module.
