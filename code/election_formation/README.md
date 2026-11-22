# From fills to the four election-cohort inputs

`rebuild.py` independently regenerates the four September purchase-size cohorts and October subject-amount matrices from the frozen trade archive, then compares them to the released de-identified vectors **row by row**. It never writes wallet addresses, cohort membership lists or wallet lookups. It prints only counts, cutoffs and numerical agreement.

From the repository root, using Python 3.11+, DuckDB and NumPy:

```sh
python code/acquisition/archive.py plan --start 2024-07-31 --end 2024-11-01
# Optional: only if these files are not already present in RAW.
python code/acquisition/archive.py fetch --root RAW --start 2024-07-31 --end 2024-11-01
python code/election_formation/rebuild.py --raw RAW
```

The buffered file range matches the original extraction. The script verifies SHA-256 hashes of all 188 v1 shards (210,404,870 bytes) before applying exact UTC timestamps. No network access is used by `rebuild.py`. A new summary can be saved with `--report REPORT.json`; an existing file is not overwritten.

1. **September formation.** Join fills to the Trump and Harris YES contracts of the 2024 presidential winner event using its frozen 17-condition dictionary. Require a taker BUY of YES, the matching YES token, positive finite USDC, a finite price in $(0,1]$, and distinct maker/taker addresses. Sum taker purchases by wallet and candidate. Purchases of NO, sales and maker-side positions do not define the groups.
2. **Strata.** Compute linear 5th and 95th percentiles of September YES purchase amount among *all* 21,301 eligible wallets, including dual buyers. Include boundary ties. Only after calculating these pooled thresholds, select wallets that bought exactly one candidate's YES claim, giving Harris-only and Trump-only low and high groups. Labels describe trades, not personal beliefs.
3. **October follow-up.** Include positive finite mapped amounts on either side of a fill; count both appearances for eligible wallets and exclude **all 17 contracts of the focal event**, not all political events. Assign the frozen 21-label event taxonomy, then apply the archived 21-event Bitcoin reassignment to create the election analysis's 22-column view. Inactive wallets remain all-zero rows in the eligible matrices; active-only matrices remove those rows.
4. **Comparison.** Within each candidate/tier, wallets are internally sorted by lowercase address, matching the historical opaque-ID construction. Compare their vectors in this order, without matching or sorting on amounts, to both public reference CSVs in [Fig6_8_election](../../data/Source_Data/Fig6_8_election). Each zero/nonzero cell must match exactly; all amounts must meet the numerical tolerance. Indistinguishable all-zero rows cannot establish individual identity.

### Observed comparison

The [recorded verification](verified_refit.json) used all 188 local frozen shards and the published Gamma mapping. The pooled cutoffs matched at **1.09414 and 2,787.73554 USDC**. All 1,630 eligible and 971 active subject vectors agreed in their original group order up to floating-point summation. The largest cell difference was **0.000000289 USDC**.

| Cohort | Eligible | October active |
|---|---:|---:|
| Harris only, bottom 5% | 504 | 360 |
| Trump only, bottom 5% | 541 | 308 |
| Harris only, top 5% | 334 | 168 |
| Trump only, top 5% | 251 | 135 |

The refit does **not** claim byte-identical floats or recompute the bootstrap. Use the deposited **eligible** CSV for the historical seeded bootstrap; the **active-only** CSV is the permutation input. The replica is a portable adaptation of the archived formation/extraction and vector-construction programs, limited to the September–October comparison. It does not recreate other studies or August analyses.
