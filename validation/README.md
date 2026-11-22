# What was rerun for this release

| Check | Evidence and extent |
| --- | --- |
| Frozen observations | [Data checks](../data/verification.json): original Gamma file checksums; event dictionary, mapping and classifications; rank and amount comparisons with dated API responses; concentration inputs and degree counts. |
| Numerical results | [Compact reanalysis](compact_reanalysis.json): event/wallet concentration; three monetary tail fits, including empirical intervals and goodness of fit; election profile/network contrasts and bootstrap; subject-overlap null. [Point estimates and quotient](point_estimates_and_quotient.json) additionally checks the amount-weighted quotient totals and modularity **given the deposited partition summaries**. |
| Election randomization | [Nine-thousand-nine-hundred-ninety-nine label randomizations](election_randomization.json) from the released de-identified active-wallet vectors, distinct from the eligible-wallet bootstrap. |
| Election cohort formation and follow-up | [Rebuild from fixed trade partitions](../code/election_formation/): the 188 files covering the case study with date buffers were checked by SHA-256. Across the four exclusive high/low cohorts, the 1,630 eligible and 971 active rows matched the released vectors in order and zero pattern; the largest difference in any amount was below 0.000001 USDC. |
| Six main figures | All six rebuilt figure PDFs were rendered at 72 dpi using the same renderer as the six figure PDFs supplied to LaTeX. Dimensions and RGB pixel values matched for every figure. PDF binary equality across rendering environments is not required. |
| Upstream aggregation | The acquisition adapter was tested offline on the first real archive day, using the frozen condition mapping. See [the acquisition guide](../code/acquisition/README.md) for the counts and test limits. The complete 1.14-billion-fill aggregation and full-network community search were not repeated for packaging. |

Successful checks establish agreement with the archived samples and results, not completeness of the exchange history, political identity of wallets, or independent human validation of the taxonomy.
