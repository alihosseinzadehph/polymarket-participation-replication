# Optional full-population quotient refinement

## Purpose and status

`coarse_refine.py` provides an optional established multilevel operation: aggregate a current partition, optimize the complete directed quotient with Leiden, and expand its labels back to every original wallet and event. This is not a rank cutoff, a sampled network, a category substitution, or an edge backbone. The frozen sparse optimizer is unchanged. No full dataset is processed when this module is imported.

The motivation is computational: mutually best disjoint pair mergers can combine only one partner of a hub in each round. A quotient search can optimize movements of complete existing groups using a published implementation. It cannot split an existing group at this stage. Subsequent individual-node sweeps and connected-component checks on the original graph are required; quotient convergence alone is insufficient.

## Objective-preservation proof

Let A_ue be the chosen positive bipartite weights (raw amounts or row-normalized wallet shares), and let b(u), b(e) denote the current complete partition into K groups. Form the directed quotient

W_cd = sum_{u,e: b(u)=c,b(e)=d} A_ue.

Every original edge appears once in this sum. Keep W_cc as a self-loop. The quotient total weight remains M; its outgoing strength at c is the original wallet strength S_c, and its incoming strength is the original event strength T_c.

For any new partition h of these K quotient nodes, directed configuration modularity is

Q(h) = (1/M) sum_cd [W_cd - gamma S_c T_d/M] 1[h(c)=h(d)].

Substituting the definition of W and distributing the strength sums gives

Q(h) = (1/M) sum_ue [A_ue - gamma s_u t_e/M] 1[h(b(u))=h(b(e))].

This is exactly original weighted Barber modularity for the expanded coarsening. The equality holds for every coarsening, not merely for the optimizer output. It does not hold if self-loops or small quotient edges are discarded, if the quotient is symmetrized, or if an undirected null model replaces the directed one. The edge orientation is algebraic bookkeeping, not an inference about cash flow or causality.

The configuration objective follows [Barber, 2007](https://doi.org/10.1103/PhysRevE.76.066102) and [Leicht and Newman, 2008](https://doi.org/10.1103/PhysRevLett.100.118703). The identity above is the explicit application of aggregation to this study's bipartite objective. Repeated aggregation is a standard part of multilevel modularity algorithms: [Blondel et al., 2008](https://arxiv.org/abs/0803.0476), [Traag, Waltman and van Eck, 2019](https://doi.org/10.1038/s41598-019-41695-z).

## Implementation and resource rules

1. Compress existing membership IDs reversibly; no wallet or event is excluded. Read raw CSR edges in bounded batches. In `wallet_share` mode divide every edge by its original wallet's complete raw strength before aggregation. Do not renormalize quotient rows: that would change the objective.
2. Feed Arrow record batches directly to DuckDB. Aggregate all directed group pairs with `fsum(weight)` and retain the count of original pairs represented. No full expanded pair file or Python edge-tuple list is produced.
3. Default DuckDB settings are a 2 GiB buffer-manager limit and 4 GiB temporary-spill cap. Temporary files exist only in a newly created directory beneath the supplied work directory. The buffer-manager limit is not a guarantee of total process memory. Database aggregation is a complete stage; the native Leiden time budget does not bound its duration.
4. Count the complete quotient before allocating its full edge arrays. Estimate graph-stage memory conservatively as 216 bytes per quotient edge, 512 bytes per quotient vertex and 32 bytes per original node for mappings/auxiliary state. If this exceeds the default 3 GiB graph-stage budget, return `skipped_memory_gate`. No edge removal or smaller sampled run occurs. This heuristic is not a measured peak guarantee; system memory must also be monitored by the caller.
5. Close DuckDB before constructing igraph. Build a directed graph including loops. Use `RBConfigurationVertexPartition` with the same gamma, initialized to singleton quotient groups, and a specified random seed. Native quality divided by M must agree with an independently evaluated directed formula. Perform one native Leiden iteration at a time. A finite iteration/time limit is a bounded stop, not convergence. Native iterations cannot be interrupted by the helper's between-iteration time check.
6. Expand the quotient memberships to every original node. Recompute original Barber Q by scanning every original CSR edge, before and after refinement, and verify both quotient identities and nondecrease within the stated floating-point tolerances. Return the complete memberships for subsequent original-node refinement.

The relevant API and directed normalization are documented in the [official leidenalg reference](https://leidenalg.readthedocs.io/en/stable/reference.html#rbconfigurationvertexpartition). The implementation records igraph/leidenalg versions and iteration history. Float64 aggregation is not exact arithmetic; compensated sums and conservation checks limit and expose numerical discrepancy.

## Required validation and interpretation

Independent tests should enumerate small base partitions and their coarsenings, compare dense original Barber and directed quotient scores, check self-loop mass and original-edge counts, verify all nodes survive expansion, and test the resource refusal path. A successful quotient refinement proves neither a global optimum nor statistically significant or economically homogeneous communities. Sensitivity across seeds, weighting definitions, temporal windows and appropriate null models remains necessary. Any empirical report must distinguish the full-data representation, this coarsening-only search and later original-node refinement, with each stopping status preserved.
