"""Optional, objective-preserving quotient-graph Leiden refinement.

All original edges are summed by their current endpoint groups, including
self-loops. This is not a node-selection or graph-backbone operation. Only
coarsenings of the supplied partition are searched in this stage; subsequent
individual-node refinement remains the caller's responsibility.
"""
from __future__ import annotations

import math
from pathlib import Path
import tempfile
import time

import numpy as np
from numba import njit

COARSE_METHOD_VERSION = "barber-directed-quotient-leiden-1.0"
GIB = 2 ** 30


def estimate_graph_peak_bytes(n_vertices, n_edges, n_original_nodes=0):
    """Conservative implementation heuristic, NOT a measured bound/guarantee."""
    return int(216 * n_edges + 512 * n_vertices + 32 * n_original_nodes)


@njit(cache=True, nogil=True)
def _fill_edge_batch(ptr, ids, weights, raw, wl, el, lo, hi, share):
    src = np.empty(hi - lo, np.int32)
    dst = np.empty(hi - lo, np.int32)
    value = np.empty(hi - lo, np.float64)
    u = np.searchsorted(ptr, lo, side="right") - 1
    for j in range(lo, hi):
        while ptr[u + 1] <= j:
            u += 1
        src[j - lo] = wl[u]
        dst[j - lo] = el[ids[j]]
        value[j - lo] = weights[j] / raw[u] if share else weights[j]
    return src, dst, value


def _compress_labels(wallet_labels, event_labels, n_wallets):
    w, e = np.asarray(wallet_labels), np.asarray(event_labels)
    if w.ndim != 1 or e.ndim != 1 or len(w) != n_wallets:
        raise ValueError("Invalid complete membership dimensions")
    if w.dtype.kind not in "iu" or e.dtype.kind not in "iu":
        raise ValueError("Membership labels must be integers")
    original, inverse = np.unique(np.concatenate((w, e)), return_inverse=True)
    if len(original) >= 2 ** 31:
        raise ValueError("Quotient exceeds int32 label capacity")
    return (original, inverse[:len(w)].astype(np.int32),
            inverse[len(w):].astype(np.int32))


def aggregate_coarse(wallet_indptr, wallet_events, wallet_weights,
                     wallet_labels, event_labels, *, work_directory,
                     weight_mode="amount", memory_limit_bytes=2 * GIB,
                     max_temp_bytes=4 * GIB, graph_memory_budget_bytes=3 * GIB,
                     batch_edges=1_000_000, threads=2, callback=None,
                     validate=True):
    """Stream the COMPLETE CSR into bounded DuckDB aggregation.

    The unique temporary directory contains only this call's spill files and is
    removed on exit. No expanded source-pair file is written. DuckDB's configured
    memory limit applies to its buffer manager, not all Python/process memory.
    A graph-memory gate is applied after counting every quotient edge but BEFORE
    allocating its complete edge arrays or constructing igraph. Failure to fit
    is reported without discarding any links. Out-of-memory/disk-limit errors
    propagate; they never trigger sampling or an automatic less complete run.
    """
    import duckdb
    import pyarrow as pa
    from sparse_optimizer import _raw_row_strength, _validate_orientation

    start = time.monotonic()
    if weight_mode not in ("amount", "wallet_share"):
        raise ValueError("Unknown weight mode")
    if min(memory_limit_bytes, max_temp_bytes, graph_memory_budget_bytes,
           batch_edges, threads) <= 0:
        raise ValueError("Resource limits and batch size must be positive")
    nw, ne = len(wallet_indptr) - 1, len(event_labels)
    if validate:
        _validate_orientation(wallet_indptr, wallet_events, wallet_weights, ne)
    original, wg, eg = _compress_labels(wallet_labels, event_labels, nw)
    raw = _raw_row_strength(wallet_indptr, wallet_weights)
    share = weight_mode == "wallet_share"
    expected_mass = float(np.count_nonzero(raw)) if share else math.fsum(raw)
    if not math.isfinite(expected_mass) or expected_mass <= 0:
        raise ValueError("Positive finite total weight is required")
    n_edges, n_vertices = len(wallet_events), len(original)
    base = {"method": COARSE_METHOD_VERSION, "n_vertices": n_vertices,
            "wallet_groups": wg, "event_groups": eg,
            "original_group_labels": original, "weight_mode": weight_mode,
            "n_original_wallets": nw, "n_original_events": ne,
            "n_original_edges": n_edges, "expected_total_weight": expected_mass,
            "nodes_excluded": 0, "links_excluded": 0,
            "self_loops_retained": True, "directed": True,
            "duckdb_memory_limit_bytes": int(memory_limit_bytes),
            "duckdb_spill_limit_bytes": int(max_temp_bytes),
            "graph_memory_budget_bytes": int(graph_memory_budget_bytes)}
    schema = pa.schema([("src", pa.int32()), ("dst", pa.int32()), ("weight", pa.float64())])
    def batches():
        for lo in range(0, n_edges, batch_edges):
            hi = min(n_edges, lo + batch_edges)
            s, d, v = _fill_edge_batch(wallet_indptr, wallet_events, wallet_weights,
                                     raw, wg, eg, lo, hi, share)
            yield pa.RecordBatch.from_arrays([pa.array(s), pa.array(d), pa.array(v)], schema=schema)
            if callback is not None:
                callback({"stage": "coarse_input", "edges_done": hi,
                          "edges_total": n_edges, "elapsed_seconds": time.monotonic() - start})
    directory = Path(work_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="coarse_spill_", dir=directory) as temporary:
        resolved_temporary = Path(temporary).resolve()
        if resolved_temporary.parent != directory or not resolved_temporary.name.startswith("coarse_spill_"):
            raise ValueError("Temporary spill directory is outside the intended work directory")
        con = duckdb.connect(":memory:")
        try:
            con.execute("SET memory_limit = ?", [str(int(memory_limit_bytes)) + "B"])
            con.execute("SET temp_directory = ?", [temporary])
            con.execute("SET max_temp_directory_size = ?", [str(int(max_temp_bytes)) + "B"])
            con.execute("SET threads = ?", [int(threads)])
            con.execute("SET preserve_insertion_order = false")
            reader = pa.RecordBatchReader.from_batches(schema, batches())
            con.register("complete_edge_stream", reader)
            con.execute("""CREATE TEMP TABLE quotient AS
                SELECT src,dst,fsum(weight) AS weight,count(*)::UBIGINT AS original_pairs
                FROM complete_edge_stream GROUP BY src,dst""")
            con.unregister("complete_edge_stream")
            row = con.execute("""SELECT count(*),fsum(weight),sum(original_pairs),
                count(*) FILTER(WHERE src=dst),
                coalesce(fsum(weight) FILTER(WHERE src=dst),0),
                count(*) FILTER(WHERE NOT isfinite(weight) OR weight<=0)
                FROM quotient""").fetchone()
            coarse_edges, mass, represented_edges, loops, loop_mass, invalid = row
            estimate = estimate_graph_peak_bytes(n_vertices, coarse_edges, nw + ne)
            base.update(n_edges=int(coarse_edges), total_weight=float(mass),
                        represented_original_edges=int(represented_edges),
                        n_self_loops=int(loops), self_loop_weight=float(loop_mass),
                        estimated_graph_peak_bytes=estimate)
            if invalid or represented_edges != n_edges:
                raise ArithmeticError("Quotient lost edges or contains invalid aggregate weights")
            if not math.isclose(mass, expected_mass, rel_tol=2e-12, abs_tol=0.0):
                raise ArithmeticError("Quotient total weight does not match complete input")
            if callback is not None:
                callback({"stage": "coarse_aggregated", "n_vertices": n_vertices,
                          "n_edges": int(coarse_edges), "total_weight": float(mass),
                          "estimated_graph_peak_bytes": estimate,
                          "elapsed_seconds": time.monotonic() - start})
            if estimate > graph_memory_budget_bytes:
                return {**base, "status": "skipped_memory_gate",
                        "seconds": time.monotonic() - start}
            edges = np.empty((coarse_edges, 2), np.int64)
            weights = np.empty(coarse_edges, np.float64)
            offset = 0
            output = con.execute("SELECT src,dst,weight FROM quotient ORDER BY src,dst").fetch_record_batch(batch_edges)
            for batch in output:
                end = offset + batch.num_rows
                edges[offset:end, 0] = batch.column(0).to_numpy()
                edges[offset:end, 1] = batch.column(1).to_numpy()
                weights[offset:end] = batch.column(2).to_numpy()
                offset = end
            if offset != coarse_edges:
                raise ArithmeticError("Incomplete quotient materialization")
        finally:
            con.close()
    return {**base, "status": "ready", "edges": edges, "weights": weights,
            "seconds": time.monotonic() - start}


def coarse_modularity(edges, weights, n_vertices, membership, *, gamma=1.0):
    """Directed configuration quality / M, including loops exactly once."""
    labels = np.asarray(membership)
    if len(labels) != n_vertices or labels.ndim != 1 or labels.dtype.kind not in "iu":
        raise ValueError("Invalid quotient membership")
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("Invalid gamma")
    if edges.ndim != 2 or edges.shape != (len(weights), 2):
        raise ValueError("Invalid edge dimensions")
    if len(edges) and (edges.min() < 0 or edges.max() >= n_vertices):
        raise ValueError("Invalid endpoint IDs")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Positive finite weights are required")
    mass = math.fsum(weights)
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("Positive finite total weight is required")
    _, lab = np.unique(labels, return_inverse=True)
    k = int(lab.max()) + 1
    source, target = edges[:, 0], edges[:, 1]
    S = np.bincount(lab[source], weights=weights / mass, minlength=k)
    T = np.bincount(lab[target], weights=weights / mass, minlength=k)
    within = lab[source] == lab[target]
    return float(math.fsum(weights[within]) / mass - gamma * math.fsum(S * T))


def run_coarse_leiden(edges, weights, n_vertices, *, seed,
                      gamma=1.0, max_iterations=10, max_seconds=300,
                      callback=None, graph_memory_budget_bytes=3 * GIB):
    """Run a bounded published Leiden search on an already complete quotient.

    Native iterations cannot be interrupted by this function's time budget.
    An iteration/time cap never means convergence. Only zero native improvement
    is a convergence marker, limited to this quotient (not original node moves).
    """
    import igraph as ig
    import leidenalg as la
    if max_iterations < 0 or max_seconds < 0 or not np.isfinite(max_seconds):
        raise ValueError("Use finite nonnegative iteration/time budgets")
    estimate = estimate_graph_peak_bytes(n_vertices, len(weights))
    if estimate > graph_memory_budget_bytes:
        raise MemoryError("Complete quotient exceeds graph-memory gate; no links removed")
    initial = np.arange(n_vertices, dtype=np.int32)
    q0 = coarse_modularity(edges, weights, n_vertices, initial, gamma=gamma)
    mass = math.fsum(weights)
    started = time.monotonic()
    graph = ig.Graph(n=n_vertices, edges=edges, directed=True)
    # Do not simplify: loops encode all original within-group mass.
    partition = la.RBConfigurationVertexPartition(graph, weights=weights,
                                                 resolution_parameter=gamma)
    qnative = float(partition.quality() / mass)
    if not math.isclose(qnative, q0, rel_tol=1e-9, abs_tol=2e-10):
        raise ArithmeticError("Initial directed native/quotient objective mismatch")
    optimiser = la.Optimiser()
    optimiser.set_rng_seed(int(seed))
    history = [{"stage": "coarse_initial", "Q": q0, "n_communities": n_vertices}]
    reason, converged = "max_iterations", False
    for iteration in range(1, max_iterations + 1):
        if time.monotonic() - started >= max_seconds:
            reason = "max_seconds"
            break
        old = qnative
        native_delta = float(optimiser.optimise_partition(partition, n_iterations=1))
        qnative = float(partition.quality() / mass)
        if qnative < old - 2e-10:
            raise ArithmeticError("Native coarse objective decreased")
        item = {"stage": "coarse_leiden_iteration", "iteration": iteration,
                "Q": qnative, "delta_Q": qnative - old,
                "native_delta": native_delta, "n_communities": len(partition),
                "elapsed_seconds": time.monotonic() - started}
        history.append(item)
        if callback is not None:
            callback(item.copy())
        if native_delta == 0.0:
            reason, converged = "zero_native_improvement_on_quotient", True
            break
        if native_delta < 0.0:
            reason = "negative_native_delta_numerical_stall"
            break
    labels = np.asarray(partition.membership, dtype=np.int32)
    qfinal = coarse_modularity(edges, weights, n_vertices, labels, gamma=gamma)
    if not math.isclose(qfinal, qnative, rel_tol=1e-9, abs_tol=2e-10):
        raise ArithmeticError("Final directed native/quotient objective mismatch")
    if qfinal < q0 - 2e-10:
        raise ArithmeticError("Coarse refinement decreased complete objective")
    return {"membership": labels, "Q": qfinal, "Q_initial": q0,
            "delta_Q": qfinal - q0, "history": history, "converged": converged,
            "stop_reason": reason, "n_communities": len(np.unique(labels)),
            "seconds": time.monotonic() - started, "igraph_version": ig.__version__,
            "leidenalg_version": getattr(la, "__version__", getattr(la, "version", None)),
            "gamma": float(gamma), "seed": int(seed),
            "convergence_scope": "coarsenings_of_supplied_original_partition_only"}


def expand_labels(wallet_groups, event_groups, coarse_membership):
    labels = np.asarray(coarse_membership)
    return labels[wallet_groups].astype(np.int32), labels[event_groups].astype(np.int32)


def refine_coarse_leiden(wallet_indptr, wallet_events, wallet_weights,
                         wallet_labels, event_labels, *, work_directory, seed,
                         weight_mode="amount", gamma=1.0, max_iterations=10,
                         max_seconds=300, callback=None, return_quotient=False,
                         **aggregate_options):
    """Aggregate, run quotient Leiden, expand and verify against ALL CSR edges.

    Returned memberships must be passed to sparse individual-node/component
    refinement before claiming convergence on the original network. This helper
    intentionally does not run that additional stage automatically.
    max_seconds bounds the native-stage budget only; aggregation is separate.
    """
    from sparse_optimizer import modularity
    quotient = aggregate_coarse(wallet_indptr, wallet_events, wallet_weights,
                                wallet_labels, event_labels, work_directory=work_directory,
                                weight_mode=weight_mode, callback=callback, **aggregate_options)
    metadata = {k: v for k, v in quotient.items() if not isinstance(v, np.ndarray)}
    if quotient["status"] != "ready":
        return {"status": quotient["status"], "wallet_labels": np.asarray(wallet_labels).copy(),
                "event_labels": np.asarray(event_labels).copy(), "aggregation": metadata,
                "original_network_converged": False, "subsequent_sparse_refinement_required": True}
    q_before = modularity(wallet_indptr, wallet_events, wallet_weights, wallet_labels,
                          event_labels, weight_mode=weight_mode, gamma=gamma)
    result = run_coarse_leiden(quotient["edges"], quotient["weights"], quotient["n_vertices"],
                              seed=seed, gamma=gamma, max_iterations=max_iterations,
                              max_seconds=max_seconds, callback=callback,
                              graph_memory_budget_bytes=aggregate_options.get("graph_memory_budget_bytes", 3 * GIB))
    if not math.isclose(q_before, result["Q_initial"], rel_tol=1e-9, abs_tol=2e-10):
        raise ArithmeticError("Original initial Barber / quotient identity failed")
    w, e = expand_labels(quotient["wallet_groups"], quotient["event_groups"], result["membership"])
    q_after = modularity(wallet_indptr, wallet_events, wallet_weights, w, e,
                         weight_mode=weight_mode, gamma=gamma)
    if not math.isclose(q_after, result["Q"], rel_tol=1e-9, abs_tol=2e-10):
        raise ArithmeticError("Expanded original Barber / quotient identity failed")
    if q_after < q_before - 2e-10:
        raise ArithmeticError("Expanded refinement decreased original objective")
    output = {"status": "refined", "wallet_labels": w, "event_labels": e,
            "Q_before": q_before, "Q_after": q_after, "delta_Q": q_after - q_before,
            "aggregation": metadata, "coarse_result": result,
            "original_network_converged": False, "subsequent_sparse_refinement_required": True,
            "nodes_excluded": 0, "links_excluded": 0}
    if return_quotient:
        output.update(quotient_edges=quotient["edges"], quotient_weights=quotient["weights"],
                      original_group_labels=quotient["original_group_labels"],
                      wallet_groups=quotient["wallet_groups"], event_groups=quotient["event_groups"])
    return output
