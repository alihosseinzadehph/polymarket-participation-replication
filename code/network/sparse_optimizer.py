"""Sparse monotone ascent of weighted Barber bipartite modularity.

This is a separately specified heuristic, NOT an exact LPAwb+ or Leiden port.
See METHOD.md. All declared nodes and all positive supplied edges are retained.
The caller must verify that CSR and CSC encode the same weighted edge multiset.
No full-network run is performed by importing this module.
"""
from __future__ import annotations

import math
import time
import numpy as np
from numba import njit

METHOD_VERSION = "sparse-barber-monotone-1.1.2"


def move_gain(node_strength, old_edge_mass, new_edge_mass,
              old_opposite_strength, new_opposite_strength, total_weight,
              gamma=1.0):
    """Exact normalized delta Q for a one-side node move; pre-move strengths."""
    if total_weight <= 0:
        raise ValueError("Modularity is undefined for zero total weight")
    return ((new_edge_mass - old_edge_mass) / total_weight
            - gamma * (node_strength / total_weight)
            * ((new_opposite_strength - old_opposite_strength) / total_weight))


def merge_gain(cross_weight, wallet_strength_a, event_strength_a,
               wallet_strength_b, event_strength_b, total_weight, gamma=1.0):
    """Exact delta Q; cross_weight includes BOTH bipartite cross directions."""
    if total_weight <= 0:
        raise ValueError("Modularity is undefined for zero total weight")
    return (cross_weight / total_weight - gamma * (
        (wallet_strength_a / total_weight) * (event_strength_b / total_weight)
        + (wallet_strength_b / total_weight) * (event_strength_a / total_weight)))


def _normalization_inverse(raw_wallet_strength, total_weight, share):
    """Reject unsupported reciprocal scales before producing nonfinite scores.

    This input-domain check does not rescale amounts or change any optimizer
    trajectory for supported inputs. The reciprocals are the same float64
    operations used by the existing normalization kernels.
    """
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("Normalization requires a finite positive total weight")
    inverse = 1.0 / total_weight
    if not math.isfinite(inverse):
        raise ValueError("Unsupported numeric scale: reciprocal total weight is nonfinite")
    if share:
        if not np.all(np.isfinite(raw_wallet_strength)):
            raise ValueError("Normalization requires finite wallet strengths")
        smallest = float(np.min(raw_wallet_strength, where=raw_wallet_strength > 0,
                                initial=np.inf))
        if not math.isfinite(1.0 / smallest):
            raise ValueError("Unsupported numeric scale: reciprocal wallet strength is nonfinite")
    return inverse


def _validate_orientation(ptr, ids, weights, n_other):
    if ptr.ndim != 1 or ids.ndim != 1 or weights.ndim != 1:
        raise ValueError("Sparse arrays must be one-dimensional")
    if ptr.dtype.kind not in "iu" or ids.dtype.kind not in "iu":
        raise ValueError("Pointers and endpoint IDs must be integers")
    if weights.dtype != np.dtype("float64"):
        raise ValueError("Use float64 weights; implicit edge-sized copies are prohibited")
    if len(ptr) == 0 or ptr[0] != 0 or ptr[-1] != len(ids) or len(ids) != len(weights):
        raise ValueError("Invalid sparse pointers or lengths")
    if np.any(ptr[1:] < ptr[:-1]):
        raise ValueError("Sparse pointers must be nondecreasing")
    for lo in range(0, len(ids), 1_000_000):
        ii, ww = ids[lo:lo + 1_000_000], weights[lo:lo + 1_000_000]
        if len(ii) and (ii.min() < 0 or ii.max() >= n_other):
            raise ValueError("Endpoint ID out of bounds")
        if np.any(~np.isfinite(ww)) or np.any(ww <= 0):
            raise ValueError("Stored weights must be finite and strictly positive")


@njit(cache=True, nogil=True)
def _raw_row_strength(ptr, weights):
    out = np.zeros(len(ptr) - 1, np.float64)
    for u in range(len(out)):
        value, correction = 0.0, 0.0
        for j in range(ptr[u], ptr[u + 1]):
            y = weights[j] - correction
            temp = value + y
            correction = (temp - value) - y
            value = temp
        out[u] = value
    return out


@njit(cache=True, nogil=True)
def _event_strength_csr(ptr, ids, weights, raw_wallet, ne, share):
    result = np.zeros(ne, np.float64)
    errors = np.zeros(ne, np.float64)
    for u in range(len(ptr) - 1):
        factor = 1.0
        if share and raw_wallet[u] > 0:
            factor = 1.0 / raw_wallet[u]
        for j in range(ptr[u], ptr[u + 1]):
            e = ids[j]
            y = weights[j] * factor - errors[e]
            temp = result[e] + y
            errors[e] = (temp - result[e]) - y
            result[e] = temp
    return result


@njit(cache=True, nogil=True)
def _add_compensated(values, index, x):
    """Neumaier two-part sum, retaining small terms across large subtraction."""
    previous = values[0, index]
    total = previous + x
    if abs(previous) >= abs(x):
        values[1, index] += (previous - total) + x
    else:
        values[1, index] += (x - total) + previous
    values[0, index] = total


@njit(cache=True, nogil=True)
def _group_state(wl, el, sw, se, n):
    S = np.zeros((2, n), np.float64)
    T = np.zeros((2, n), np.float64)
    count = np.zeros((3, n), np.int32)  # all, wallet, event node counts
    free = np.empty(n, np.int32)
    for u in range(len(wl)):
        _add_compensated(S, wl[u], sw[u])
        count[0, wl[u]] += 1
        count[1, wl[u]] += 1
    for e in range(len(el)):
        _add_compensated(T, el[e], se[e])
        count[0, el[e]] += 1
        count[2, el[e]] += 1
    nf = 0
    for c in range(n):
        if count[0, c] == 0:
            free[nf] = c
            nf += 1
    return S, T, count, free, nf


@njit(cache=True, nogil=True)
def _quality(ptr, ids, weights, raw_wallet, wl, el, S, T, invM, share, gamma):
    internal, error = 0.0, 0.0
    for u in range(len(wl)):
        factor = invM
        if share and raw_wallet[u] > 0:
            factor /= raw_wallet[u]
        for j in range(ptr[u], ptr[u + 1]):
            if wl[u] == el[ids[j]]:
                y = weights[j] * factor - error
                temp = internal + y
                error = (temp - internal) - y
                internal = temp
    null, error = 0.0, 0.0
    for c in range(S.shape[1]):
        y = (S[0, c] + S[1, c]) * (T[0, c] + T[1, c]) - error
        temp = null + y
        error = (temp - null) - y
        null = temp
    return internal - gamma * null


def modularity(wallet_indptr, wallet_events, wallet_weights, wallet_labels,
               event_labels, *, weight_mode="amount", gamma=1.0):
    """Independent complete-CSR objective evaluation; no dense projection."""
    if weight_mode not in ("amount", "wallet_share") or not np.isfinite(gamma) or gamma < 0:
        raise ValueError("Invalid weight mode or gamma")
    nw, ne = len(wallet_indptr) - 1, len(event_labels)
    if len(wallet_labels) != nw:
        raise ValueError("Membership dimension mismatch")
    _validate_orientation(wallet_indptr, wallet_events, wallet_weights, ne)
    wl = np.asarray(wallet_labels)
    el = np.asarray(event_labels)
    if wl.dtype.kind not in "iu" or el.dtype.kind not in "iu":
        raise ValueError("Memberships must be integers")
    # Compress labels without imposing a target number of groups.
    labels, inv = np.unique(np.concatenate((wl, el)), return_inverse=True)
    wl, el = inv[:nw].astype(np.int32), inv[nw:].astype(np.int32)
    raw = _raw_row_strength(wallet_indptr, wallet_weights)
    share = weight_mode == "wallet_share"
    M = float(np.count_nonzero(raw)) if share else math.fsum(raw)
    if M <= 0:
        raise ValueError("Modularity is undefined for zero total weight")
    invM = _normalization_inverse(raw, M, share)
    sw = (raw > 0).astype(np.float64) / M if share else raw / M
    se = _event_strength_csr(wallet_indptr, wallet_events, wallet_weights, raw, ne, share) / M
    S, T, _, _, _ = _group_state(wl, el, sw, se, nw + ne)
    return float(_quality(wallet_indptr, wallet_events, wallet_weights, raw,
                          wl, el, S, T, invM, share, gamma))


@njit(cache=True, nogil=True)
def _wallet_factor_cache(raw_wallet, invM):
    """Cache the SAME scalar invM/raw operation formerly repeated per edge.

    No fastmath, factor reassociation, rounding change or edge-weight mutation.
    Empty rows have no edges; their sentinel matches the old wallet-side branch.
    """
    factors = np.empty(len(raw_wallet), np.float64)
    for u in range(len(raw_wallet)):
        factors[u] = invM / raw_wallet[u] if raw_wallet[u] > 0 else invM
    return factors


@njit(cache=True, nogil=True)
def _move_batch(ptr, ids, weights, labels, opposite_labels, node_strength,
                own_group, opposite_group, count, free, nf, order, start, end,
                raw_wallet, wallet_side, share, invM, gamma, tolerance,
                tally, seen, touched, wallet_factors=None):
    moves, gain_total, gain_error = 0, 0.0, 0.0
    for oi in range(start, end):
        u = order[oi]
        if node_strength[u] <= 0:
            continue
        nt = 0
        for j in range(ptr[u], ptr[u + 1]):
            v = ids[j]
            c = opposite_labels[v]
            if not seen[c]:
                seen[c] = True
                touched[nt] = c
                nt += 1
            factor = invM
            if share:
                if wallet_factors is None:
                    factor /= raw_wallet[u if wallet_side else v]
                else:
                    factor = wallet_factors[u if wallet_side else v]
            _add_compensated(tally, c, weights[j] * factor)
        old = labels[u]
        old_mass = tally[0, old] + tally[1, old]
        old_null = gamma * node_strength[u] * (opposite_group[0, old] + opposite_group[1, old])
        old_score = old_mass - old_null
        best, best_score = old, old_score
        for p in range(nt):
            c = touched[p]
            score = (tally[0, c] + tally[1, c]) - gamma * node_strength[u] * (opposite_group[0, c] + opposite_group[1, c])
            if score > best_score or (score == best_score and best != old and c < best):
                best, best_score = c, score
        # Every non-neighbor group has non-positive score. A free singleton
        # supplies score zero, so neighbor/current/empty candidates are complete.
        use_empty = best_score < 0.0 and count[0, old] > 1 and nf > 0
        if use_empty:
            best_score = 0.0
        new_mass = 0.0 if use_empty else tally[0, best] + tally[1, best]
        new_null = 0.0 if use_empty else gamma * node_strength[u] * (opposite_group[0, best] + opposite_group[1, best])
        delta = best_score - old_score
        scale = max(abs(old_mass), abs(old_null), abs(new_mass), abs(new_null))
        if delta > tolerance * scale and (use_empty or best != old):
            if use_empty:
                nf -= 1
                best = free[nf]
            labels[u] = best
            _add_compensated(own_group, old, -node_strength[u])
            _add_compensated(own_group, best, node_strength[u])
            side_index = 1 if wallet_side else 2
            count[0, old] -= 1
            count[0, best] += 1
            count[side_index, old] -= 1
            count[side_index, best] += 1
            if count[side_index, old] == 0:
                own_group[0, old] = 0.0
                own_group[1, old] = 0.0
            if count[0, old] == 0:
                opposite_group[0, old] = 0.0
                opposite_group[1, old] = 0.0
                free[nf] = old
                nf += 1
            moves += 1
            y = delta - gain_error
            tmp = gain_total + y
            gain_error = (tmp - gain_total) - y
            gain_total = tmp
        for p in range(nt):
            c = touched[p]
            tally[0, c] = 0.0
            tally[1, c] = 0.0
            seen[c] = False
    return nf, moves, gain_total


@njit(cache=True, nogil=True)
def _member_lists(wl, el, n):
    wh = np.full(n, -1, np.int32)
    eh = np.full(n, -1, np.int32)
    wn = np.full(len(wl), -1, np.int32)
    en = np.full(len(el), -1, np.int32)
    for u in range(len(wl)):
        c = wl[u]
        wn[u] = wh[c]
        wh[c] = u
    for e in range(len(el)):
        c = el[e]
        en[e] = eh[c]
        eh[c] = e
    return wh, eh, wn, en


@njit(cache=True, nogil=True)
def _best_mergers(wp, wi, ww, ep, ei, ew, raw, wl, el, S, T,
                  wh, eh, wn, en, start, end, share, invM, gamma, tolerance,
                  tally, seen, touched, best, gains, wallet_factors=None):
    for c in range(start, end):
        nt = 0
        u = wh[c]
        while u >= 0:
            factor = invM
            if share:
                if wallet_factors is None:
                    factor = invM / raw[u] if raw[u] > 0 else invM
                else:
                    factor = wallet_factors[u]
            for j in range(wp[u], wp[u + 1]):
                d = el[wi[j]]
                if d != c:
                    if not seen[d]:
                        seen[d] = True
                        touched[nt] = d
                        nt += 1
                    _add_compensated(tally, d, ww[j] * factor)
            u = wn[u]
        e = eh[c]
        while e >= 0:
            for j in range(ep[e], ep[e + 1]):
                u = ei[j]
                d = wl[u]
                if d != c:
                    if not seen[d]:
                        seen[d] = True
                        touched[nt] = d
                        nt += 1
                    factor = invM
                    if share:
                        factor = invM / raw[u] if wallet_factors is None else wallet_factors[u]
                    _add_compensated(tally, d, ew[j] * factor)
            e = en[e]
        best[c], gains[c] = -1, 0.0
        for p in range(nt):
            d = touched[p]
            null = gamma * ((S[0, c] + S[1, c]) * (T[0, d] + T[1, d])
                            + (S[0, d] + S[1, d]) * (T[0, c] + T[1, c]))
            mass = tally[0, d] + tally[1, d]
            gain = mass - null
            if gain > tolerance * max(abs(mass), abs(null)):
                if gain > gains[c] or (gain == gains[c] and d < best[c]):
                    best[c], gains[c] = d, gain
        for p in range(nt):
            d = touched[p]
            tally[0, d] = 0.0
            tally[1, d] = 0.0
            seen[d] = False


@njit(cache=True, nogil=True)
def _apply_mergers(wl, el, best, gains):
    mapping = np.arange(len(best), dtype=np.int32)
    merges, delta = 0, 0.0
    for c in range(len(best)):
        d = best[c]
        if d > c and best[d] == c:
            mapping[d] = c
            merges += 1
            delta += gains[c]
    for u in range(len(wl)):
        wl[u] = mapping[wl[u]]
    for e in range(len(el)):
        el[e] = mapping[el[e]]
    return merges, delta


def optimize_bipartite(wallet_indptr, wallet_events, wallet_weights,
                       event_indptr, event_wallets, event_weights, *,
                       weight_mode="amount", seed=20260913, gamma=1.0,
                       max_sweeps=100, max_merge_rounds=50, tolerance=1e-12,
                       max_seconds=None, callback=None, batch_nodes=50_000,
                       merge_batch_groups=10_000, initial_wallet_labels=None,
                       initial_event_labels=None, validate=True,
                       connectivity_refinement=True):
    """Optimize all supplied nodes/links, returning labels and explicit status.

    Both orientations contain RAW float64 amounts. wallet_share normalizes on
    demand; no extra edge-sized weight array is allocated. max_sweeps counts
    all wallet+event sweeps across merger rounds. Time limits are checked BETWEEN
    compiled batches/stages, not within an individual high-degree node or scan.
    callback receives JSON-safe metadata only, never full arrays. Exceptions are
    propagated; the caller owns checkpoint persistence. Resume labels continue
    optimization but do not reproduce an uninterrupted RNG trajectory.
    """
    started = time.monotonic()
    wp, wi, ww = wallet_indptr, wallet_events, wallet_weights
    ep, ei, ew = event_indptr, event_wallets, event_weights
    nw, ne = len(wp) - 1, len(ep) - 1
    n = nw + ne
    if weight_mode not in ("amount", "wallet_share"):
        raise ValueError("Unknown weight mode")
    if not np.isfinite(gamma) or gamma < 0 or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Gamma and tolerance must be finite and nonnegative")
    if min(max_sweeps, max_merge_rounds) < 0 or min(batch_nodes, merge_batch_groups) <= 0:
        raise ValueError("Invalid iteration/batch limits")
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds < 0):
        raise ValueError("Invalid time limit")
    if n >= 2 ** 31 or nw < 0 or ne < 0 or len(wi) != len(ei):
        raise ValueError("Invalid dimensions or int32 node capacity exceeded")
    if validate:
        _validate_orientation(wp, wi, ww, ne)
        _validate_orientation(ep, ei, ew, nw)
    share = weight_mode == "wallet_share"
    rng = np.random.default_rng(seed)
    if (initial_wallet_labels is None) != (initial_event_labels is None):
        raise ValueError("Supply both resume membership arrays")
    if initial_wallet_labels is None:
        labels = np.arange(n, dtype=np.int32)
        rng.shuffle(labels)  # Seeded tie priorities, no chosen cluster count.
        wl, el = labels[:nw].copy(), labels[nw:].copy()
        del labels
    else:
        a, b = np.asarray(initial_wallet_labels), np.asarray(initial_event_labels)
        if len(a) != nw or len(b) != ne or a.dtype.kind not in "iu" or b.dtype.kind not in "iu":
            raise ValueError("Invalid resume memberships")
        _, inv = np.unique(np.concatenate((a, b)), return_inverse=True)
        wl, el = inv[:nw].astype(np.int32), inv[nw:].astype(np.int32)
    history = []
    def emit(kind, **kwargs):
        item = {"stage": kind, "elapsed_seconds": time.monotonic() - started, **kwargs}
        history.append(item)
        if callback is not None:
            callback(item.copy())
    def expired():
        return max_seconds is not None and time.monotonic() - started >= max_seconds
    raw = _raw_row_strength(wp, ww)
    M = float(np.count_nonzero(raw)) if share else math.fsum(raw)
    emit("strengths", n_wallets=nw, n_events=ne, n_edges=len(wi), total_weight=M)
    if not np.isfinite(M):
        raise ValueError("Total weight overflow")
    if M <= 0:
        return {"method": METHOD_VERSION, "wallet_labels": wl, "event_labels": el,
                "Q": None, "converged": False, "stop_reason": "zero_weight_undefined",
                "history": history, "n_communities": n, "n_sweeps": 0,
                "n_merge_rounds": 0, "weight_mode": weight_mode, "seed": int(seed),
                "gamma": gamma, "total_weight": 0.0}
    invM = _normalization_inverse(raw, M, share)
    wallet_factors = _wallet_factor_cache(raw, invM) if share else None
    sw = (raw > 0).astype(np.float64) * invM if share else raw * invM
    se = _event_strength_csr(wp, wi, ww, raw, ne, share) * invM
    S, T, count, free, nf = _group_state(wl, el, sw, se, n)
    q = float(_quality(wp, wi, ww, raw, wl, el, S, T, invM, share, gamma))
    tally = np.zeros((2, n), np.float64)
    seen = np.zeros(n, np.bool_)
    touched = np.empty(n, np.int32)
    wo, eo = np.arange(nw, dtype=np.int32), np.arange(ne, dtype=np.int32)
    emit("initial", Q=q, n_communities=int(np.count_nonzero(count[0])))
    sweeps, merge_rounds, extra_components_total = 0, 0, 0
    reason, converged = "max_sweeps", False
    while sweeps < max_sweeps:
        if expired():
            reason = "max_seconds"
            break
        sweep_moves, predicted_delta = 0, 0.0
        interrupted = False
        for side, ptr, ids, weights, labels, opposite, strength, own, other, order in (
            ("wallet", wp, wi, ww, wl, el, sw, S, T, wo),
            ("event", ep, ei, ew, el, wl, se, T, S, eo)):
            rng.shuffle(order)
            for lo in range(0, len(order), batch_nodes):
                if expired():
                    interrupted = True
                    break
                hi = min(len(order), lo + batch_nodes)
                nf, moves, delta = _move_batch(
                    ptr, ids, weights, labels, opposite, strength, own, other,
                    count, free, nf, order, lo, hi, raw, side == "wallet", share,
                    invM, gamma, tolerance, tally, seen, touched, wallet_factors)
                sweep_moves += moves
                predicted_delta += delta
                emit("move_batch", sweep=sweeps + 1, side=side, nodes_done=hi,
                     nodes_total=len(order), moves=int(moves), predicted_delta_Q=float(delta))
            if interrupted:
                break
        old_q = q
        S, T, count, free, nf = _group_state(wl, el, sw, se, n)
        q = float(_quality(wp, wi, ww, raw, wl, el, S, T, invM, share, gamma))
        if q < old_q - 1e-10 * max(1.0, abs(old_q), abs(q)):
            raise ArithmeticError("Recomputed Q decreased after node moves")
        sweeps += 1
        emit("sweep", sweep=sweeps, complete=not interrupted, Q=q,
             delta_Q=q - old_q, predicted_delta_Q=predicted_delta,
             moves=int(sweep_moves), n_communities=int(np.count_nonzero(count[0])))
        if interrupted or expired():
            reason = "max_seconds"
            break
        if sweep_moves:
            continue
        # A complete zero-move sweep is required before testing group mergers.
        if connectivity_refinement:
            from connectivity import split_disconnected
            split = split_disconnected(wp, wi, wl, el)
            if split["extra_components"] > 0:
                wl, el = split["wallet_labels"], split["event_labels"]
                S, T, count, free, nf = _group_state(wl, el, sw, se, n)
                old_q = q
                q = float(_quality(wp, wi, ww, raw, wl, el, S, T, invM, share, gamma))
                if q < old_q - 1e-10 * max(1.0, abs(old_q), abs(q)):
                    raise ArithmeticError("Recomputed Q decreased after component splitting")
                extra_components_total += split["extra_components"]
                emit("connectivity_split", sweep=sweeps, Q=q, delta_Q=q - old_q,
                     extra_components=split["extra_components"],
                     n_communities=split["n_components"], nodes_excluded=0, links_excluded=0)
                continue  # The refined partition must be tested by another sweep.
            emit("connectivity_check", sweep=sweeps, Q=q, extra_components=0,
                 n_communities=split["n_components"])
            del split
        if merge_rounds >= max_merge_rounds:
            reason = "max_merge_rounds"
            break
        wh, eh, wn, en = _member_lists(wl, el, n)
        best, gains = np.full(n, -1, np.int32), np.zeros(n, np.float64)
        interrupted = False
        for lo in range(0, n, merge_batch_groups):
            if expired():
                interrupted = True
                break
            hi = min(n, lo + merge_batch_groups)
            _best_mergers(wp, wi, ww, ep, ei, ew, raw, wl, el, S, T, wh, eh,
                          wn, en, lo, hi, share, invM, gamma, tolerance,
                          tally, seen, touched, best, gains, wallet_factors)
            emit("merge_scan", merge_round=merge_rounds + 1, groups_done=hi, groups_total=n)
        del wh, eh, wn, en
        if interrupted:
            reason = "max_seconds"
            break
        merges, predicted = _apply_mergers(wl, el, best, gains)
        positive_candidates = int(np.count_nonzero(best >= 0))
        del best, gains
        merge_rounds += 1
        old_q = q
        S, T, count, free, nf = _group_state(wl, el, sw, se, n)
        q = float(_quality(wp, wi, ww, raw, wl, el, S, T, invM, share, gamma))
        if q < old_q - 1e-10 * max(1.0, abs(old_q), abs(q)):
            raise ArithmeticError("Recomputed Q decreased after mergers")
        emit("merge", merge_round=merge_rounds, Q=q, delta_Q=q - old_q,
             predicted_delta_Q=float(predicted), merges=int(merges),
             positive_candidate_groups=positive_candidates,
             n_communities=int(np.count_nonzero(count[0])))
        if merges == 0:
            if positive_candidates:
                reason = "no_mutual_pair_numerical_or_tie_stall"
            else:
                reason, converged = "no_positive_node_move_or_pair_merge", True
            break
    emit("finished", Q=q, converged=converged, stop_reason=reason)
    return {"method": METHOD_VERSION, "wallet_labels": wl, "event_labels": el,
            "Q": q, "converged": converged, "stop_reason": reason,
            "history": history, "n_communities": int(np.count_nonzero(count[0])),
            "n_sweeps": sweeps, "n_merge_rounds": merge_rounds,
            "weight_mode": weight_mode, "seed": int(seed), "gamma": float(gamma),
            "total_weight": M, "tolerance": tolerance,
            "zero_strength_wallets": int(np.count_nonzero(raw == 0)),
            "zero_strength_events": int(np.count_nonzero(se == 0)),
            "membership_inference_for_zero_strength_nodes": False,
            "connectivity_refinement": bool(connectivity_refinement),
            "extra_components_created": int(extra_components_total),
            "wallet_factor_cache_bytes": int(wallet_factors.nbytes) if share else 0,
            "connected_communities_guaranteed": bool(converged and connectivity_refinement)}
