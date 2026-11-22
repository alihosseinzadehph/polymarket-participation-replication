"""Independent numerical tests of the sparse bipartite optimizer.

Reference scores are evaluated directly from tiny dense matrices. No downloaded
or real-market data are used, and these tests do not establish a global optimum,
statistical significance, convergence on the full graph, or economic meaning.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import unittest
import sys

import numpy as np


def partitions(n):
    """Enumerate all set partitions as canonical restricted-growth strings."""
    if n == 0:
        yield np.empty(0, dtype=np.int32)
        return

    def visit(prefix, maximum):
        if len(prefix) == n:
            yield np.array(prefix, dtype=np.int32)
            return
        for group in range(maximum + 2):
            yield from visit(prefix + [group], max(maximum, group))

    yield from visit([0], 0)


def sparse_arrays(a):
    """Independently construct both sparse orientations from a tiny dense A."""
    a = np.asarray(a, dtype=np.float64)
    nw, ne = a.shape
    wi, ej = np.nonzero(a)
    wallet_ptr = np.concatenate(([0], np.cumsum(np.bincount(wi, minlength=nw)))).astype(np.int64)
    wallet_events = ej.astype(np.int32)
    wallet_weights = a[wi, ej].copy()
    order = np.lexsort((wi, ej))
    event_ptr = np.concatenate(([0], np.cumsum(np.bincount(ej, minlength=ne)))).astype(np.int64)
    event_wallets = wi[order].astype(np.int32)
    event_weights = a[wi[order], ej[order]].copy()
    return wallet_ptr, wallet_events, wallet_weights, event_ptr, event_wallets, event_weights


def normalized(a, mode):
    a = np.asarray(a, dtype=np.float64)
    if mode == "amount":
        return a.copy()
    strengths = a.sum(axis=1)
    return np.divide(a, strengths[:, None], out=np.zeros_like(a), where=strengths[:, None] > 0)


def dense_quality(a, wallet_labels, event_labels, gamma=1.0):
    """Direct matrix-form Barber score, independent of implementation kernels."""
    a = np.asarray(a, dtype=np.float64)
    total = float(a.sum())
    if total <= 0:
        return 0.0
    null = np.outer(a.sum(axis=1), a.sum(axis=0)) / total
    same = np.asarray(wallet_labels)[:, None] == np.asarray(event_labels)[None, :]
    return float(np.sum((a - gamma * null) * same) / total)


def dense_node_gain(a, wallet_labels, event_labels, node_type, node, target, gamma=1.0):
    before = dense_quality(a, wallet_labels, event_labels, gamma)
    w = np.array(wallet_labels, copy=True)
    e = np.array(event_labels, copy=True)
    if node_type == "wallet":
        w[node] = target
    else:
        e[node] = target
    return dense_quality(a, w, e, gamma) - before


def dense_merge_gain(a, wallet_labels, event_labels, source, target, gamma=1.0):
    before = dense_quality(a, wallet_labels, event_labels, gamma)
    w = np.array(wallet_labels, copy=True)
    e = np.array(event_labels, copy=True)
    w[w == source] = target
    e[e == source] = target
    return dense_quality(a, w, e, gamma) - before


def dense_induced_components(a, wallet_labels, event_labels):
    """Independent breadth-first components; no union-find or sparse kernels."""
    nw, ne = a.shape
    old = np.r_[wallet_labels, event_labels]
    adjacency = [[] for _ in range(nw + ne)]
    for u in range(nw):
        for v in range(ne):
            if a[u, v] > 0 and old[u] == old[nw + v]:
                adjacency[u].append(nw + v)
                adjacency[nw + v].append(u)
    result = np.full(nw + ne, -1, dtype=np.int32)
    label = 0
    for root in range(nw + ne):
        if result[root] >= 0:
            continue
        todo = [root]
        result[root] = label
        for node in todo:
            for neighbor in adjacency[node]:
                if result[neighbor] < 0:
                    result[neighbor] = label
                    todo.append(neighbor)
        label += 1
    return result


TOY = np.array([[0.25, 1.7, 0.0], [0.0, 3.125, 0.0625], [0.0, 0.0, 5.9]])


class IndependentReferenceTests(unittest.TestCase):
    """Checks mathematical candidate restrictions without implementation code."""

    def test_partition_generator(self):
        self.assertEqual(sum(1 for _ in partitions(6)), 203)

    def test_neighbor_or_empty_covers_best_single_node_move(self):
        # A non-neighbor community has no observed contribution and a
        # non-positive null contribution. An empty label weakly dominates it.
        for mode in ("amount", "wallet_share"):
            a = normalized(TOY, mode)
            for gamma in (0.5, 1.0, 2.0):
                for labels in partitions(6):
                    w, e = labels[:3], labels[3:]
                    for node_type in ("wallet", "event"):
                        for node in range(3):
                            if node_type == "wallet":
                                candidates = set(e[a[node, :] > 0].tolist()) | {int(w[node]), 6}
                            else:
                                candidates = set(w[a[:, node] > 0].tolist()) | {int(e[node]), 6}
                            all_gains = [dense_node_gain(a, w, e, node_type, node, dest, gamma)
                                         for dest in range(7)]
                            restricted = [all_gains[dest] for dest in candidates]
                            self.assertAlmostEqual(max(all_gains), max(restricted), places=12)

    def test_nonadjacent_community_merge_cannot_improve(self):
        a = normalized(TOY, "amount")
        for gamma in (0.5, 1.0, 2.0):
            for labels in partitions(6):
                w, e = labels[:3], labels[3:]
                groups = np.unique(labels)
                for i, source in enumerate(groups):
                    for target in groups[i + 1:]:
                        connecting = ((w[:, None] == source) & (e[None, :] == target)) | (
                            (w[:, None] == target) & (e[None, :] == source))
                        if not np.any(a[connecting] > 0):
                            self.assertLessEqual(dense_merge_gain(a, w, e, source, target, gamma), 1e-12)


class SparseImplementationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sparse_optimizer
        cls.impl = sparse_optimizer

    def quality(self, a, labels, mode="amount", gamma=1.0):
        nw = a.shape[0]
        wi, we, ww, *_ = sparse_arrays(a)
        return self.impl.modularity(wi, we, ww, labels[:nw], labels[nw:],
                                    weight_mode=mode, gamma=gamma)

    def optimize(self, a, mode="amount", seed=19, **kwargs):
        return self.impl.optimize_bipartite(*sparse_arrays(a), weight_mode=mode,
                                           seed=seed, **kwargs)

    def assert_solution(self, a, result, mode="amount", gamma=1.0):
        w = np.asarray(result["wallet_labels"])
        e = np.asarray(result["event_labels"])
        self.assertEqual(w.shape, (a.shape[0],))
        self.assertEqual(e.shape, (a.shape[1],))
        self.assertTrue(np.issubdtype(w.dtype, np.integer))
        self.assertTrue(np.issubdtype(e.dtype, np.integer))
        self.assertTrue(np.all(w >= 0))
        self.assertTrue(np.all(e >= 0))
        expected = dense_quality(normalized(a, mode), w, e, gamma)
        self.assertAlmostEqual(result["Q"], expected, places=11)
        self.assertGreaterEqual(expected, -1e-11)
        previous = None
        for item in result["history"]:
            if "Q" in item:
                if previous is not None:
                    self.assertGreaterEqual(item["Q"], previous - 1e-10)
                previous = item["Q"]
            if item["stage"] in ("sweep", "merge"):
                self.assertAlmostEqual(item["delta_Q"], item["predicted_delta_Q"], places=10)
            if item["stage"] == "move_batch":
                self.assertGreaterEqual(item["predicted_delta_Q"], -1e-12)
        if result.get("connected_communities_guaranteed", False):
            labels = np.r_[w, e]
            expected_components = dense_induced_components(a, w, e)
            np.testing.assert_array_equal(labels[:, None] == labels[None, :],
                                          expected_components[:, None] == expected_components[None, :])

    def test_objective_all_six_node_partitions(self):
        for mode in ("amount", "wallet_share"):
            a = normalized(TOY, mode)
            for gamma in (0.5, 1.0, 2.0):
                for labels in partitions(6):
                    actual = self.quality(TOY, labels, mode, gamma)
                    expected = dense_quality(a, labels[:3], labels[3:], gamma)
                    self.assertAlmostEqual(actual, expected, places=12)

    def test_move_gain_all_partitions_and_destinations(self):
        for mode in ("amount", "wallet_share"):
            a = normalized(TOY, mode)
            total = float(a.sum())
            sw, se = a.sum(axis=1), a.sum(axis=0)
            for gamma in (0.5, 1.0, 2.0):
                for labels in partitions(6):
                    w, e = labels[:3], labels[3:]
                    for kind in ("wallet", "event"):
                        for node in range(3):
                            edge = a[node, :] if kind == "wallet" else a[:, node]
                            own = w if kind == "wallet" else e
                            other = e if kind == "wallet" else w
                            own_strength = sw if kind == "wallet" else se
                            other_strength = se if kind == "wallet" else sw
                            source = own[node]
                            for target in range(7):
                                actual = self.impl.move_gain(
                                    own_strength[node], float(edge[other == source].sum()),
                                    float(edge[other == target].sum()),
                                    float(other_strength[other == source].sum()),
                                    float(other_strength[other == target].sum()), total, gamma)
                                expected = dense_node_gain(a, w, e, kind, node, target, gamma)
                                self.assertAlmostEqual(actual, expected, places=12)

    def test_merge_gain_all_partitions(self):
        for mode in ("amount", "wallet_share"):
            a = normalized(TOY, mode)
            total = float(a.sum())
            sw, se = a.sum(axis=1), a.sum(axis=0)
            for gamma in (0.5, 1.0, 2.0):
                for labels in partitions(6):
                    w, e = labels[:3], labels[3:]
                    groups = np.unique(labels)
                    for i, source in enumerate(groups):
                        for target in groups[i + 1:]:
                            connecting = ((w[:, None] == source) & (e[None, :] == target)) | (
                                (w[:, None] == target) & (e[None, :] == source))
                            actual = self.impl.merge_gain(float(a[connecting].sum()),
                                float(sw[w == source].sum()), float(se[e == source].sum()),
                                float(sw[w == target].sum()), float(se[e == target].sum()), total, gamma)
                            expected = dense_merge_gain(a, w, e, source, target, gamma)
                            self.assertAlmostEqual(actual, expected, places=12)

    def test_actual_single_node_kernel_selects_unrestricted_best_gain(self):
        wp, wi, ww, ep, ei, ew = sparse_arrays(TOY)
        raw = TOY.sum(axis=1)
        for mode in ("amount", "wallet_share"):
            share = mode == "wallet_share"
            a = normalized(TOY, mode)
            total = float(a.sum())
            sw, se = a.sum(axis=1) / total, a.sum(axis=0) / total
            for gamma in (0.5, 1.0, 2.0):
                for labels in partitions(6):
                    initial_w, initial_e = labels[:3], labels[3:]
                    for kind in ("wallet", "event"):
                        for node in range(3):
                            w, e = initial_w.copy(), initial_e.copy()
                            S, T, count, free, nf = self.impl._group_state(w, e, sw, se, 6)
                            if kind == "wallet":
                                args = wp, wi, ww, w, e, sw, S, T
                            else:
                                args = ep, ei, ew, e, w, se, T, S
                            _, moves, gain = self.impl._move_batch(
                                *args, count, free, nf, np.array([node], dtype=np.int32), 0, 1,
                                raw, kind == "wallet", share, 1 / total, gamma, 0.0,
                                np.zeros((2, 6)), np.zeros(6, dtype=np.bool_), np.empty(6, dtype=np.int32))
                            brute = max(dense_node_gain(a, initial_w, initial_e, kind, node, dest, gamma)
                                        for dest in range(7))
                            actual = dense_quality(a, w, e, gamma) - dense_quality(a, initial_w, initial_e, gamma)
                            self.assertAlmostEqual(actual, brute, places=12)
                            self.assertAlmostEqual(gain, actual, places=12)
                            self.assertIn(moves, (0, 1))
                            np.testing.assert_allclose(S.sum(axis=0), np.bincount(w, weights=sw, minlength=6), atol=1e-13)
                            np.testing.assert_allclose(T.sum(axis=0), np.bincount(e, weights=se, minlength=6), atol=1e-13)
                            np.testing.assert_array_equal(count[0], np.bincount(np.r_[w, e], minlength=6))
                            np.testing.assert_array_equal(count[1], np.bincount(w, minlength=6))
                            np.testing.assert_array_equal(count[2], np.bincount(e, minlength=6))

    def test_actual_merge_candidates_and_disjoint_mergers(self):
        wp, wi, ww, ep, ei, ew = sparse_arrays(TOY)
        raw = TOY.sum(axis=1)
        for mode in ("amount", "wallet_share"):
            a = normalized(TOY, mode)
            total = float(a.sum())
            sw, se = a.sum(axis=1) / total, a.sum(axis=0) / total
            for gamma in (0.5, 1.0, 2.0):
                for labels in partitions(6):
                    w, e = labels[:3].copy(), labels[3:].copy()
                    S, T, _, _, _ = self.impl._group_state(w, e, sw, se, 6)
                    wh, eh, wn, en = self.impl._member_lists(w, e, 6)
                    best, gains = np.full(6, -1, np.int32), np.zeros(6)
                    self.impl._best_mergers(wp, wi, ww, ep, ei, ew, raw, w, e, S, T,
                        wh, eh, wn, en, 0, 6, mode == "wallet_share", 1 / total, gamma, 1e-12,
                        np.zeros((2, 6)), np.zeros(6, dtype=np.bool_), np.empty(6, dtype=np.int32), best, gains)
                    for source in np.unique(labels):
                        brute = max([0.0] + [dense_merge_gain(a, w, e, source, target, gamma)
                                             for target in np.unique(labels) if target != source])
                        self.assertAlmostEqual(gains[source], brute, places=12)
                        if best[source] >= 0:
                            self.assertAlmostEqual(gains[source],
                                dense_merge_gain(a, w, e, source, best[source], gamma), places=12)
                    before = dense_quality(a, w, e, gamma)
                    merges, predicted = self.impl._apply_mergers(w, e, best, gains)
                    after = dense_quality(a, w, e, gamma)
                    self.assertGreaterEqual(after, before - 1e-12)
                    self.assertAlmostEqual(after - before, predicted, places=12)
                    self.assertGreaterEqual(merges, 0)

    def test_compensated_sums_retain_small_terms_after_large_subtraction(self):
        patterns = ([1.0, 1e-20, -1.0], [1e20, 1.0, -1e20],
                    [1.0, -1.0, 1e-200], [1e-100, 1.0, -1.0],
                    [1.0] + [1e-20] * 1000 + [-1.0])
        for pattern in patterns:
            values = np.zeros((2, 1))
            for term in pattern:
                self.impl._add_compensated(values, 0, term)
            actual = float(values[0, 0] + values[1, 0])
            self.assertTrue(math.isclose(actual, math.fsum(pattern), rel_tol=5e-14, abs_tol=0.0))

    def test_large_wallet_move_preserves_tiny_remaining_group_strength(self):
        a = np.diag([1e20, 1.0])
        wp, wi, ww, _, _, _ = sparse_arrays(a)
        raw = a.sum(axis=1)
        total = float(math.fsum(raw))
        sw, se = raw / total, a.sum(axis=0) / total
        w, e = np.array([0, 0], np.int32), np.array([1, 2], np.int32)
        S, T, count, free, nf = self.impl._group_state(w, e, sw, se, 4)
        tally, seen, touched = np.zeros((2, 4)), np.zeros(4, dtype=np.bool_), np.empty(4, dtype=np.int32)
        order = np.array([0, 1], np.int32)
        nf, moves, _ = self.impl._move_batch(wp, wi, ww, w, e, sw, S, T, count, free, nf,
            order, 0, 1, raw, True, False, 1 / total, 0.5, 1e-12, tally, seen, touched)
        self.assertEqual(moves, 1)
        self.assertEqual(w[0], e[0])
        self.assertEqual(count[1, 0], 1)
        self.assertTrue(math.isclose(float(S[0, 0] + S[1, 0]), 1e-20, rel_tol=1e-14))
        nf, moves, _ = self.impl._move_batch(wp, wi, ww, w, e, sw, S, T, count, free, nf,
            order, 1, 2, raw, True, False, 1 / total, 0.5, 1e-12, tally, seen, touched)
        self.assertEqual(moves, 1)
        self.assertEqual(w[1], e[1])
        self.assertEqual(count[0, 0], 0)
        self.assertEqual(count[1, 0], 0)
        np.testing.assert_array_equal(S[:, 0], [0.0, 0.0])
        np.testing.assert_array_equal(T[:, 0], [0.0, 0.0])

    def test_complete_uniform_bipartite_has_no_positive_modularity(self):
        a = np.ones((3, 2))
        for labels in partitions(5):
            self.assertAlmostEqual(self.quality(a, labels), 0.0, places=12)
        result = self.optimize(a)
        self.assert_solution(a, result)
        self.assertAlmostEqual(result["Q"], 0.0, places=12)

    def test_single_link_and_declared_isolates_retained(self):
        a = np.zeros((4, 3))
        a[1, 2] = 7.5
        for mode in ("amount", "wallet_share"):
            result = self.optimize(a, mode)
            self.assert_solution(a, result, mode)
            self.assertAlmostEqual(result["Q"], 0.0, places=12)

    def test_disconnected_equal_pairs_find_separate_blocks(self):
        for count in (2, 3, 7):
            a = np.eye(count)
            result = self.optimize(a)
            self.assert_solution(a, result)
            np.testing.assert_array_equal(result["wallet_labels"], result["event_labels"])
            self.assertEqual(len(np.unique(result["wallet_labels"])), count)
            self.assertAlmostEqual(result["Q"], 1 - 1 / count, places=11)

    def test_seed_reproducibility(self):
        for mode in ("amount", "wallet_share"):
            first, second = self.optimize(TOY, mode, 678), self.optimize(TOY, mode, 678)
            self.assert_solution(TOY, first, mode)
            np.testing.assert_array_equal(first["wallet_labels"], second["wallet_labels"])
            np.testing.assert_array_equal(first["event_labels"], second["event_labels"])
            self.assertEqual(first["Q"], second["Q"])

    def test_random_small_graphs_scores_and_node_preservation(self):
        rng = np.random.default_rng(539104)
        for fixture in range(18):
            nw, ne = int(rng.integers(1, 8)), int(rng.integers(1, 8))
            a = rng.lognormal(0, 2, (nw, ne)) * (rng.random((nw, ne)) < 0.4)
            a[0, 0] = 0.5
            for mode in ("amount", "wallet_share"):
                for gamma in (0.5, 1.0, 2.0):
                    with self.subTest(fixture=fixture, mode=mode, gamma=gamma):
                        result = self.optimize(a, mode, seed=fixture, gamma=gamma)
                        self.assert_solution(a, result, mode, gamma)
                        if result["converged"]:
                            aa = normalized(a, mode)
                            w, e = result["wallet_labels"], result["event_labels"]
                            for kind, count in (("wallet", nw), ("event", ne)):
                                for node in range(count):
                                    for dest in range(nw + ne + 1):
                                        self.assertLessEqual(dense_node_gain(aa, w, e, kind, node, dest, gamma), 1e-10)
                            groups = np.unique(np.r_[w, e])
                            for index, source in enumerate(groups):
                                for target in groups[index + 1:]:
                                    self.assertLessEqual(dense_merge_gain(aa, w, e, source, target, gamma), 1e-10)

    def test_resource_caps_are_not_reported_as_convergence(self):
        for kwargs in ({"max_sweeps": 0}, {"max_sweeps": 1}, {"max_seconds": 0.0},
                       {"max_merge_rounds": 0}):
            with self.subTest(kwargs=kwargs):
                result = self.optimize(TOY, **kwargs)
                self.assertFalse(result["converged"])
                self.assertTrue(result["stop_reason"].startswith("max_"))
                self.assert_solution(TOY, result)

    def test_integrated_connectivity_refinement_splits_a_stationary_disconnected_group(self):
        a = np.eye(3)
        for mode in ("amount", "wallet_share"):
            result = self.optimize(a, mode, initial_wallet_labels=np.zeros(3, np.int32),
                initial_event_labels=np.zeros(3, np.int32), connectivity_refinement=True)
            self.assert_solution(a, result, mode)
            self.assertTrue(result["converged"])
            self.assertTrue(result["connected_communities_guaranteed"])
            splits = [item for item in result["history"] if item["stage"] == "connectivity_split"]
            self.assertGreaterEqual(len(splits), 1)
            self.assertEqual(splits[0]["extra_components"], 2)
            self.assertAlmostEqual(splits[0]["delta_Q"], 2 / 3, places=12)
            self.assertEqual(result["n_communities"], 3)

    def test_toy_result_bounded_by_exhaustive_global_optimum(self):
        # The optimizer is not required to reach this bound; it must not report
        # a score exceeding any mathematically possible partition of the graph.
        for mode in ("amount", "wallet_share"):
            a = normalized(TOY, mode)
            for gamma in (0.5, 1.0, 2.0):
                optimum = max(dense_quality(a, p[:3], p[3:], gamma) for p in partitions(6))
                for seed in (9, 49, 399):
                    result = self.optimize(TOY, mode, seed, gamma=gamma)
                    self.assert_solution(TOY, result, mode, gamma)
                    self.assertLessEqual(result["Q"], optimum + 1e-11)

    def test_optimizer_does_not_modify_sparse_inputs(self):
        arrays = sparse_arrays(TOY)
        saved = tuple(a.copy() for a in arrays)
        self.impl.optimize_bipartite(*arrays, seed=8, weight_mode="wallet_share")
        for original, previous in zip(arrays, saved):
            np.testing.assert_array_equal(original, previous)

    def test_wallet_normalization_invariant_to_wallet_amount_units(self):
        changed = TOY * np.array([0.5, 1024.0, 8.0])[:, None]
        for labels in partitions(6):
            self.assertAlmostEqual(self.quality(TOY, labels, "wallet_share"),
                                   self.quality(changed, labels, "wallet_share"), places=12)
        first, second = self.optimize(TOY, "wallet_share"), self.optimize(changed, "wallet_share")
        self.assertAlmostEqual(first["Q"], second["Q"], places=12)

    def test_global_amount_scale_invariance(self):
        changed = TOY * 1024.0
        for labels in partitions(6):
            self.assertAlmostEqual(self.quality(TOY, labels), self.quality(changed, labels), places=12)
        first, second = self.optimize(TOY), self.optimize(changed)
        self.assertAlmostEqual(first["Q"], second["Q"], places=12)

    def test_broad_normal_float_scales_have_finite_consistent_scores(self):
        for factor in (1e-100, 1e100):
            a = TOY * factor
            for mode in ("amount", "wallet_share"):
                for labels in (np.arange(6), np.zeros(6, dtype=np.int32), np.array([0, 1, 2, 0, 1, 2])):
                    self.assertAlmostEqual(self.quality(a, labels, mode), self.quality(TOY, labels, mode), places=11)
                result = self.optimize(a, mode, seed=77)
                self.assertTrue(math.isfinite(result["Q"]))
                # Dense null products can overflow at much larger scales; the
                # present normal-scale fixtures stay within reference range.
                self.assert_solution(a, result, mode)

    def test_unsupported_reciprocal_scales_rejected_before_nonfinite_scores(self):
        cases = ((np.eye(2) * 1e-320, "amount"),
                 (np.diag([1e-320, 1.0]), "wallet_share"))
        for a, mode in cases:
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "Unsupported numeric scale"):
                    self.optimize(a, mode)
                with self.assertRaisesRegex(ValueError, "Unsupported numeric scale"):
                    self.quality(a, np.array([0, 1, 0, 1]), mode)

    def test_domain_guard_preserves_previous_supported_input_trajectory(self):
        path = Path(__file__).with_name("versions") / "sparse_optimizer_d6f5a68e.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "d6f5a68e21e835e5d57ed18ea15c6e94c910fbf8926f6a152ff97bd132b55a52")
        name = "independent_previous_sparse_optimizer"
        spec = importlib.util.spec_from_file_location(name, path)
        previous = importlib.util.module_from_spec(spec)
        sys.modules[name] = previous
        spec.loader.exec_module(previous)
        for scale in (1e-100, 1.0, 1e100):
            a = TOY * scale
            for mode in ("amount", "wallet_share"):
                old = previous.optimize_bipartite(*sparse_arrays(a), weight_mode=mode, seed=993)
                new = self.optimize(a, mode, seed=993)
                np.testing.assert_array_equal(old["wallet_labels"], new["wallet_labels"])
                np.testing.assert_array_equal(old["event_labels"], new["event_labels"])
                self.assertEqual(old["Q"], new["Q"])
                self.assertEqual(old["n_sweeps"], new["n_sweeps"])
                self.assertEqual(old["n_merge_rounds"], new["n_merge_rounds"])
                self.assertEqual(old["stop_reason"], new["stop_reason"])

    def test_wallet_factor_cache_is_bitwise_scalar_equivalent(self):
        raw = np.array([0.0, 2.0, 3.0, 1e-100, 1e100])
        for inverse in (0.25, 0.00037, 1.0 / 3106514):
            cached = self.impl._wallet_factor_cache(raw, inverse)
            reference = np.array([inverse / float(value) if value > 0 else inverse for value in raw])
            np.testing.assert_array_equal(cached.view(np.uint64), reference.view(np.uint64))

    def test_cached_and_legacy_direct_kernels_are_bitwise_equivalent(self):
        wp, wi, ww, ep, ei, ew = sparse_arrays(TOY)
        raw = TOY.sum(axis=1)
        a = normalized(TOY, "wallet_share")
        total = float(a.sum())
        sw, se = a.sum(axis=1) / total, a.sum(axis=0) / total
        factors = self.impl._wallet_factor_cache(raw, 1 / total)
        for labels in partitions(6):
            for kind in ("wallet", "event"):
                runs = []
                for cached in (None, factors):
                    w, e = labels[:3].copy(), labels[3:].copy()
                    S, T, count, free, nf = self.impl._group_state(w, e, sw, se, 6)
                    args = (wp, wi, ww, w, e, sw, S, T) if kind == "wallet" else (ep, ei, ew, e, w, se, T, S)
                    result = self.impl._move_batch(*args, count, free, nf, np.array([2, 0, 1], np.int32),
                        0, 3, raw, kind == "wallet", True, 1 / total, 1.0, 1e-12,
                        np.zeros((2, 6)), np.zeros(6, dtype=np.bool_), np.empty(6, np.int32), cached)
                    runs.append((result, w, e, S, T, count, free[:result[0]].copy()))
                self.assertEqual(runs[0][0], runs[1][0])
                for first, second in zip(runs[0][1:], runs[1][1:]):
                    if first.dtype == np.float64:
                        np.testing.assert_array_equal(first.view(np.uint64), second.view(np.uint64))
                    else:
                        np.testing.assert_array_equal(first, second)
            w, e = labels[:3].copy(), labels[3:].copy()
            S, T, _, _, _ = self.impl._group_state(w, e, sw, se, 6)
            wh, eh, wn, en = self.impl._member_lists(w, e, 6)
            candidates = []
            for cached in (None, factors):
                best, gains = np.full(6, -1, np.int32), np.zeros(6)
                self.impl._best_mergers(wp, wi, ww, ep, ei, ew, raw, w, e, S, T, wh, eh, wn, en,
                    0, 6, True, 1 / total, 1.0, 1e-12, np.zeros((2, 6)), np.zeros(6, dtype=np.bool_),
                    np.empty(6, np.int32), best, gains, cached)
                candidates.append((best, gains))
            np.testing.assert_array_equal(candidates[0][0], candidates[1][0])
            np.testing.assert_array_equal(candidates[0][1].view(np.uint64), candidates[1][1].view(np.uint64))

    def test_cached_optimizer_preserves_archived_v111_trajectory(self):
        path = Path(__file__).with_name("versions") / "sparse_optimizer_f5a7c8ca.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "f5a7c8ca6f1fae8d2bf3849e3d98cdb13a07c99209e7b12c56abfbe7bbbd9c75")
        name = "independent_sparse_optimizer_v111"
        spec = importlib.util.spec_from_file_location(name, path)
        previous = importlib.util.module_from_spec(spec)
        sys.modules[name] = previous
        spec.loader.exec_module(previous)
        for scale in (1e-100, 1.0, 1e100):
            a = TOY * scale
            for mode in ("amount", "wallet_share"):
                for seed in (93, 707):
                    old = previous.optimize_bipartite(*sparse_arrays(a), weight_mode=mode, seed=seed)
                    new = self.optimize(a, mode, seed=seed)
                    np.testing.assert_array_equal(old["wallet_labels"], new["wallet_labels"])
                    np.testing.assert_array_equal(old["event_labels"], new["event_labels"])
                    self.assertEqual(old["Q"], new["Q"])
                    self.assertEqual(old["stop_reason"], new["stop_reason"])
                    self.assertEqual(old["n_sweeps"], new["n_sweeps"])
                    self.assertEqual(old["n_merge_rounds"], new["n_merge_rounds"])
                    strip = lambda history: [{key: value for key, value in item.items() if key != "elapsed_seconds"}
                                             for item in history]
                    self.assertEqual(strip(old["history"]), strip(new["history"]))
                    self.assertEqual(new["wallet_factor_cache_bytes"], 8 * len(a) if mode == "wallet_share" else 0)

    def test_zero_mass_is_explicitly_noninferential(self):
        for shape in ((3, 4), (0, 0), (0, 3), (4, 0)):
            a = np.zeros(shape)
            result = self.optimize(a)
            self.assertIsNone(result["Q"])
            self.assertEqual(len(result["wallet_labels"]), shape[0])
            self.assertEqual(len(result["event_labels"]), shape[1])
            self.assertFalse(result["converged"])
            with self.assertRaises(ValueError):
                self.quality(a, np.arange(sum(shape)))

    def test_read_only_sparse_arrays_supported(self):
        arrays = sparse_arrays(TOY)
        for array in arrays:
            array.setflags(write=False)
        for mode in ("amount", "wallet_share"):
            result = self.impl.optimize_bipartite(*arrays, seed=901, weight_mode=mode)
            self.assert_solution(TOY, result, mode)

    def test_malformed_sparse_arrays_rejected(self):
        changes = ((0, np.array([0, 2, 1, 5], dtype=np.int64)),
                   (1, np.array([0, 1, 1, 2, 3], dtype=np.int32)),
                   (1, np.array([0, 1, 1, 2, 2], dtype=np.float64)),
                   (2, np.ones(5, dtype=np.float32)))
        for index, replacement in changes:
            arrays = list(sparse_arrays(TOY))
            arrays[index] = replacement
            with self.subTest(index=index, dtype=str(replacement.dtype)):
                with self.assertRaises(ValueError):
                    self.impl.optimize_bipartite(*arrays, seed=3)

    def test_negative_nan_infinite_amounts_rejected(self):
        for value in (-1.0, np.nan, np.inf):
            arrays = list(sparse_arrays(TOY))
            arrays[2] = arrays[2].copy()
            arrays[2][0] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.impl.optimize_bipartite(*arrays, seed=3)


class ConnectivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import connectivity
        cls.impl = connectivity

    def assert_split(self, a, labels):
        wp, wi, *_ = sparse_arrays(a)
        nw = a.shape[0]
        w, e = labels[:nw], labels[nw:]
        original = tuple(x.copy() for x in (wp, wi, w, e))
        result = self.impl.split_disconnected(wp, wi, w, e)
        new = np.r_[result["wallet_labels"], result["event_labels"]]
        expected = dense_induced_components(a, w, e)
        np.testing.assert_array_equal(new[:, None] == new[None, :], expected[:, None] == expected[None, :])
        # No output component may join members of two input communities.
        self.assertTrue(np.all((new[:, None] != new[None, :]) | (labels[:, None] == labels[None, :])))
        self.assertEqual(len(new), sum(a.shape))
        self.assertEqual(result["n_components"], len(np.unique(expected)))
        self.assertEqual(result["extra_components"], len(np.unique(expected)) - len(np.unique(labels)))
        self.assertEqual(result["nodes_excluded"], 0)
        self.assertEqual(result["links_excluded"], 0)
        for before, after in zip(original, (wp, wi, w, e)):
            np.testing.assert_array_equal(before, after)
        new_w, new_e = new[:nw], new[nw:]
        old_internal = w[:, None] == e[None, :]
        new_internal = new_w[:, None] == new_e[None, :]
        np.testing.assert_array_equal(old_internal[a > 0], new_internal[a > 0])
        for mode in ("amount", "wallet_share"):
            aa = normalized(a, mode)
            for gamma in (0.0, 0.5, 1.0, 2.0):
                self.assertGreaterEqual(dense_quality(aa, new_w, new_e, gamma),
                                        dense_quality(aa, w, e, gamma) - 1e-12)
        second = self.impl.split_disconnected(wp, wi, new_w, new_e)
        self.assertEqual(second["extra_components"], 0)

    def test_all_six_node_partitions_against_independent_breadth_first_search(self):
        for labels in partitions(6):
            self.assert_split(TOY, labels)

    def test_disconnected_isolates_single_link_and_empty_graphs(self):
        for a in (np.diag([1.0, 3.0, 5.0]), np.zeros((3, 4)), np.array([[1.0, 0], [0, 0]]),
                  np.zeros((0, 0)), np.zeros((0, 3)), np.zeros((4, 0))):
            self.assert_split(a, np.zeros(sum(a.shape), dtype=np.int32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    implementation = Path(__file__).with_name("sparse_optimizer.py")
    connectivity_path = Path(__file__).with_name("connectivity.py")
    before_hash = hashlib.sha256(implementation.read_bytes()).hexdigest() if implementation.exists() else None
    before_connectivity_hash = hashlib.sha256(connectivity_path.read_bytes()).hexdigest() if connectivity_path.exists() else None
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after_hash = hashlib.sha256(implementation.read_bytes()).hexdigest() if implementation.exists() else None
    after_connectivity_hash = hashlib.sha256(connectivity_path.read_bytes()).hexdigest() if connectivity_path.exists() else None
    code_stable = before_hash == after_hash and before_connectivity_hash == after_connectivity_hash
    report = {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "passed": result.wasSuccessful() and code_stable,
        "seconds": time.perf_counter() - started,
        "numpy_version": np.__version__,
        "exhaustive_six_node_quality_cases": 203 * 2 * 3,
        "exhaustive_six_node_move_gain_cases": 203 * 2 * 3 * 6 * 7,
        "exhaustive_single_node_kernel_cases": 203 * 2 * 3 * 6,
        "exhaustive_merge_kernel_partition_cases": 203 * 2 * 3,
        "random_small_graph_mode_resolution_cases": 18 * 2 * 3,
        "real_market_data_used": False,
        "downloads": False,
        "scope": "Independent objective/delta and small-graph behavior tests; not full-network convergence, significance or a global-optimum proof",
        "implementation_sha256": after_hash,
        "connectivity_sha256": after_connectivity_hash,
        "source_unchanged_during_tests": code_stable,
        "method_version": getattr(getattr(SparseImplementationTests, "impl", None), "METHOD_VERSION", None),
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "failure_details": [detail for _, detail in result.failures],
        "error_details": [detail for _, detail in result.errors],
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() and code_stable else 1)
