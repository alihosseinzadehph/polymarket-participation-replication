"""Independent tiny-graph tests of lossless directed quotient refinement.

No market data are used. Coarsening an existing partition constrains its search
space; exact quotient scores do not establish a global optimum or significance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from test_sparse_method import TOY, dense_quality, normalized, partitions, sparse_arrays


class CoarseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import coarse_refine
        cls.impl = coarse_refine

    def aggregate(self, a, labels, directory, mode="amount", **kwargs):
        wp, wi, ww, *_ = sparse_arrays(a)
        nw = a.shape[0]
        return self.impl.aggregate_coarse(wp, wi, ww, labels[:nw], labels[nw:],
            work_directory=directory, weight_mode=mode, **kwargs)

    def assert_quotient(self, a, labels, quotient, mode):
        self.assertEqual(quotient["status"], "ready")
        w, e = quotient["wallet_groups"], quotient["event_groups"]
        n = quotient["n_vertices"]
        self.assertEqual(len(w), a.shape[0])
        self.assertEqual(len(e), a.shape[1])
        grouped = np.r_[w, e]
        np.testing.assert_array_equal(grouped[:, None] == grouped[None, :],
                                      labels[:, None] == labels[None, :])
        self.assertEqual(n, len(np.unique(labels)))
        reference = np.zeros((n, n))
        aa = normalized(a, mode)
        for u in range(a.shape[0]):
            for v in range(a.shape[1]):
                reference[w[u], e[v]] += aa[u, v]
        edges, weights = quotient["edges"], quotient["weights"]
        actual = np.zeros((n, n))
        np.add.at(actual, (edges[:, 0], edges[:, 1]), weights)
        np.testing.assert_allclose(actual, reference, rtol=1e-13, atol=1e-15)
        self.assertAlmostEqual(float(weights.sum()), float(aa.sum()), places=12)
        self.assertEqual(len(edges), int(np.count_nonzero(reference)))
        self.assertEqual(quotient["represented_original_edges"], int(np.count_nonzero(a)))
        self.assertEqual(quotient["n_self_loops"], int(np.count_nonzero(np.diag(reference))))
        self.assertAlmostEqual(quotient["self_loop_weight"], float(np.trace(reference)), places=12)
        self.assertEqual(quotient["nodes_excluded"], 0)
        self.assertEqual(quotient["links_excluded"], 0)
        return w, e

    def test_exhaustive_base_partitions_and_every_possible_coarsening(self):
        parent = Path(__file__).resolve().parent
        for mode in ("amount", "wallet_share"):
            aa = normalized(TOY, mode)
            for index, labels in enumerate(partitions(6)):
                with tempfile.TemporaryDirectory(prefix="coarse-synthetic-", dir=parent) as work:
                    quotient = self.aggregate(TOY, labels, Path(work), mode)
                    w, e = self.assert_quotient(TOY, labels, quotient, mode)
                    for coarse_labels in partitions(quotient["n_vertices"]):
                        for gamma in (0.5, 1.0, 2.0):
                            score = self.impl.coarse_modularity(quotient["edges"], quotient["weights"],
                                quotient["n_vertices"], coarse_labels, gamma=gamma)
                            expected = dense_quality(aa, coarse_labels[w], coarse_labels[e], gamma)
                            self.assertAlmostEqual(score, expected, places=11)

    def test_single_edge_batches_and_empty_rows_preserve_transpose_information(self):
        a = np.array([[0, 0, 0], [0.4, 0, 1.0], [0, 0, 0], [0, 3.0, 0]])
        labels = np.array([0, 1, 2, 3, 1, 3, 0], dtype=np.int32)
        parent = Path(__file__).resolve().parent
        for mode in ("amount", "wallet_share"):
            for batch in (1, 2, 100):
                with tempfile.TemporaryDirectory(prefix="coarse-synthetic-", dir=parent) as work:
                    quotient = self.aggregate(a, labels, work, mode, batch_edges=batch)
                    self.assert_quotient(a, labels, quotient, mode)

    def test_all_internal_mass_retained_as_a_directed_self_loop(self):
        a = np.ones((2, 2))
        labels = np.zeros(4, dtype=np.int32)
        parent = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="coarse-synthetic-", dir=parent) as work:
            q = self.aggregate(a, labels, work)
            self.assert_quotient(a, labels, q, "amount")
            np.testing.assert_array_equal(q["edges"], [[0, 0]])
            np.testing.assert_array_equal(q["weights"], [4.0])
            for gamma in (0.5, 1.0, 2.0):
                r = self.impl.run_coarse_leiden(q["edges"], q["weights"], 1, seed=76, gamma=gamma)
                self.assertAlmostEqual(r["Q"], 1 - gamma, places=12)
                self.assertAlmostEqual(r["Q_initial"], 1 - gamma, places=12)

    def test_graph_memory_gate_does_not_materialize_or_filter_quotient(self):
        parent = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="coarse-synthetic-", dir=parent) as work:
            labels = np.arange(6)
            q = self.aggregate(TOY, labels, work, graph_memory_budget_bytes=1)
            self.assertEqual(q["status"], "skipped_memory_gate")
            self.assertNotIn("edges", q)
            self.assertNotIn("weights", q)
            self.assertEqual(q["represented_original_edges"], int(np.count_nonzero(TOY)))
            self.assertEqual(q["nodes_excluded"], 0)
            self.assertEqual(q["links_excluded"], 0)

    def test_end_to_end_expansion_is_monotone_complete_and_explicitly_not_original_convergence(self):
        parent = Path(__file__).resolve().parent
        initial = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
        wp, wi, ww, *_ = sparse_arrays(TOY)
        for mode in ("amount", "wallet_share"):
            with tempfile.TemporaryDirectory(prefix="coarse-synthetic-", dir=parent) as work:
                r = self.impl.refine_coarse_leiden(wp, wi, ww, initial[:3], initial[3:],
                    work_directory=work, seed=153, weight_mode=mode, return_quotient=True)
                self.assertEqual(r["status"], "refined")
                self.assertEqual(len(r["wallet_labels"]), 3)
                self.assertEqual(len(r["event_labels"]), 3)
                self.assertFalse(r["original_network_converged"])
                self.assertTrue(r["subsequent_sparse_refinement_required"])
                self.assertEqual(r["nodes_excluded"], 0)
                self.assertEqual(r["links_excluded"], 0)
                aa = normalized(TOY, mode)
                self.assertAlmostEqual(r["Q_before"], dense_quality(aa, initial[:3], initial[3:]), places=11)
                self.assertAlmostEqual(r["Q_after"], dense_quality(aa, r["wallet_labels"], r["event_labels"]), places=11)
                self.assertGreaterEqual(r["Q_after"], r["Q_before"] - 1e-11)
                new = np.r_[r["wallet_labels"], r["event_labels"]]
                self.assertTrue(np.all((initial[:, None] != initial[None, :]) | (new[:, None] == new[None, :])))
                np.testing.assert_array_equal(r["wallet_labels"], r["coarse_result"]["membership"][r["wallet_groups"]])
                np.testing.assert_array_equal(r["event_labels"], r["coarse_result"]["membership"][r["event_groups"]])
                self.assert_quotient(TOY, initial,
                    {**r["aggregation"], "edges": r["quotient_edges"], "weights": r["quotient_weights"],
                     "wallet_groups": r["wallet_groups"], "event_groups": r["event_groups"]}, mode)
                scores = [item["Q"] for item in r["coarse_result"]["history"]]
                self.assertTrue(all(after >= before - 1e-11 for before, after in zip(scores, scores[1:])))

    def test_native_iteration_and_time_caps_are_not_convergence(self):
        a = TOY
        parent = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="coarse-synthetic-", dir=parent) as work:
            q = self.aggregate(a, np.arange(6), work)
            for options in ({"max_iterations": 0}, {"max_seconds": 0}):
                r = self.impl.run_coarse_leiden(q["edges"], q["weights"], q["n_vertices"], seed=991, **options)
                self.assertFalse(r["converged"])
                self.assertTrue(r["stop_reason"].startswith("max_"))
                self.assertEqual(r["Q"], r["Q_initial"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    source = Path(__file__).with_name("coarse_refine.py")
    before_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(CoarseTests))
    after_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    report = {
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "passed": result.wasSuccessful() and before_hash == after_hash,
        "seconds": time.perf_counter() - started,
        "source_unchanged_during_tests": before_hash == after_hash,
        "coarse_implementation_sha256": after_hash,
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "real_market_data_used": False, "downloads": False,
        "base_partitions_per_weight_mode": 203,
        "weight_modes": 2,
        "coarsened_partitions_per_weight_mode": 2471,
        "resolution_values": 3,
        "exhaustive_quotient_objective_comparisons": 2471 * 2 * 3,
        "failure_details": [details for _, details in result.failures],
        "error_details": [details for _, details in result.errors],
        "scope": "Lossless quotient numerical identity and tiny-graph implementation checks; not a global optimum or significance test",
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
