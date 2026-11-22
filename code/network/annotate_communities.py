"""Describe finished full-population communities using existing event metadata.

This is an annotation of fixed memberships, not a community-detection method,
significance test, semantic classifier, or name generator. No edge table is read.
The default final run must exist and report optimizer convergence; partial,
coarse-only, running, and time-limited runs are rejected.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import html
import json
import math
from pathlib import Path
from datetime import datetime, timezone
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "full-network-2026-09-13" / "data"
VERSION = "exact-full-community-annotations-1.0"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def complete_alignment(ids, labels, reference_ids):
    require(ids.dtype.kind in "iu" and labels.dtype.kind in "iu", "IDs and labels must be integers")
    require(len(ids) == len(labels) == len(reference_ids), "Full-population node count mismatch")
    require(np.all(ids >= 0) and np.all(labels >= 0), "Negative membership ID/label")
    if len(ids) > 1 and not np.all(ids[1:] > ids[:-1]):
        order = np.argsort(ids, kind="stable")
        ids, labels = ids[order], labels[order]
    require(len(ids) < 2 or np.all(ids[1:] > ids[:-1]), "Duplicated membership ID")
    require(np.array_equal(ids, reference_ids), "Exact full-node identity mismatch; intersection forbidden")
    require(len(labels) == 0 or int(labels.max()) <= np.iinfo(np.int32).max, "Community ID exceeds int32")
    return np.array(labels, dtype=np.int32, copy=True)


def update_top_five(heap, participation_usd, event_index, event_id, title, category_id, observed_fill_usd):
    # Higher amount first, then smaller original event index for ties.
    item = (float(participation_usd), -int(event_index), event_id, title, category_id, float(observed_fill_usd))
    if len(heap) < 5:
        heapq.heappush(heap, item)
    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(heap, item)


def self_test():
    reference = np.array([0, 2, 7], dtype=np.int32)
    actual = complete_alignment(np.array([7, 0, 2]), np.array([1, 4, 3]), reference)
    require(np.array_equal(actual, [4, 3, 1]), "Exact join reordering fixture failed")
    rejected = 0
    for ids in ([0, 2], [0, 2, 8], [0, 2, 2], [0, 2, 7, 8]):
        try:
            complete_alignment(np.array(ids), np.zeros(len(ids), dtype=np.int32), reference)
        except ValueError:
            rejected += 1
    require(rejected == 4, "Exact-join rejection fixtures failed")
    heap = []
    for i, amount in enumerate([3, 9, 9, 8, 7, 6, 5, 9]):
        update_top_five(heap, amount, i, str(i), "fixture", "a", amount / 2)
    require([-item[1] for item in sorted(heap, reverse=True)] == [1, 2, 7, 3, 4], "Top-five ranking/tie fixture failed")
    group = np.array([0, 0, 1]); category = np.array([0, 1, 1]); money = np.array([1., 3., 6.])
    cell = group * 2 + category
    counts = np.bincount(cell, minlength=4).reshape(2, 2)
    amounts = np.bincount(cell, weights=money, minlength=4).reshape(2, 2)
    require(np.array_equal(counts, [[1, 1], [0, 1]]), "Category count fixture")
    require(np.array_equal(amounts.sum(axis=1), [4., 6.]), "Category amount conservation fixture")
    require(np.array_equal(amounts[0] / amounts[0].sum(), [.25, .75]), "Category share fixture")
    return {"passed": True, "full_join_reordering": True, "invalid_full_joins_rejected": rejected,
            "top_five_ranking_and_ties": True, "category_count_amount_share_conservation": True}


def md_text(value):
    if value is None or value == "":
        return "[missing in source]"
    return html.escape(str(value), quote=False).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def markdown(report):
    p = report["population"]
    rows = report["communities"]
    largest = sorted(rows, key=lambda item: (-item["event_side_participation_usd"], item["community_id"]))[:10]
    shown = sum(c["event_side_participation_usd"] for c in largest)
    lines = ["# Descriptive annotation of the full wallet-event communities", "",
        f"The finalized partition contains {p['n_communities']:,} communities, {p['n_wallets']:,} wallets, "
        f"and {p['n_events']:,} events. All community memberships and event records were matched by exact original IDs. "
        "No nodes or communities were removed for annotation. The number of communities was determined by the optimizer; "
        "it is not a selected number of nodes or a fixed group-count input.", "",
        "These are statistical communities from the recorded network optimizer. Category labels and event titles "
        "are attached afterward from the existing event metadata; they were not used to choose memberships. "
        "No community is automatically named, and the tables do not establish causal relationships, economic identities, "
        "or statistical significance. This annotation concerns one recorded optimization trajectory "
        "(including its documented continuation/refinement), not a multi-seed consensus partition.", "",
        "## Meaning of the quantities", "",
        "Wallet share is the number of member wallets divided by the total number of wallets. Event-side participation "
        "amount sums the observed participation amounts of events assigned to a community, including their links to wallets "
        "outside that community. It is therefore **not internal trading volume**. Maker and taker participation are both "
        "counted: participation amount is twice the corresponding observed fill amount, not an additional independent volume measure.", "",
        "Category event share counts events equally. Category amount share weights each event by its observed participation "
        "amount. A top event is a descriptive example chosen by that same amount, not a community label or evidence that all "
        "other events have the same subject. Category composition reflects the recorded taxonomy, not an independent truth label.", "",
        "## Largest ten communities by event-side participation amount", "",
        f"The {len(largest)} displayed communities contain {shown / p['event_side_participation_usd']:.6%} of the total "
        "event-side amount. This is a presentation ranking only. All communities remain in the calculations and "
        "are listed below; the JSON contains every category count/share and up to five event examples for each community.", "",
        "| Community ID | Wallets | Wallet share | Events | Event-side amount share | Largest category by amount | Category amount share |",
        "|---|---:|---:|---:|---:|---|---:|"]
    for row in largest:
        cat = max(row["categories"], key=lambda c: (c["participation_usd"], c["category_id"]), default=None)
        lines.append(f"| {row['community_id']} | {row['wallet_count']:,} | {row['fraction_all_wallets']:.4%} | "
                     f"{row['event_count']:,} | {row['fraction_all_event_side_amount']:.4%} | "
                     f"{md_text(cat['category'] if cat else None)} | {cat['fraction_community_event_side_amount'] if cat else 0:.4%} |")
    other = [row for row in rows if row not in largest]
    lines += ["", f"Other {len(other)} communities combined: {sum(c['wallet_count'] for c in other):,} wallets, "
        f"{sum(c['event_count'] for c in other):,} events, "
        f"{sum(c['event_side_participation_usd'] for c in other)/p['event_side_participation_usd']:.6%} of event-side amount.", "",
        "## Event examples for the displayed communities", ""]
    for row in largest:
        lines += [f"### Community {row['community_id']}", "",
                  "| Original event ID | Recorded event title | Recorded category | Observed fill amount (USDC) |",
                  "|---|---|---|---:|"]
        for event in row["top_five_events_by_participation_amount"]:
            lines.append(f"| {md_text(event['event_id'])} | {md_text(event['title'])} | "
                         f"{md_text(event['category'])} | {event['observed_fill_usd']:,.2f} |")
        lines += [""]
    lines += ["## All-community inventory", "",
        "Community IDs are identifiers only; numerical order has no substantive interpretation.", "",
        "| Community ID | Wallets | Events | Event-side participation amount (USDC) | Share of all event-side amount |",
        "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['community_id']} | {row['wallet_count']:,} | {row['event_count']:,} | "
                     f"{row['event_side_participation_usd']:,.2f} | {row['fraction_all_event_side_amount']:.6%} |")
    lines += ["", "## Validation and provenance", "",
        "Membership file hashes, the storage manifest, inverse node-ID maps, and event-statistics hash were checked. "
        "Every event-statistics row appears once, and no membership is unmatched. Per-community wallet/event counts and "
        "event-side amounts agree with the independent optimizer summary. Category totals conserve each community's event count "
        "and amount. All inputs were rehashed after annotation.", "",
        f"Run: `{report['run_directory']}`.", "",
        f"Participation table SHA-256: `{report['source_participation_sha256']}`.", "",
        f"Recorded objective: Q = {report['optimizer']['Q']:.12f}, gamma = {report['optimizer']['gamma']}; "
        f"weight mode `{report['optimizer']['weight_mode']}`. "
        "The annotation does not rerun this objective, its optimizer, or any null simulation.", "",
        "`annotate_communities.py --run <finished-full-run> --storage <matching-storage>` reproduces the annotations. "
        "The event-statistics input must cover exactly the same full event population and observation window. "
        "Running or unconverged runs are refused. Detailed input hashes, tests, source categories, and all communities "
        "are retained in the companion JSON.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "full_amount_refined_seed_20260913")
    parser.add_argument("--storage", type=Path, default=ROOT / "storage_all")
    parser.add_argument("--event-statistics", type=Path, default=DATA / "event_statistics.parquet")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "full_community_annotations")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    tests = self_test()
    if args.self_test:
        print(json.dumps(tests)); return
    started = time.perf_counter()
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    inputs = {}
    def bind(path, expected=None):
        path = Path(path).resolve()
        sha = digest(path)
        require(expected is None or sha == expected, f"Provenance hash mismatch: {path}")
        inputs[str(path)] = {"sha256": sha, "bytes": path.stat().st_size}
        return path
    status_path = bind(args.run / "run_status.json")
    status = load_json(status_path)
    require(status.get("status") == "finished_for_declared_budget_not_a_significance_test", "Not a finished original-network optimization run")
    result_path = bind(args.run / "result.json")
    run = load_json(result_path)
    require(run.get("optimizer", {}).get("converged") is True, "Final full-network optimizer has not converged")
    require(run["optimizer"].get("stop_reason") == "no_positive_node_move_or_pair_merge", "Unexpected convergence criterion")
    require(run.get("all_nodes_assigned") is True and run["nodes_excluded"] == run["links_excluded"] == 0, "Incomplete population")
    require(run.get("groups_with_multiple_internal_components") == 0, "Disconnected community needs review")
    run["optimizer"].pop("history", None)
    gc.collect()
    storage_path = bind(args.storage / "storage.json", run["storage_manifest_sha256"])
    storage = load_json(storage_path)
    require(storage["status"] == "validated_complete", "Unvalidated storage")
    for field in ("n_wallets", "n_events", "n_links"):
        require(run[field] == storage[field], "Storage/run population mismatch: " + field)
    event_checkpoint_path = bind(args.event_statistics.with_suffix(".finalizer.json"))
    event_checkpoint = load_json(event_checkpoint_path)
    event_path = bind(args.event_statistics, event_checkpoint["sha256"])
    event_file = pq.ParquetFile(event_path)
    require(event_file.metadata.num_rows == event_checkpoint["rows"] == run["n_events"], "Event metadata must cover exactly this event population")
    expected_ids, labels, counts = {}, {}, {}
    n_groups = run["n_communities"]
    for side in ("wallet", "event"):
        info = storage["arrays"][side + "_original_ids"]
        original = np.load(bind(args.storage / info["file"], info["sha256"]), mmap_mode="r", allow_pickle=False)
        require(original.shape == (run["n_" + side + "s"],) and np.all(original[1:] > original[:-1]), "Invalid inverse-ID map")
        expected_ids[side] = original
        name = side + "_memberships.parquet"
        matches = [entry for entry in run["memberships"] if entry["file"] == name]
        require(len(matches) == 1, "Missing/duplicate membership provenance")
        table = pq.read_table(bind(args.run / name, matches[0]["sha256"]), use_threads=False)
        require(all(col.null_count == 0 for col in table.columns), "Null membership")
        labels[side] = complete_alignment(table["original_dimension_id"].to_numpy(), table["community"].to_numpy(), original)
        require(np.all(labels[side] < n_groups), "Invalid canonical community ID")
        counts[side] = np.bincount(labels[side], minlength=n_groups)
        del table
    require(np.all(counts["wallet"] + counts["event"] > 0), "Empty or missing canonical community")
    summary_path = bind(args.run / "community_summary.parquet")
    summary = pq.read_table(summary_path, use_threads=False)
    summary_order = np.argsort(summary["community"].to_numpy())
    require(np.array_equal(summary["community"].to_numpy()[summary_order], np.arange(n_groups)), "Summary community IDs mismatch")
    for side in ("wallet", "event"):
        require(np.array_equal(counts[side], summary[side + "s"].to_numpy()[summary_order]), "Summary node count mismatch")
    expected_amount = summary["event_raw_participation_usd"].to_numpy()[summary_order]
    require(np.all(np.isfinite(expected_amount)) and np.all(expected_amount >= 0), "Invalid summary amounts")
    # Read only small category columns first to allocate a community-by-category table.
    category_table = event_file.read(columns=["category_id", "category"], use_threads=False)
    require(all(col.null_count == 0 for col in category_table.columns), "Null source taxonomy")
    category_ids = sorted(set(category_table["category_id"].to_pylist()))
    category_names = {}
    for cid, cname in zip(category_table["category_id"].to_pylist(), category_table["category"].to_pylist()):
        require(cid not in category_names or category_names[cid] == cname, "Inconsistent category name")
        category_names[cid] = cname
    del category_table
    category_index = {cid: i for i, cid in enumerate(category_ids)}
    nc = len(category_ids)
    require(0 < nc and n_groups * nc <= 8_000_000, "Category aggregation exceeds bounded memory budget")
    category_counts = np.zeros((n_groups, nc), dtype=np.int64)
    category_amounts = np.zeros((n_groups, nc), dtype=np.float64)
    event_counts = np.zeros(n_groups, dtype=np.int64)
    event_amounts = np.zeros(n_groups, dtype=np.float64)
    fill_amounts = np.zeros(n_groups, dtype=np.float64)
    participation_counts = np.zeros(n_groups, dtype=np.int64)
    seen = np.zeros(run["n_events"], dtype=np.bool_)
    top = [[] for _ in range(n_groups)]
    missing_titles = 0
    columns = ["event_index", "event_id", "title", "category_id", "participation_usd", "observed_fill_usd", "participation_fills", "observed_fills", "active_mapped"]
    for batch in event_file.iter_batches(batch_size=32768, columns=columns, use_threads=False):
        arrays = {name: batch.column(i) for i, name in enumerate(columns)}
        require(all(a.null_count == 0 for name, a in arrays.items() if name != "title"), "Null source event statistic")
        ids = arrays["event_index"].to_numpy()
        pos = np.searchsorted(expected_ids["event"], ids)
        require(np.all(pos < len(seen)), "Extra event ID")
        require(np.array_equal(expected_ids["event"][pos], ids), "Unknown event ID")
        require(len(np.unique(pos)) == len(pos) and not np.any(seen[pos]), "Duplicate source event ID")
        seen[pos] = True
        community = labels["event"][pos]
        amount = arrays["participation_usd"].to_numpy()
        observed = arrays["observed_fill_usd"].to_numpy()
        fills = arrays["participation_fills"].to_numpy()
        require(np.all(np.isfinite(amount)) and np.all(amount > 0), "Invalid source participation amount")
        require(np.all(arrays["active_mapped"].to_numpy(zero_copy_only=False)), "Inactive event in full active network")
        require(np.array_equal(fills, 2 * arrays["observed_fills"].to_numpy()), "Maker+taker count convention mismatch")
        require(np.allclose(amount, 2 * observed, rtol=1e-9, atol=1e-5), "Participation/fill amount convention mismatch")
        raw_categories = arrays["category_id"].to_pylist()
        cindex = np.fromiter((category_index[c] for c in raw_categories), dtype=np.int32, count=len(raw_categories))
        cells = community.astype(np.int64) * nc + cindex
        category_counts += np.bincount(cells, minlength=n_groups * nc).reshape(n_groups, nc)
        category_amounts += np.bincount(cells, weights=amount, minlength=n_groups * nc).reshape(n_groups, nc)
        event_counts += np.bincount(community, minlength=n_groups)
        event_amounts += np.bincount(community, weights=amount, minlength=n_groups)
        fill_amounts += np.bincount(community, weights=observed, minlength=n_groups)
        np.add.at(participation_counts, community, fills)
        titles = arrays["title"].to_pylist(); external_ids = arrays["event_id"].to_pylist()
        missing_titles += sum(t is None or t == "" for t in titles)
        for i, group in enumerate(community):
            update_top_five(top[int(group)], amount[i], ids[i], external_ids[i], titles[i], raw_categories[i], observed[i])
    require(np.all(seen), "Missing event-statistics identity")
    require(np.array_equal(event_counts, counts["event"]), "Event population not conserved")
    require(np.array_equal(category_counts.sum(axis=1), event_counts), "Category event counts not conserved")
    require(np.allclose(category_amounts.sum(axis=1), event_amounts, rtol=1e-12, atol=1e-4), "Category amounts not conserved")
    require(np.allclose(event_amounts, expected_amount, rtol=1e-9, atol=1e-4), "Optimizer event-side amount mismatch")
    total_amount = float(event_amounts.sum())
    require(math.isclose(total_amount, storage["raw_participation_usd"], rel_tol=1e-9, abs_tol=1e-4), "Global amount not conserved")
    require(int(participation_counts.sum()) == storage["participation_fills"], "Global participation count mismatch")
    communities = []
    for group in range(n_groups):
        categories = []
        for j in range(nc):
            if category_counts[group, j] == 0:
                continue
            categories.append({"category_id": category_ids[j], "category": category_names[category_ids[j]],
                "event_count": int(category_counts[group, j]),
                "fraction_community_events": float(category_counts[group, j] / event_counts[group]),
                "participation_usd": float(category_amounts[group, j]),
                "fraction_community_event_side_amount": float(category_amounts[group, j] / event_amounts[group])})
        examples = [{"event_index": -item[1], "event_id": item[2], "title": item[3],
                     "category_id": item[4], "category": category_names[item[4]],
                     "participation_usd": item[0], "observed_fill_usd": item[5],
                     "fraction_community_event_side_amount": item[0] / event_amounts[group]}
                    for item in sorted(top[group], reverse=True)]
        communities.append({"community_id": group, "assigned_semantic_name": None,
            "wallet_count": int(counts["wallet"][group]), "event_count": int(event_counts[group]),
            "fraction_all_wallets": float(counts["wallet"][group] / run["n_wallets"]),
            "fraction_all_events": float(event_counts[group] / run["n_events"]),
            "event_side_participation_usd": float(event_amounts[group]),
            "observed_fill_usd_of_member_events": float(fill_amounts[group]),
            "fraction_all_event_side_amount": float(event_amounts[group] / total_amount),
            "participation_fills_of_member_events": int(participation_counts[group]),
            "categories": sorted(categories, key=lambda c: (-c["participation_usd"], c["category_id"])),
            "top_five_events_by_participation_amount": examples})
    for path, record in inputs.items():
        require(digest(path) == record["sha256"], "Input changed during annotation: " + path)
    report = {"status": "complete_descriptive_annotation_of_converged_full_partition",
        "version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(args.run.resolve()), "source_participation_sha256": storage["source_sha256"],
        "optimizer": {"method": run["optimizer"]["method"], "Q": run["q_barber_checked"],
                      "weight_mode": run["weight_mode"], "gamma": run["gamma"], "seed": run["seed"],
                      "converged": True, "stop_reason": run["optimizer"]["stop_reason"],
                      "community_count_fixed": run["community_count_fixed"],
                      "single_recorded_optimization_trajectory_not_consensus": True},
        "population": {"n_wallets": run["n_wallets"], "n_events": run["n_events"],
                       "n_links": run["n_links"], "n_communities": n_groups,
                       "event_side_participation_usd": total_amount,
                       "observed_fill_usd": float(fill_amounts.sum()),
                       "participation_fills": int(participation_counts.sum())},
        "interpretation": {"community_names_inferred": False, "taxonomy_used_to_infer_memberships": False,
            "category_source": "Existing recorded event taxonomy; exact labels retained",
            "event_side_amount_includes_cross_community_wallet_links": True,
            "participation_amount_counts_both_maker_and_taker": True,
            "top_events_are_descriptive_examples_not_community_labels": True,
            "inferential_p_value": None, "causal_claim": None},
        "validation": {"exact_complete_typed_original_ID_joins": True, "nodes_dropped": 0,
            "all_event_metadata_rows_seen_once": True, "all_community_counts_match_optimizer_summary": True,
            "all_event_side_amounts_match_optimizer_summary": True,
            "max_community_amount_residual_usd": float(np.max(np.abs(event_amounts - expected_amount))),
            "category_counts_and_amounts_conserved": True, "total_count_and_amount_conserved": True,
            "all_inputs_rehashed_after_analysis": True, "missing_event_titles": missing_titles},
        "category_dimension": [{"category_id": c, "category": category_names[c]} for c in category_ids],
        "communities": communities, "inputs": inputs, "code_sha256": digest(__file__),
        "self_tests": tests, "no_raw_or_source_edge_scan": True, "no_download": True,
        "seconds": time.perf_counter() - started, "versions": {"numpy": np.__version__, "pyarrow": pa.__version__}}
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension, text in ((".json", json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)), (".md", markdown(report))):
        target = args.output_prefix.with_suffix(extension)
        temporary = target.with_suffix(extension + ".pending")
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(target)
    print(json.dumps({key: report[key] for key in ("status", "population", "validation", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
