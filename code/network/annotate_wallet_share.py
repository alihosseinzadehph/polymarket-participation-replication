"""Annotate a completed wallet-normalized pipeline with explicit raw-amount scope.

The unchanged annotation engine describes categories using raw observed event
USDC. That quantity is not confused with the normalized optimization weights.
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

import annotate_communities as engine


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "full_wallet_share_pipeline_seed_20260913" / "final")
    parser.add_argument("--storage", type=Path, default=ROOT / "storage_all")
    parser.add_argument("--event-statistics", type=Path, default=engine.DATA / "event_statistics.parquet")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "full_wallet_share_annotations")
    args = parser.parse_args()
    result = engine.load_json(args.run / "result.json")
    engine.require(result["weight_mode"] == "wallet_share", "This wrapper requires wallet-normalized memberships")
    pipeline_path = args.run.parent / "pipeline_result.json"
    pipeline = engine.load_json(pipeline_path)
    pipeline_sha = engine.digest(pipeline_path)
    engine.require(pipeline["original_optimizer_converged"] is True, "Normalized pipeline not finished/converged")
    finals = [item for item in pipeline["stages"] if item["stage"] == "final"]
    engine.require(len(finals) == 1 and finals[0]["result_sha256"] == engine.digest(args.run / "result.json"), "Pipeline final-result identity mismatch")
    previous = sys.argv
    try:
        sys.argv = [str(Path(engine.__file__)), "--run", str(args.run), "--storage", str(args.storage),
                    "--event-statistics", str(args.event_statistics), "--output-prefix", str(args.output_prefix)]
        with contextlib.redirect_stdout(io.StringIO()):
            engine.main()
    finally:
        sys.argv = previous
    report = engine.load_json(args.output_prefix.with_suffix(".json"))
    report["interpretation"].update(
        community_detection_weight="A(u,e)/sum_f A(u,f); total weight one per positive-strength wallet",
        category_annotation_weight="raw observed event participation USDC, not wallet-normalized optimization mass",
        raw_USDC_category_shares_are_not_objective_decomposition=True,
        no_claim_that_raw_amount_dominant_category_explains_normalized_community=True)
    communities = report["communities"]
    def size_subset(rows, definition):
        wallets = sum(row["wallet_count"] for row in rows)
        events = sum(row["event_count"] for row in rows)
        amount = sum(row["event_side_participation_usd"] for row in rows)
        return {"definition": definition, "n_communities": len(rows),
            "wallets": wallets, "events": events,
            "fraction_all_communities": len(rows)/len(communities),
            "fraction_all_wallets": wallets/report["population"]["n_wallets"],
            "fraction_all_events": events/report["population"]["n_events"],
            "raw_event_side_participation_usd": amount,
            "fraction_all_raw_event_side_amount": amount/report["population"]["event_side_participation_usd"],
            "descriptive_summary_only_no_membership_or_data_filter": True}
    sizes = [row["wallet_count"] + row["event_count"] for row in communities]
    report["community_size_summary"] = {
        "median_nodes_per_community": float(engine.np.median(sizes)),
        "minimum_nodes_per_community": min(sizes), "maximum_nodes_per_community": max(sizes),
        "one_wallet_one_event": size_subset([row for row in communities if row["wallet_count"] == row["event_count"] == 1],
                                              "Exactly one member wallet and one member event"),
        "at_most_ten_nodes": size_subset([row for row in communities if row["wallet_count"] + row["event_count"] <= 10],
                                         "At most ten total member nodes; descriptive size bin, not a graph cutoff"),
        "groups_are_not_assumed_to_be_comparably_large_subject_communities": True,
        "two_node_group_does_not_imply_an_isolated_component_of_the_full_graph": True}
    report["pipeline_result_sha256"] = pipeline_sha
    report["inputs"][str(pipeline_path.resolve())] = {"sha256": pipeline_sha, "bytes": pipeline_path.stat().st_size}
    report["wrapper_version"] = "explicit-normalized-partition-raw-annotation-1.0"
    report["wrapper_code_sha256"] = engine.digest(__file__)
    for path, record in report["inputs"].items():
        engine.require(engine.digest(path) == record["sha256"], "Input changed during normalized annotation")
    text = engine.markdown(report).replace(
        "# Descriptive annotation of the full wallet-event communities\n",
        "# Descriptive annotation of wallet-normalized full-network communities\n\n"
        "**Two different weights must be distinguished.** Memberships were inferred using each wallet's "
        "proportional allocation, with total weight one per positive-strength wallet. The category amount "
        "shares and largest-community display below instead summarize raw observed event USDC from the "
        "existing metadata. They are descriptive raw-amount annotations, not a decomposition of the "
        "wallet-normalized objective. Raw-USDC category dominance must not be interpreted as what "
        "caused the normalized optimizer to form a community.\n", 1)
    pair = report["community_size_summary"]["one_wallet_one_event"]
    small = report["community_size_summary"]["at_most_ten_nodes"]
    size_text = ("## Community sizes\n\n"
        f"The median community has {report['community_size_summary']['median_nodes_per_community']:g} nodes. "
        f"There are {pair['n_communities']:,} communities containing exactly one wallet and one event; "
        f"they contain {pair['fraction_all_wallets']:.6%} of all wallets. "
        f"A descriptive bin of at most ten nodes contains {small['n_communities']:,} communities, "
        f"{small['wallets']:,} wallets, and {small['events']:,} events "
        f"({small['fraction_all_raw_event_side_amount']:.6%} of raw event-side amount). "
        "These size summaries do not filter the graph: all groups remain included. The community count must "
        "not be interpreted as the number of comparably large subject areas. A two-node community is a "
        "two-node part of this partition, not necessarily an isolated connected component of the full graph.\n\n")
    text = text.replace("## Meaning of the quantities\n", size_text + "## Meaning of the quantities\n", 1)
    for extension, content in ((".json", json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)), (".md", text)):
        target = args.output_prefix.with_suffix(extension)
        pending = target.with_suffix(extension + ".pending")
        pending.write_text(content + "\n", encoding="utf-8"); pending.replace(target)
    print(json.dumps({key: report[key] for key in ("status", "population", "validation", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
