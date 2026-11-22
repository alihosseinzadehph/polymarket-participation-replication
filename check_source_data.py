"""Cross-check the frozen observations against the source data behind Figs. 1–2.

Run after installing NumPy and DuckDB. No downloads or database writes.
"""
import csv
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parent / "data"


def rows(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    db = duckdb.connect(":memory:")
    events = str(ROOT / "study/mapped_event_totals.parquet")
    dictionary = str(ROOT / "gamma/event_subjects_v3.parquet")
    condition_map = str(ROOT / "gamma/condition_map/*.parquet")
    assert db.execute("SELECT count(*), count(DISTINCT condition_id) FROM read_parquet(?)",
                      [condition_map]).fetchone() == (2946628, 2946628)
    assert db.execute("SELECT count(*), count(DISTINCT event_id), count(DISTINCT category_id) FROM read_parquet(?)",
                      [dictionary]).fetchone() == (931023, 931023, 21)
    matched = db.execute("SELECT count(*), count(DISTINCT e.event_id) FROM read_parquet(?) e JOIN read_parquet(?) t USING (event_id)",
                         [events, dictionary]).fetchone()
    assert matched == (764378, 764378)
    x = db.execute("SELECT usd_volume FROM read_parquet(?) ORDER BY usd_volume", [events]).fetchnumpy()["usd_volume"]
    assert len(x) == 764378 and x[0] > 0
    n = x.size
    gini = (2 * np.dot(np.arange(1,n+1,dtype=float), x) / x.sum() - (n + 1)) / n
    share = x[-int(np.ceil(.01*n)):].sum() / x.sum()
    published = next(r for r in rows("study/composition_concentration.csv") if r["scope"] == "all_events")
    np.testing.assert_allclose([gini,share], [float(published["gini"]),float(published["top_0.01_share"])],rtol=1e-10)
    wallet_source = str(ROOT / "study/wallet_amounts/*.parquet")
    w = db.execute("SELECT participant_volume_usdc FROM read_parquet(?) ORDER BY 1", [wallet_source]).fetchnumpy()["participant_volume_usdc"]
    nw = len(w)
    assert nw == 3106517 and w[0] > 0
    wgini = (2 * np.dot(np.arange(1,nw+1,dtype=float), w) / w.sum() - (nw + 1)) / nw
    wshare = w[-int(np.ceil(.01*nw)):].sum() / w.sum()
    wallet_published, = rows("Source_Data/Fig1_concentration/wallet_concentration.csv")
    np.testing.assert_allclose([wgini,wshare], [float(wallet_published["gini"]),float(wallet_published["top_0.01_share"])],rtol=1e-10)
    hist = rows("study/bipartite_degree_counts.csv")
    for partition, nodes in [("wallet",3106514),("event",764378)]:
        freq = [r for r in hist if r["node_partition"] == partition]
        assert sum(int(r["nodes_at_degree"]) for r in freq) == nodes
        assert sum(int(r["degree"]) * int(r["nodes_at_degree"]) for r in freq) == 255704359

    ranks = rows("leaderboard/overall_all_top1000.csv")
    points = rows("Source_Data/Fig2_powerlaw/three_tail_plot_points.csv")
    samples = {"top1000_selected": x[-1000:]}
    for order, key, field in [("VOL", "leaderboard_volume", "reported_volume"), ("PNL", "leaderboard_pnl", "reported_pnl")]:
        records = [r for r in ranks if r["order_by"] == order]
        assert sorted(int(r["rank"]) for r in records) == list(range(1,1001))
        samples[key] = np.sort([float(r[field]) for r in records])
    for key, sample in samples.items():
        plotted = np.sort([float(r["x"]) for r in points if r["key"] == key])
        assert len(plotted) == len(sample) == 1000
        np.testing.assert_allclose(sample, plotted, rtol=1e-10, atol=1e-6)
    print(f"Gamma joins passed. Event Gini={gini:.6f}; top 1% share={share:.6f}.")
    print(f"Wallet Gini={wgini:.6f}; top 1% share={wshare:.6f}. Both degree sums equal 255,704,359.")
    print("All three 1,000-value Fig. 2 samples agree with the frozen event or leaderboard data.")


if __name__ == "__main__":
    main()
