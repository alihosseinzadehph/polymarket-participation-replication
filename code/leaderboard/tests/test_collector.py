import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from polymarket_leaderboard.api import JsonResponse
from polymarket_leaderboard.collector import Query, QueryResult, collect_query, make_long_rows, make_trader_rows, make_wide_rows
from polymarket_leaderboard.storage import write_snapshot


def api_row(rank, wallet, pnl=10.0, volume=20.0):
    return {
        "rank": str(rank),
        "proxyWallet": wallet,
        "userName": f"user-{rank}",
        "vol": volume,
        "pnl": pnl,
        "profileImage": "",
        "xUsername": "",
        "verifiedBadge": False,
        "futureField": "preserved",
    }


class FakeClient:
    def __init__(self, pages):
        self.pages = pages
        self.offsets = []

    def leaderboard(self, **params):
        offset = params["offset"]
        self.offsets.append(offset)
        return JsonResponse(
            url=f"https://example.test?offset={offset}",
            fetched_at="2026-01-01T00:00:00.000Z",
            status=200,
            headers={"etag": "test"},
            data=self.pages.get(offset, []),
        )


class CollectorTests(unittest.TestCase):
    def test_paginates_until_short_page_and_preserves_unknown_fields(self):
        client = FakeClient({0: [api_row(i, f"0x{i:040x}") for i in range(1, 51)], 50: [api_row(51, f"0x{51:040x}")]})
        result = collect_query(client, Query("OVERALL", "DAY", "PNL"), page_size=50, max_offset=1000)
        self.assertEqual(client.offsets, [0, 50])
        self.assertEqual(len(result.rows), 51)
        self.assertFalse(result.possibly_truncated)
        rows = make_long_rows([result], "sample")
        self.assertEqual(json.loads(rows[0]["raw_json"])["futureField"], "preserved")

    def test_marks_full_final_page_as_possibly_truncated(self):
        client = FakeClient({0: [api_row(1, "0x" + "1" * 40)]})
        result = collect_query(client, Query("OVERALL", "DAY", "PNL"), page_size=1, max_offset=0)
        self.assertTrue(result.reached_offset_ceiling)
        self.assertTrue(result.possibly_truncated)

    def test_wide_rows_keep_both_ranks(self):
        wallet = "0x" + "a" * 40
        base = {
            "snapshot_id": "sample", "category": "OVERALL", "time_period": "DAY",
            "proxy_wallet": wallet, "user_name": "alice", "profile_image": "", "x_username": "",
            "verified_badge": False, "pnl": 12.0, "volume": 99.0,
        }
        wide = make_wide_rows([{**base, "order_by": "PNL", "rank": 2}, {**base, "order_by": "VOL", "rank": 7}])
        self.assertEqual(len(wide), 1)
        self.assertEqual((wide[0]["pnl_rank"], wide[0]["volume_rank"]), (2, 7))

    def test_writes_queryable_snapshot(self):
        wallet = "0x" + "b" * 40
        source = api_row(1, wallet)
        source.update({
            "category": "OVERALL", "time_period": "DAY", "order_by": "PNL", "page_offset": 0,
            "page_fetched_at": "2026-01-01T00:00:00.000Z", "source_url": "https://example.test",
        })
        result = QueryResult(Query("OVERALL", "DAY", "PNL"), [{"data": [api_row(1, wallet)]}], [source], False, False)
        long_rows = make_long_rows([result], "sample")
        wide_rows = make_wide_rows(long_rows)
        trader_rows = make_trader_rows(long_rows, {}, "sample")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "snapshot"
            write_snapshot(
                destination, results=[result], profiles={}, long_rows=long_rows, wide_rows=wide_rows,
                trader_rows=trader_rows, manifest={"snapshot_id": "sample"},
            )
            self.assertTrue((destination / "leaderboard.csv").exists())
            self.assertTrue((destination / "raw" / "leaderboards" / "overall_day_pnl.json").exists())
            with closing(sqlite3.connect(destination / "leaderboard.sqlite")) as connection:
                self.assertEqual(connection.execute("SELECT count(*) FROM leaderboard").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
