"""Run with python -B -m unittest discover -s code/acquisition (no network access)."""
import gzip
import json
from datetime import date
from pathlib import Path
import tempfile
import unittest
from archive import inventory, WINDOW
from aggregate import build, timestamp
from platform_snapshot import capture

class Offline(unittest.TestCase):
    def test_inventory(self):
        files=list(inventory(*WINDOW))
        self.assertEqual((len(files),sum(r[1] for r in files)),(2375,24060654592))
        self.assertEqual(len(set(r[0] for r in files)),2375)
        self.assertTrue(all(len(r[2])==64 for r in files))

    def test_gamma(self):
        calls=[]
        event={"id":"10","tags":[{"id":"1","label":"Test"}],"markets":[
            {"conditionId":"c1","outcomes":'["Yes", "No"]'}]}
        def fake(p):
            calls.append(p)
            return {"events": [event], "next_cursor":"next"} if "after_cursor" not in p else {"events":[]}
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/"new"
            capture(out,fake)
            meta=json.loads((out/"complete.json").read_text())
            self.assertEqual((meta["events"],meta["markets"],meta["tags"],meta["pages"]),(1,1,1,4))
            self.assertEqual([p["closed"] for p in calls],["false","false","true","true"])
            with gzip.open(out/"condition_event_map.jsonl.gz","rt",encoding="utf-8") as m:
                self.assertEqual(json.loads(m.readline())["outcomes"],["Yes","No"])
            with self.assertRaises(FileExistsError):
                capture(out,fake)

    def test_edges(self):
        import duckdb
        import pyarrow as pa
        import pyarrow.parquet as pq
        d=date(2024,9,1)
        h=lambda n:n.to_bytes(32,"big")
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); out=base/"results"
            pq.write_table(pa.table({"condition_id":pa.array([h(10),h(2)],type=pa.binary(32)),
                "event_id":pa.array([10,2],type=pa.int32())}),base/"map.parquet")
            pq.write_table(pa.table({"condition_id":["0x"+h(x).hex() for x in [10,2,3,10,10]],
                "maker":["A","b","c","b","x"],"taker":["A","a","a","a","x"],
                "usdc_amount":[4.,2.,1.,-8.,8.],
                "block_timestamp":[timestamp(d)]*4+[timestamp(d)+86400]}),base/"fill.parquet")
            stats=build([base/"fill.parquet"],[base/"map.parquet"],out,d,d,"512MB",1,"1GB")
            self.assertEqual((stats["positive_finite_fills"],stats["mapped_fills"],
                stats["mapped_event_usdc"],stats["all_wallet_gross_usdc"]),(3,2,6,14))
            with duckdb.connect() as con:
                edges=con.execute("select * from read_parquet(?)",[str(out/"participation.parquet")]).fetchall()
            self.assertEqual(edges,[(0,0,2,8.),(0,1,1,2.),(1,1,1,2.)])
            self.assertEqual(stats["files"],{"wallets":{"rows":3},"events":{"rows":2}})
