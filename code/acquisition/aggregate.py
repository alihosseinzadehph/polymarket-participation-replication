"""Build the observed positive-fill aggregates from committed archive files, offline."""
import argparse
from datetime import date, datetime, time, timezone, timedelta
import json
import math
from pathlib import Path
import tempfile

from archive import checked_path, inventory, WINDOW

def quote(s):
    return "'" + str(s).replace("\\","/").replace("'","''") + "'"

def parquet(files):
    return "read_parquet([" + ",".join(quote(p) for p in files) + "],union_by_name=true)"

def timestamp(d):
    return int(datetime.combine(d,time(),tzinfo=timezone.utc).timestamp())

def build(files, maps, out, start, end, memory="4GB", threads=2, temp_limit="100GB"):
    import duckdb
    if not files or not maps:
        raise ValueError("No fill files or frozen mapping shards.")
    if not 1 <= threads <= 32:
        raise ValueError("threads must be 1–32")
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        con = duckdb.connect(str(Path(tmp)/"build.duckdb"))
        try:
            con.execute("SET memory_limit = " + quote(memory))
            con.execute("SET threads = " + str(threads))
            con.execute("SET temp_directory = " + quote(Path(tmp)/"spill"))
            con.execute("SET max_temp_directory_size = " + quote(temp_limit))
            con.execute("SET preserve_insertion_order = false")
            # In the historical clean map each valid condition mapped to one event.
            con.execute(f"""CREATE TABLE mapping AS SELECT '0x'||lower(hex(condition_id)) AS c,
                          cast(event_id AS VARCHAR) AS event_id FROM {parquet(maps)}
                          WHERE condition_id IS NOT NULL AND event_id IS NOT NULL""")
            if con.execute("SELECT count(*) != count(DISTINCT c) FROM mapping").fetchone()[0]:
                raise ValueError("Frozen condition_id is not unique; refusing to multiply fills.")
            con.execute(f"""CREATE VIEW fills AS SELECT lower(condition_id) c,
                     lower(maker) maker, lower(taker) taker, usdc_amount amt FROM {parquet(files)}
                     WHERE block_timestamp >= {timestamp(start)} AND block_timestamp < {timestamp(end+timedelta(days=1))}
                       AND usdc_amount>0 AND isfinite(usdc_amount)""")
            def export(name,query):
                dest = out/(name+".parquet")
                con.execute(f"COPY ({query}) TO {quote(dest)} (FORMAT PARQUET, COMPRESSION ZSTD)")
                return dest
            # Preserve unmapped fills in wallet totals, but never invent an event.
            wt=export("wallet_totals","""SELECT wallet,count(*)::BIGINT participant_fills,
                     sum(amt)::DOUBLE participant_volume FROM
                     (SELECT unnest([maker,taker]) wallet,amt FROM fills) t
                     WHERE wallet IS NOT NULL GROUP BY wallet""")
            et=export("event_totals","""SELECT m.event_id, count(*)::BIGINT trade_count,
                    sum(f.amt)::DOUBLE usd_volume FROM fills f JOIN mapping m USING(c)
                    GROUP BY m.event_id""")
            # Historical event_id is a string: lexical order, not numeric order.
            wd=export("wallets",f"""SELECT (row_number() OVER(ORDER BY wallet)-1)::INTEGER wallet_id,
                      wallet FROM read_parquet({quote(wt)})""")
            ed=export("events",f"""SELECT (row_number() OVER(ORDER BY event_id)-1)::INTEGER event_index,
                      event_id FROM read_parquet({quote(et)})""")
            con.execute(f"""CREATE TABLE conditions AS SELECT m.c,d.event_index
                        FROM mapping m JOIN read_parquet({quote(ed)}) d USING(event_id)""")
            con.execute(f"CREATE TABLE wallets AS SELECT * FROM read_parquet({quote(wd)})")
            edges=export("participation", """SELECT wallet_id,event_index,
                      count(*)::BIGINT participation_fills, sum(amt)::DOUBLE usd FROM (
                      SELECT unnest([wm.wallet_id,wt.wallet_id]) wallet_id,c.event_index,f.amt
                      FROM fills f JOIN conditions c USING(c)
                      JOIN wallets wm ON wm.wallet=f.maker JOIN wallets wt ON wt.wallet=f.taker
                      ) x GROUP BY wallet_id,event_index ORDER BY wallet_id,event_index""")
            def stats(p,expr):
                return con.execute(f"SELECT {expr} FROM read_parquet({quote(p)})").fetchone()
            nw, all_sides, all_amt=stats(wt,"count(*),sum(participant_fills),sum(participant_volume)")
            ne, mapped, cash=stats(et,"count(*),sum(trade_count),sum(usd_volume)")
            pairs, sides, side_amt=stats(edges,"count(*),sum(participation_fills),sum(usd)")
            raw=con.execute("SELECT count(*) FROM fills").fetchone()[0]
            if not ne or not nw or sides!=2*mapped or all_sides!=2*raw or not math.isclose(side_amt,2*cash,rel_tol=1e-9,abs_tol=0.00001):
                raise ValueError("No matched population, missing endpoint, or mapped fill-to-edge conservation failed.")
            manifest={"start":str(start),"end_inclusive":str(end),"source_files":len(files),
                "window_is_full_study": (start,end)==WINDOW,
                "positive_finite_fills":raw,"mapped_fills":int(mapped),"unmapped_fills":raw-int(mapped),
                "distinct_edges":pairs,"mapped_event_usdc":cash,"all_wallet_gross_usdc":all_amt,
                "files":{"wallets":{"rows":nw},"events":{"rows":ne}}}
            (out/"dimensions.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
            return manifest
        finally:
            con.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True,help="Raw source directory from archive.py.")
    parser.add_argument("--map-dir",type=Path,default=Path(__file__).resolve().parents[2]/"data/gamma/condition_map")
    parser.add_argument("--out",type=Path,required=True,help="New output directory outside the release.")
    parser.add_argument("--start",type=date.fromisoformat,required=True)
    parser.add_argument("--end",type=date.fromisoformat,required=True,help="Last UTC date, inclusive.")
    parser.add_argument("--memory",default="4GB")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--temporary-disk-limit",default="100GB",help="DuckDB spill limit; excludes DB and outputs.")
    args=parser.parse_args()
    paths=[]
    for rel,size,_,_ in inventory(args.start,args.end):
        p=checked_path(args.root,rel)
        if not p.is_file() or p.stat().st_size!=size:
            raise ValueError(f"Missing or resized pinned shard: {rel}. Run archive.py verify.")
        paths.append(p)
    print(json.dumps(build(paths,sorted(args.map_dir.glob("part-*.parquet")),args.out,
        args.start,args.end,args.memory,args.threads,args.temporary_disk_limit),indent=2))

if __name__=="__main__":
    main()
