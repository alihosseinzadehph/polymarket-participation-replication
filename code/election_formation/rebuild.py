"""Offline replay of September election-side formation and October allocation."""
import argparse
import csv
from datetime import date
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import duckdb

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code/acquisition"))
from archive import checked_path, digest, inventory
from aggregate import parquet, quote

SOURCE=ROOT/"data/Source_Data/Fig6_8_election"
FORMS=(("HB","Harris","Bottom 5%"),("TB","Trump","Bottom 5%"),
       ("HT","Harris","Top 5%"),("TT","Trump","Top 5%"))

def read_ref(path, columns):
    groups={g:[] for g,_,_ in FORMS}
    with path.open(encoding="utf-8",newline="") as src:
        rows=csv.DictReader(src)
        if rows.fieldnames[4:]!=columns:
            raise ValueError("Published column order changed")
        for row in rows:
            groups[row["cohort"]].append([float(row[c]) for c in columns])
    return {g:np.asarray(a) for g,a in groups.items()}

def build(raw):
    # Use the original extraction's buffered v1 inventory and set UTC boundaries
    # using timestamps, not the date from a shard name.
    files=[]
    for rel,size,h,_ in inventory(date(2024,7,31),date(2024,11,1)):
        if not rel.startswith("polymarket_v1/"):
            raise ValueError("Unexpected exchange generation")
        f=checked_path(raw,rel)
        if not f.is_file() or f.stat().st_size!=size or digest(f)!=h:
            raise ValueError(f"Missing or unverified study shard: {rel}")
        files.append(f)
    if len(files)!=188:
        raise ValueError("Expected 188 buffered input shards")
    gamma=ROOT/"data/gamma"
    registry=json.loads((gamma/"presidential_election_2024_conditions.json").read_text(encoding="utf-8"))
    if len({r["condition_id"] for r in registry})!=17 or {r["event_slug"] for r in registry}!={"presidential-election-winner-2024"}:
        raise ValueError("Unexpected focal event registry")
    names={"Will Donald Trump win the 2024 US Presidential Election?":"Trump",
           "Will Kamala Harris win the 2024 US Presidential Election?":"Harris"}
    selected=[(r["condition_id"],names[r["question"]],str(r["clob_token_ids"][r["outcomes"].index("Yes")]))
              for r in registry if r["question"] in names]
    if len(selected)!=2:
        raise ValueError("Cannot identify the two focal YES contracts")
    columns=json.loads((SOURCE/"election_october_eligible_wallet_vectors_metadata.json").read_text(encoding="utf-8"))["subjects"]
    btc=json.loads((SOURCE/"bitcoin_title_audit.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp, duckdb.connect() as c:
        c.execute("SET threads=1")
        c.execute("SET memory_limit='3GB'")
        c.execute("SET temp_directory="+quote(tmp))
        c.execute("CREATE TABLE candidates(cond VARCHAR, candidate VARCHAR, yes_asset VARCHAR)")
        c.executemany("INSERT INTO candidates VALUES (?,?,?)",selected)
        c.execute("CREATE TABLE focal(cond VARCHAR PRIMARY KEY)")
        c.executemany("INSERT INTO focal VALUES (?)",[(r["condition_id"],) for r in registry])
        c.execute("CREATE TABLE btc(event_id VARCHAR PRIMARY KEY)")
        c.executemany("INSERT INTO btc VALUES (?)",[(str(r["event_id"]),) for r in btc])
        c.execute(f"CREATE VIEW fills AS SELECT * FROM {parquet(files)}")
        c.execute("""CREATE VIEW buys AS SELECT lower(t.taker) wallet,g.candidate,t.usdc_amount usd,
                cast(t.asset_id as varchar) asset,g.yes_asset FROM fills t JOIN candidates g
                ON lower(t.condition_id)=g.cond WHERE t.block_timestamp>=1725148800
                AND t.block_timestamp<1727740800 AND t.taker_direction='BUY'
                AND lower(t.outcome_label)='yes' AND t.usdc_amount>0 AND isfinite(t.usdc_amount)
                AND t.price>0 AND t.price<=1 AND isfinite(t.price) AND lower(t.maker)<>lower(t.taker)""")
        if c.execute("SELECT count(*) FROM buys WHERE asset IS DISTINCT FROM yes_asset").fetchone()[0]:
            raise ValueError("YES label/token disagreement")
        c.execute("""CREATE TABLE buyers AS SELECT wallet,
                 coalesce(sum(usd) FILTER(WHERE candidate='Harris'),0) h,
                 coalesce(sum(usd) FILTER(WHERE candidate='Trump'),0) t
                 FROM buys GROUP BY wallet""")
        values=c.execute("SELECT h+t FROM buyers ORDER BY wallet").fetchnumpy()
        size=np.asarray(next(iter(values.values())),float)
        lo,hi=np.quantile(size,[.05,.95],method="linear")
        c.execute("""CREATE TABLE selected AS SELECT wallet,
              (CASE WHEN h>0 THEN 'H' ELSE 'T' END)||
              (CASE WHEN h+t<=? THEN 'B' ELSE 'T' END) grp
              FROM buyers WHERE (h=0 OR t=0) AND (h+t<=? OR h+t>=?)""",[lo,lo,hi])
        maps=sorted((gamma/"condition_map").glob("part-*.parquet"))
        c.execute(f"""CREATE TABLE mapping AS SELECT '0x'||lower(hex(m.condition_id)) cond,
            t.event_id,t.category_id FROM {parquet(maps)} m
            JOIN read_parquet({quote(gamma/'event_subjects_v3.parquet')}) t
            ON cast(m.event_id AS VARCHAR)=t.event_id""")
        if c.execute("SELECT count(*)<>count(distinct cond) FROM mapping").fetchone()[0]:
            raise ValueError("Duplicated conditions")
        # Follow-up counts both roles, retaining every other mapped event (including other election events).
        c.execute("""CREATE TABLE wallet_event AS SELECT wallet,m.event_id,m.category_id,sum(amt) usd
            FROM (SELECT unnest([lower(t.maker),lower(t.taker)]) wallet,
            lower(t.condition_id) cond,t.usdc_amount amt FROM fills t
            WHERE t.block_timestamp>=1727740800 AND t.block_timestamp<1730419200
            AND t.usdc_amount>0 AND isfinite(t.usdc_amount)) x JOIN selected s USING(wallet)
            JOIN mapping m USING(cond) ANTI JOIN focal f USING(cond)
            GROUP BY wallet,m.event_id,m.category_id""")
        rows=c.execute("""SELECT s.grp,w.wallet,CASE WHEN b.event_id IS NOT NULL THEN 'bitcoin'
            ELSE w.category_id END subject,sum(w.usd) FROM wallet_event w
            JOIN selected s USING(wallet) LEFT JOIN btc b USING(event_id)
            GROUP BY s.grp,w.wallet,subject ORDER BY w.wallet,subject""").fetchall()
        # Transient addresses establish the original within-cohort lexical order. They are never emitted.
        keys={g:[r[0] for r in c.execute("SELECT wallet FROM selected WHERE grp=? ORDER BY wallet",[g]).fetchall()] for g,_,_ in FORMS}
    groups={g:np.zeros((len(ks),len(columns))) for g,ks in keys.items()}
    lookup={g:{w:i for i,w in enumerate(ws)} for g,ws in keys.items()}
    for g,w,col,usd in rows:
        groups[g][lookup[g][w],columns.index(col)]=usd
    return groups,columns,{"buffered_files":len(files),"buffered_bytes":sum(p.stat().st_size for p in files),
         "pooled_buyers":len(size),"lower_usdc":float(lo),"upper_usdc":float(hi)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw",type=Path,required=True,help="Raw files from code/acquisition/archive.py")
    p.add_argument("--report",type=Path,help="Optional new JSON file, aggregate diagnostics only")
    args=p.parse_args()
    groups,columns,report=build(args.raw)
    for name,drop_zero in (("eligible",False),("active",True)):
        stem="election_october_"+("eligible_" if name=="eligible" else "")+"wallet_vectors.csv"
        ref=read_ref(SOURCE/stem,columns); evaluations={}
        for g,x in groups.items():
            if drop_zero:
                x=x[x.sum(1)>0]
            if x.shape!=ref[g].shape:
                raise ValueError(f"{name}/{g} shape mismatch: {x.shape} vs {ref[g].shape}")
            error=float(np.max(np.abs(x-ref[g])))
            if not np.array_equal(x==0,ref[g]==0) or not np.allclose(x,ref[g],rtol=1e-11,atol=1e-7):
                raise ValueError(f"{name}/{g}: public-row or subject-amount mismatch; maximum error {error}")
            evaluations[g]={"rows":len(x),"max_cell_error_usdc":error,
                            "exact_float_match":bool(np.array_equal(x,ref[g]))}
        report[name]=evaluations
    print(json.dumps(report,indent=2))
    if args.report:
        with args.report.open("x",encoding="utf-8") as out:
            json.dump(report,out,indent=2);out.write("\n")

if __name__=="__main__":
    main()
