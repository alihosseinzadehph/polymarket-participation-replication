"""Exact sparse wallet--event aggregation of the existing frozen trade archive.

No download, ranking cutoff, similarity projection, edge pruning, or sampling.
The default command prepares dimensions and reports the plan; --all performs
the bounded, checkpointed scan. Integer IDs compress identifiers, not nodes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import threading
import time
from pathlib import Path

import duckdb

VERSION = "full-participation-v1"
BASE = Path(__file__).resolve().parent
WORKSPACE = BASE.parent
PACKAGE_ROOT = BASE.parents[1]
EXTERNAL_INPUTS = Path(os.environ.get("POLYMARKET_INPUT_ROOT", PACKAGE_ROOT / "external_inputs")).expanduser()
DEFAULT_SOURCE = Path(os.environ.get("POLYMARKET_SOURCE_ROOT", EXTERNAL_INPUTS / "dataset/source")).expanduser()
DEFAULT_MANIFEST = Path(os.environ.get("POLYMARKET_VERIFICATION_REPORT", DEFAULT_SOURCE.parent / "manifests/verification_report.json")).expanduser()
DERIVED = Path(os.environ.get("POLYMARKET_DERIVED_ROOT", EXTERNAL_INPUTS / "derived")).expanduser()
TAXONOMY = Path(os.environ.get("POLYMARKET_TAXONOMY", EXTERNAL_INPUTS / "taxonomy/events_v3.parquet")).expanduser()
START = dt.datetime(2022, 11, 21, tzinfo=dt.timezone.utc)
END = dt.datetime(2026, 8, 16, tzinfo=dt.timezone.utc)


def sql(value):
    return "'" + str(value).replace("\\", "/").replace("'", "''") + "'"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def quarter_bounds(name):
    year, q = int(name[:4]), int(name[-1])
    start = dt.datetime(year, 3 * q - 2, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(year + (q == 4), 1 if q == 4 else 3 * q + 1, 1, tzinfo=dt.timezone.utc)
    return max(start, START), min(end, END)


def source_plan(root, verification):
    records = json.loads(verification.read_text(encoding="utf-8"))["files"]
    groups = {}
    seen = set()
    for item in records:
        if item["status"] != "valid":
            raise ValueError("Archived verification contains an invalid input")
        version, rel = item["key"].split(":", 1)
        if version not in {"polymarket_v1", "polymarket_v2"} or Path(rel).parts[0] not in {"daily_aligned", "daily_aligned_multi"}:
            raise ValueError(f"Unexpected source layer: {item['key']}")
        file = root / version / rel
        if file.resolve() in seen or not file.is_file() or file.stat().st_size != item["size"]:
            raise ValueError(f"Duplicate, absent or resized source: {file}")
        seen.add(file.resolve())
        day = dt.datetime.strptime(file.stem, "%Y_%m_%d").replace(tzinfo=dt.timezone.utc)
        if not START <= day < END:
            raise ValueError(f"File day outside frozen observation window: {file}")
        quarter = f"{day.year}-Q{(day.month - 1) // 3 + 1}"
        groups.setdefault(quarter, []).append({**item, "current_path": str(file)})
    discovered = {p.resolve() for p in root.glob("*/daily_aligned*/*.parquet")}
    if seen != discovered:
        raise ValueError("Archive paths differ from the previously verified file inventory")
    return dict(sorted(groups.items()))


def connect(output, memory_gb=4, threads=4):
    temp = output / "duckdb_spill"
    temp.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads={threads}")
    con.execute(f"SET memory_limit='{memory_gb}GB'")
    con.execute(f"SET temp_directory={sql(temp)}")
    con.execute("SET max_temp_directory_size='8GB'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET TimeZone='UTC'")
    return con


class ResourceGuard:
    def __init__(self, con, output, reserve_gib, output_gib):
        self.con, self.output = con, output
        self.reserve, self.maximum = reserve_gib * 2**30, output_gib * 2**30
        self.stop = threading.Event()
        self.reason = None

    def check(self):
        free = shutil.disk_usage(self.output).free
        size = sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file() and "duckdb_spill" not in p.parts)
        if free < self.reserve:
            raise RuntimeError("Disk reserve reached; aggregation stopped without deleting inputs")
        if size > self.maximum:
            raise RuntimeError("Derived-output budget reached; aggregation stopped without deleting inputs")

    def __enter__(self):
        self.check()
        def watch():
            while not self.stop.wait(2):
                try:
                    self.check()
                except Exception as exc:
                    self.reason = str(exc)
                    self.con.interrupt()
                    return
        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        if self.reason:
            raise RuntimeError(self.reason)


def write_parquet(con, query, final):
    temporary = final.with_suffix(".pending.parquet")
    if temporary.exists():
        raise RuntimeError(f"Unverified interrupted output exists; inspect before retry: {temporary}")
    con.execute(f"COPY ({query}) TO {sql(temporary)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)")
    return temporary


def dimensions(con, output):
    dims = output / "dimensions"
    dims.mkdir(exist_ok=True)
    queries = {
        "wallets": f"SELECT (row_number() OVER (ORDER BY wallet)-1)::INTEGER wallet_id, lower(wallet) wallet FROM read_parquet({sql(DERIVED/'wallet_totals.parquet')})",
        "events": f"""SELECT (row_number() OVER (ORDER BY e.event_id)-1)::INTEGER event_index,
          e.event_id, e.category_id, e.category, e.title, t.usd_volume observed_fill_usd,
          t.trade_count observed_fills
          FROM read_parquet({sql(DERIVED/'event_totals.parquet')}) t
          JOIN read_parquet({sql(TAXONOMY)}) e USING(event_id)
          WHERE t.usd_volume>0 AND isfinite(t.usd_volume)""",
    }
    inputs = {str(p): sha(p) for p in [DERIVED/'wallet_totals.parquet', DERIVED/'event_totals.parquet', DERIVED/'condition_event_map_clean.parquet', DERIVED/'quarterly_event_totals.parquet', TAXONOMY]}
    meta = dims / "manifest.json"
    if meta.exists():
        old = json.loads(meta.read_text())
        if old["inputs"] != inputs or old["version"] != VERSION:
            raise RuntimeError("Dimension input or extraction version changed")
        for name, info in old["files"].items():
            if sha(dims / f"{name}.parquet") != info["sha256"]:
                raise RuntimeError(f"Changed dimension file: {name}")
    else:
        if any(dims.glob("*.parquet")):
            raise RuntimeError("Dimension outputs exist without their verified manifest")
        for name, query in queries.items():
            final = dims / f"{name}.parquet"
            write_parquet(con, query, final).replace(final)
        files = {}
        for name, key in [("wallets", "wallet"), ("events", "event_id")]:
            p = dims / f"{name}.parquet"
            n, unique, absent = con.execute(f"SELECT count(*), count(DISTINCT {key}),count(*) FILTER(WHERE {key} IS NULL) FROM read_parquet({sql(p)})").fetchone()
            if n != unique or absent:
                raise RuntimeError(f"Nonunique or absent {key} dimension")
            files[name] = {"rows": n, "sha256": sha(p), "bytes": p.stat().st_size}
        if files['events']['rows'] != 764378 or files['wallets']['rows'] != 3106517:
            raise RuntimeError("Dimensions differ from the frozen archive's audited populations")
        save_json(meta, {"version": VERSION, "inputs": inputs, "files": files})
    con.execute(f"CREATE TABLE wallets AS SELECT * FROM read_parquet({sql(dims/'wallets.parquet')})")
    con.execute(f"CREATE TABLE events AS SELECT * FROM read_parquet({sql(dims/'events.parquet')})")
    con.execute(f"CREATE TABLE conditions AS SELECT m.condition_id,e.event_index FROM read_parquet({sql(DERIVED/'condition_event_map_clean.parquet')}) m JOIN events e USING(event_id)")
    n, unique = con.execute("SELECT count(*), count(DISTINCT condition_id) FROM conditions").fetchone()
    if n != unique:
        raise RuntimeError("Condition-to-event relation is not unique")


def reconcile_quarter(con, file, name):
    start, end = quarter_bounds(name)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE expected AS
      SELECT e.event_index,sum(q.trade_count)::BIGINT fills,sum(q.usd_volume)::DOUBLE usd
      FROM read_parquet({sql(DERIVED/'quarterly_event_totals.parquet')}) q JOIN events e USING(event_id)
      WHERE q.quarter >= TIMESTAMPTZ {sql(start.isoformat())} AND q.quarter < TIMESTAMPTZ {sql(end.isoformat())}
      GROUP BY e.event_index""")
    # The partial first quarter's stored label is Oct 1, not Nov 21.
    if name == "2022-Q4":
        con.execute(f"""CREATE OR REPLACE TEMP TABLE expected AS
          SELECT e.event_index,sum(q.trade_count)::BIGINT fills,sum(q.usd_volume)::DOUBLE usd
          FROM read_parquet({sql(DERIVED/'quarterly_event_totals.parquet')}) q JOIN events e USING(event_id)
          WHERE q.quarter >= TIMESTAMPTZ '2022-10-01' AND q.quarter < TIMESTAMPTZ '2023-01-01'
          GROUP BY e.event_index""")
    con.execute(f"CREATE OR REPLACE TEMP TABLE actual AS SELECT event_index,sum(participation_fills)::BIGINT fills,sum(usd)::DOUBLE usd FROM read_parquet({sql(file)}) GROUP BY event_index")
    errors = con.execute("""SELECT count(*) FROM actual a FULL JOIN expected e USING(event_index)
      WHERE a.fills IS NULL OR e.fills IS NULL OR a.fills != 2*e.fills
      OR abs(a.usd-2*e.usd)>greatest(1e-5,1e-9*abs(2*e.usd))""").fetchone()[0]
    rows, wallets, events, fills, usd, invalid = con.execute(f"""SELECT count(*),count(DISTINCT wallet_id),count(DISTINCT event_index),
      sum(participation_fills)::BIGINT,sum(usd),count(*) FILTER(WHERE usd<=0 OR NOT isfinite(usd) OR wallet_id IS NULL OR event_index IS NULL)
      FROM read_parquet({sql(file)})""").fetchone()
    if errors or invalid:
        raise RuntimeError(f"Quarter conservation failed: {errors} events, {invalid} invalid edges")
    return {"quarter": name, "pairs": rows, "wallets": wallets, "events": events,
            "participation_fills": fills, "participation_usd": usd, "eventwise_reconciliation_errors": errors}


def build_quarter(con, output, name, records, guard_args):
    target = output / f"participation_{name}.parquet"
    checkpoint = output / f"participation_{name}.json"
    signature = hashlib.sha256(json.dumps([(x['key'], x['sha256']) for x in records]).encode()).hexdigest()
    if checkpoint.exists():
        old = json.loads(checkpoint.read_text())
        if old['input_signature'] != signature or old['version'] != VERSION or sha(target) != old['sha256']:
            raise RuntimeError(f"Checkpoint mismatch: {name}")
        print(json.dumps({"resume_verified": name, "pairs": old['pairs']}), flush=True)
        return old
    if target.exists():
        raise RuntimeError(f"Output exists without verified checkpoint: {target}")
    start, end = quarter_bounds(name)
    files = '[' + ','.join(sql(r['current_path']) for r in records) + ']'
    query = f"""SELECT wallet_id,event_index,count(*)::BIGINT participation_fills,sum(usdc_amount)::DOUBLE usd FROM (
      SELECT unnest([wm.wallet_id,wt.wallet_id]) wallet_id,c.event_index,t.usdc_amount
      FROM read_parquet({files},union_by_name=true) t
      JOIN conditions c ON lower(t.condition_id)=c.condition_id
      JOIN wallets wm ON lower(t.maker)=wm.wallet
      JOIN wallets wt ON lower(t.taker)=wt.wallet
      WHERE t.block_timestamp >= {int(start.timestamp())} AND t.block_timestamp < {int(end.timestamp())}
        AND t.usdc_amount>0 AND isfinite(t.usdc_amount)
      ) GROUP BY wallet_id,event_index"""
    began = time.time()
    print(json.dumps({"start": name, "raw_rows": sum(x['row_count'] for x in records), "files": len(records)}), flush=True)
    with ResourceGuard(con, output, *guard_args):
        pending = write_parquet(con, query, target)
        stats = reconcile_quarter(con, pending, name)
        stats.update(version=VERSION, input_signature=signature, source_files=len(records),
                     source_rows=sum(x['row_count'] for x in records), seconds=time.time()-began,
                     sha256=sha(pending), bytes=pending.stat().st_size, builder_sha256=sha(__file__))
        pending.replace(target)
        save_json(checkpoint, stats)
    print(json.dumps(stats), flush=True)
    return stats


def main():
    global DERIVED, TAXONOMY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--verification', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--input-derived', type=Path, default=DERIVED,
                        help='Directory containing the four required preprocessed Parquet inputs')
    parser.add_argument('--taxonomy', type=Path, default=TAXONOMY,
                        help='Path to the frozen events_v3.parquet taxonomy')
    parser.add_argument('--output', type=Path, default=BASE/'data')
    parser.add_argument('--quarter', action='append')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--memory-gb', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--reserve-gib', type=float, default=20)
    parser.add_argument('--output-gib', type=float, default=8)
    args = parser.parse_args()
    DERIVED = args.input_derived.expanduser().resolve()
    TAXONOMY = args.taxonomy.expanduser().resolve()
    args.source_root = args.source_root.expanduser().resolve()
    args.verification = args.verification.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    groups = source_plan(args.source_root, args.verification)
    plan = {"version": VERSION, "archive_verification_sha256": sha(args.verification),
            "no_new_download": True, "no_node_or_edge_selection": True,
            "quarters": [{"quarter": q, "files": len(rs), "rows": sum(r['row_count'] for r in rs), "raw_bytes": sum(r['size'] for r in rs)} for q,rs in groups.items()],
            "memory_limit_gb": args.memory_gb, "temporary_disk_limit_gb": 8,
            "minimum_free_disk_gib": args.reserve_gib, "derived_output_limit_gib": args.output_gib,
            "note": "Input sizes/inventory checked now; full raw hashes were verified by the archived audit, not recomputed by this script."}
    save_json(args.output/'build_plan.json', plan)
    print(json.dumps(plan), flush=True)
    con = connect(args.output,args.memory_gb,args.threads)
    dimensions(con,args.output)
    selected = list(groups) if args.all else (args.quarter or [])
    for name in selected:
        build_quarter(con,args.output,name,groups[name],(args.reserve_gib,args.output_gib))
    con.close()


if __name__ == '__main__':
    main()
