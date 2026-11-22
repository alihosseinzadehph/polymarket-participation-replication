"""Collect a NEW Gamma event/mapping snapshot, not the historical snapshot."""
import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from archive import digest

ENDPOINT = "https://gamma-api.polymarket.com/events/keyset"

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def get(params):
    req = Request(ENDPOINT + "?" + urlencode(params), headers={"Accept":"application/json"})
    with urlopen(req, timeout=90) as response:
        return json.load(response)

def pages(closed, fetch):
    params = {"limit":500, "closed":str(closed).lower(), "order":"id", "ascending":"true"}
    seen = set()
    while True:
        doc = fetch(dict(params))
        if not isinstance(doc, dict) or not isinstance(doc.get("events"), list):
            raise ValueError("Unexpected Gamma page shape")
        yield doc["events"]
        cursor = doc.get("next_cursor")
        if not doc["events"] or not cursor:
            return
        if cursor in seen:
            raise ValueError("Repeated Gamma cursor; snapshot incomplete")
        seen.add(cursor)
        params["after_cursor"] = cursor

def json_list(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    return value if isinstance(value, list) else []

def write_row(out, row):
    out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

def capture(out, fetch=get):
    # Refuse to mix live pages with frozen files or another capture.
    out.mkdir(parents=True, exist_ok=False)
    start = now()
    seen, tags = set(), {}
    m = n = 0
    def stream(name):
        return gzip.open(out / name, "wt", encoding="utf-8", newline="\n")
    with stream("events.jsonl.gz") as events, stream("condition_event_map.jsonl.gz") as markets:
        for closed in (False, True):
            for rows in pages(closed, fetch):
                n += 1
                for e in rows:
                    if not isinstance(e,dict) or e.get("id") in (None,"") or str(e["id"]) in seen:
                        continue
                    seen.add(str(e["id"]))
                    ets = [{k:t.get(k) for k in ("id","label","slug")}
                           for t in (e.get("tags") or []) if isinstance(t,dict)]
                    for t in ets:
                        key = t["id"] or t["slug"] or t["label"]
                        if key is not None:
                            tags[str(key)] = t
                    event_fields = {"event_id":"id","slug":"slug","title":"title","category":"category",
                        "subcategory":"subcategory","start_date":"startDate","end_date":"endDate",
                        "created_at":"createdAt","closed":"closed","archived":"archived","neg_risk":"negRisk",
                        "reported_volume":"volume","reported_liquidity":"liquidity"}
                    write_row(events, {**{k:e.get(v) for k,v in event_fields.items()},
                                       "tags":ets, "retrieved_at_utc":start})
                    for mk in (e.get("markets") or []):
                        if not isinstance(mk,dict):
                            continue
                        m += 1
                        fields = {"market_id":"id","condition_id":"conditionId","market_slug":"slug",
                            "question":"question","category":"category","start_date":"startDate",
                            "end_date":"endDate","closed":"closed"}
                        write_row(markets, {**{k:mk.get(v) for k,v in fields.items()},
                            "event_id":e["id"], "event_slug":e.get("slug"),
                            "neg_risk":mk.get("negRisk",e.get("negRisk")),
                            "neg_risk_market_id":mk.get("negRiskMarketID",e.get("negRiskMarketID")),
                            "clob_token_ids":json_list(mk.get("clobTokenIds")),
                            "outcomes":json_list(mk.get("outcomes"))})
    with stream("tags.jsonl.gz") as dest:
        for key in sorted(tags):
            write_row(dest, tags[key])
    manifest = {"endpoint":ENDPOINT, "snapshot_type":"new live capture, NOT the frozen study snapshot",
            "start_utc":start, "end_utc":now(), "pages":n, "events":len(seen),
            "markets":m, "tags":len(tags),
            "files":{p.name:digest(p) for p in sorted(out.glob("*.gz"))}}
    (out/"complete.json").write_text(json.dumps(manifest,indent=2)+"\n", encoding="utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-current-to", type=Path, required=True,
                        help="Must be a new directory. An interrupted capture has no complete.json.")
    capture(parser.parse_args().fetch_current_to)
