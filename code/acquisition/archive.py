"""Inspect or fetch only the study's immutable, checksum-pinned upstream files."""
import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
import re
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

INVENTORY = Path(__file__).with_name("frozen_fill_inventory.json")
INVENTORY_SHA = "fa914f5e51a1b563ce04a3594fea1f2e69ce455081909984c50f8828165d9488"
WINDOW = (date(2022, 11, 21), date(2026, 8, 15))
REPOS = {
    "polymarket_v1": ("TimeSeventeen/Polymarket-v1", "5aa1b9d52316a8b2e789e81c8ae42c7ed532e8aa"),
    "polymarket_v2": ("TimeSeventeen/Polymarket-v2", "2a8dba6472f4c9ca558869e9cc1a65b9e1614264"),
}

def inventory(start, end):
    if not WINDOW[0] <= start <= end <= WINDOW[1]:
        raise ValueError("Specify inclusive dates inside the frozen study window.")
    # Account for Git checkouts configured to convert line endings on Windows.
    raw = INVENTORY.read_bytes().replace(b"\r\n",b"\n")
    if hashlib.sha256(raw).hexdigest() != INVENTORY_SHA:
        raise ValueError("Frozen inventory has been changed.")
    for source in json.loads(raw)["sources"]:
        name = source["name"]
        repo, revision = REPOS[name]
        if (source["repo_id"], source["resolved_revision"]) != (repo, revision):
            raise ValueError("Unexpected repository or revision")
        for f in source["files"]:
            if start <= date.fromisoformat(f["partition_date"]) <= end:
                rel = f["path"]
                if not re.fullmatch(r"daily_aligned(?:_multi)?/\d{4}_\d{2}_\d{2}\.parquet", rel):
                    raise ValueError("Invalid path in inventory")
                yield (name + "/" + rel, f["expected_size"], f["remote_sha256"],
                       f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{rel}")

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def checked_path(root, rel):
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("A path resolves outside the specified archive directory.")
    return path

def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "verify", "fetch"])
    parser.add_argument("--start", type=date.fromisoformat, default=WINDOW[0])
    parser.add_argument("--end", type=date.fromisoformat, default=WINDOW[1])
    parser.add_argument("--root", type=Path, help="External raw-data directory for fetch/verify.")
    args = parser.parse_args()
    rows = list(inventory(args.start, args.end))
    print(f"{len(rows)} files; {sum(r[1] for r in rows):,} compressed bytes "
          f"({args.start} through {args.end}, inclusive).", flush=True)
    if args.action == "plan":
        return
    if args.root is None:
        parser.error("--root is required for fetch or verify")
    if args.action == "fetch":
        args.root.mkdir(parents=True, exist_ok=True)
        missing = sum(n for rel,n,_,_ in rows if not checked_path(args.root,rel).exists())
        if missing and shutil.disk_usage(args.root).free < missing + 3 * 1024**3:
            raise ValueError("Insufficient disk space for the selected files plus a 3 GiB reserve.")
    for rel, size, sha, url in rows:
        target = checked_path(args.root, rel)
        if target.is_file():
            if target.stat().st_size != size or digest(target) != sha:
                raise ValueError(f"Existing file differs; will not overwrite: {rel}")
        elif args.action == "verify":
            raise FileNotFoundError(rel)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=target.parent, suffix=".partial", delete=False) as tmp:
                partial = Path(tmp.name)
            try:
                with urlopen(url, timeout=120) as response, partial.open("wb") as out:
                    shutil.copyfileobj(response, out, length=2**20)
                if partial.stat().st_size != size or digest(partial) != sha:
                    raise ValueError(f"Downloaded bytes do not match frozen inventory: {rel}")
                if target.exists():
                    raise FileExistsError(target)
                partial.rename(target)
            finally:
                partial.unlink(missing_ok=True)
    print(f"Verified {len(rows)} files against their frozen SHA-256 digests.")

if __name__ == "__main__":
    run()
