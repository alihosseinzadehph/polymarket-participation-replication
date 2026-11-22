"""Command-line interface for collecting leaderboard snapshots."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from . import __version__
from .api import PolymarketClient, utc_now
from .collector import collect_leaderboards, collect_profiles, make_long_rows, make_trader_rows, make_wide_rows
from .constants import CATEGORIES, MAX_OFFSET, MAX_PAGE_SIZE, ORDERINGS, TIME_PERIODS
from .storage import write_latest, write_snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Polymarket public trader leaderboard datasets.")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect", help="collect one immutable snapshot")
    collect.add_argument("--output-dir", type=Path, default=Path("data"), help="dataset root (default: data)")
    collect.add_argument("--snapshot-id", help="safe directory name; defaults to the UTC start time")
    collect.add_argument("--categories", default=",".join(CATEGORIES), help="comma-separated categories")
    collect.add_argument("--periods", default=",".join(TIME_PERIODS), help="comma-separated DAY,WEEK,MONTH,ALL")
    collect.add_argument("--order-by", default=",".join(ORDERINGS), help="comma-separated PNL,VOL")
    collect.add_argument("--page-size", type=int, default=MAX_PAGE_SIZE, help="rows/request, maximum 50")
    collect.add_argument("--max-offset", type=int, default=MAX_OFFSET, help="last API offset, maximum 1000")
    collect.add_argument("--workers", type=int, default=12, help="parallel public API requests")
    collect.add_argument("--no-profiles", action="store_true", help="skip public profile enrichment")
    collect.add_argument("--timeout", type=float, default=30.0, help="request timeout in seconds")
    collect.add_argument("--retries", type=int, default=5, help="retry count for transient failures")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "collect":
        return _collect(args)
    raise AssertionError(args.command)


def _collect(args: argparse.Namespace) -> int:
    categories = _selection(args.categories, CATEGORIES, "category")
    periods = _selection(args.periods, TIME_PERIODS, "period")
    orderings = _selection(args.order_by, ORDERINGS, "ordering")
    if not 1 <= args.page_size <= MAX_PAGE_SIZE:
        raise SystemExit(f"--page-size must be between 1 and {MAX_PAGE_SIZE}")
    if not 0 <= args.max_offset <= MAX_OFFSET:
        raise SystemExit(f"--max-offset must be between 0 and {MAX_OFFSET}")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.retries < 0:
        raise SystemExit("--retries cannot be negative")

    started_at = utc_now()
    snapshot_id = args.snapshot_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", snapshot_id):
        raise SystemExit("--snapshot-id must contain only letters, digits, dot, underscore, or hyphen")
    output_dir = args.output_dir.resolve()
    snapshot_dir = output_dir / "snapshots" / snapshot_id
    if snapshot_dir.exists():
        raise SystemExit(f"Snapshot already exists: {snapshot_dir}")

    client = PolymarketClient(timeout=args.timeout, retries=args.retries)
    progress = lambda message: print(message, file=sys.stderr, flush=True)
    results = collect_leaderboards(
        client,
        categories=categories,
        time_periods=periods,
        orderings=orderings,
        page_size=args.page_size,
        max_offset=args.max_offset,
        workers=args.workers,
        progress=progress,
    )
    long_rows = make_long_rows(results, snapshot_id)
    addresses = [row["proxy_wallet"] for row in long_rows if row.get("proxy_wallet")]
    profiles: dict[str, dict] = {}
    profile_errors: list[dict[str, str]] = []
    if not args.no_profiles:
        profiles, profile_errors = collect_profiles(client, addresses, workers=args.workers, progress=progress)
    wide_rows = make_wide_rows(long_rows)
    trader_rows = make_trader_rows(long_rows, profiles, snapshot_id)
    completed_at = utc_now()
    manifest = {
        "schema_version": 1,
        "collector_version": __version__,
        "snapshot_id": snapshot_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "source": "https://data-api.polymarket.com/v1/leaderboard",
        "profile_source": "https://gamma-api.polymarket.com/public-profile" if not args.no_profiles else None,
        "parameters": {
            "categories": categories,
            "time_periods": periods,
            "orderings": orderings,
            "page_size": args.page_size,
            "max_offset": args.max_offset,
            "profiles": not args.no_profiles,
        },
        "counts": {
            "queries": len(results),
            "api_pages": sum(len(result.pages) for result in results),
            "leaderboard_rows": len(long_rows),
            "wide_rows": len(wide_rows),
            "unique_traders": len(trader_rows),
            "profiles": len(profiles),
            "profile_errors": len(profile_errors),
        },
        "possibly_truncated_queries": [result.query.key for result in results if result.possibly_truncated],
        "profile_errors": profile_errors,
        "notes": [
            "DAY, WEEK, MONTH, and ALL are API-defined rolling/aggregate windows captured at fetch time, not event-time buckets.",
            "A query is marked possibly truncated when the API returns a full page at the configured maximum offset (the default is the documented API maximum).",
            "Ranks can move between pages because the public API does not expose snapshot isolation.",
        ],
    }
    write_snapshot(
        snapshot_dir,
        results=results,
        profiles=profiles,
        long_rows=long_rows,
        wide_rows=wide_rows,
        trader_rows=trader_rows,
        manifest=manifest,
    )
    write_latest(output_dir, snapshot_id)
    print(json.dumps({"snapshot_dir": str(snapshot_dir), **manifest["counts"]}, indent=2))
    return 0


def _selection(raw: str, allowed: Sequence[str], label: str) -> list[str]:
    values = list(dict.fromkeys(value.strip().upper() for value in raw.split(",") if value.strip()))
    if not values:
        raise SystemExit(f"At least one {label} is required")
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise SystemExit(f"Unknown {label}(s): {', '.join(invalid)}. Allowed: {', '.join(allowed)}")
    return values


if __name__ == "__main__":
    raise SystemExit(main())
