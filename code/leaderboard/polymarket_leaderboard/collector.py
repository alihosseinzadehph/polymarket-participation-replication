"""Collection and normalization logic for complete leaderboard snapshots."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .api import NotFound, PolymarketClient, utc_now


@dataclass(frozen=True)
class Query:
    category: str
    time_period: str
    order_by: str

    @property
    def key(self) -> str:
        return f"{self.category.lower()}_{self.time_period.lower()}_{self.order_by.lower()}"


@dataclass
class QueryResult:
    query: Query
    pages: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    reached_offset_ceiling: bool
    possibly_truncated: bool


Progress = Callable[[str], None]


def collect_leaderboards(
    client: PolymarketClient,
    *,
    categories: Iterable[str],
    time_periods: Iterable[str],
    orderings: Iterable[str],
    page_size: int,
    max_offset: int,
    workers: int,
    progress: Progress | None = None,
) -> list[QueryResult]:
    queries = [Query(category, period, ordering) for category in categories for period in time_periods for ordering in orderings]
    results: list[QueryResult] = []

    def run(query: Query) -> QueryResult:
        result = collect_query(client, query, page_size=page_size, max_offset=max_offset)
        if progress:
            suffix = " (configured offset ceiling reached)" if result.possibly_truncated else ""
            progress(f"{query.category}/{query.time_period}/{query.order_by}: {len(result.rows):,} rows{suffix}")
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, query): query for query in queries}
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: (item.query.category, item.query.time_period, item.query.order_by))


def collect_query(
    client: PolymarketClient,
    query: Query,
    *,
    page_size: int,
    max_offset: int,
) -> QueryResult:
    pages: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    offset = 0
    reached_ceiling = False
    final_page_full = False
    while offset <= max_offset:
        response = client.leaderboard(
            category=query.category,
            time_period=query.time_period,
            order_by=query.order_by,
            limit=page_size,
            offset=offset,
        )
        payload = response.data
        pages.append(
            {
                "offset": offset,
                "limit": page_size,
                "url": response.url,
                "fetched_at": response.fetched_at,
                "status": response.status,
                "headers": response.headers,
                "row_count": len(payload),
                "data": payload,
            }
        )
        for item in payload:
            if not isinstance(item, dict):
                raise TypeError(f"Expected leaderboard row object at offset {offset}, got {type(item).__name__}")
            row = dict(item)
            row.update(
                {
                    "category": query.category,
                    "time_period": query.time_period,
                    "order_by": query.order_by,
                    "page_offset": offset,
                    "page_fetched_at": response.fetched_at,
                    "source_url": response.url,
                }
            )
            rows.append(row)
        if len(payload) < page_size:
            break
        if offset + page_size > max_offset:
            reached_ceiling = True
            final_page_full = True
            break
        offset += page_size
    return QueryResult(
        query=query,
        pages=pages,
        rows=rows,
        reached_offset_ceiling=reached_ceiling,
        possibly_truncated=reached_ceiling and final_page_full,
    )


def collect_profiles(
    client: PolymarketClient,
    addresses: Iterable[str],
    *,
    workers: int,
    progress: Progress | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    unique = sorted({address.lower() for address in addresses if address})
    profiles: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []

    def fetch(address: str) -> tuple[str, dict[str, Any] | None, str | None]:
        try:
            response = client.profile(address)
            return address, {
                "url": response.url,
                "fetched_at": response.fetched_at,
                "status": response.status,
                "headers": response.headers,
                "data": response.data,
            }, None
        except NotFound as exc:
            return address, None, str(exc)
        except Exception as exc:  # A profile failure should not discard a complete leaderboard snapshot.
            return address, None, f"{type(exc).__name__}: {exc}"

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, address): address for address in unique}
        for future in as_completed(futures):
            address, profile, error = future.result()
            if profile is not None:
                profiles[address] = profile
            if error is not None:
                errors.append({"proxy_wallet": address, "error": error})
            completed += 1
            if progress and (completed == len(unique) or completed % 250 == 0):
                progress(f"profiles: {completed:,}/{len(unique):,}")
    return profiles, sorted(errors, key=lambda item: item["proxy_wallet"])


def make_long_rows(results: Iterable[QueryResult], snapshot_id: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for result in results:
        for source in result.rows:
            known = {
                "snapshot_id": snapshot_id,
                "category": source["category"],
                "time_period": source["time_period"],
                "order_by": source["order_by"],
                "rank": _integer(source.get("rank")),
                "proxy_wallet": source.get("proxyWallet"),
                "user_name": source.get("userName"),
                "volume": source.get("vol"),
                "pnl": source.get("pnl"),
                "profile_image": source.get("profileImage"),
                "x_username": source.get("xUsername"),
                "verified_badge": source.get("verifiedBadge"),
                "page_offset": source["page_offset"],
                "page_fetched_at": source["page_fetched_at"],
                "source_url": source["source_url"],
                "raw_json": json.dumps(
                    {key: value for key, value in source.items() if key not in {"category", "time_period", "order_by", "page_offset", "page_fetched_at", "source_url"}},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
            normalized.append(known)
    return sorted(
        normalized,
        key=lambda row: (row["category"], row["time_period"], row["order_by"], row["rank"] or 10**12, row["proxy_wallet"] or ""),
    )


def make_wide_rows(long_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in long_rows:
        wallet = (row.get("proxy_wallet") or "").lower()
        key = (row["snapshot_id"], row["category"], row["time_period"], wallet)
        output = merged.setdefault(
            key,
            {
                "snapshot_id": row["snapshot_id"],
                "category": row["category"],
                "time_period": row["time_period"],
                "proxy_wallet": wallet,
                "user_name": row.get("user_name"),
                "profile_image": row.get("profile_image"),
                "x_username": row.get("x_username"),
                "verified_badge": row.get("verified_badge"),
                "pnl": row.get("pnl"),
                "volume": row.get("volume"),
                "pnl_rank": None,
                "volume_rank": None,
                "in_pnl_leaderboard": False,
                "in_volume_leaderboard": False,
            },
        )
        for field in ("user_name", "profile_image", "x_username", "verified_badge", "pnl", "volume"):
            if row.get(field) not in (None, ""):
                output[field] = row[field]
        if row["order_by"] == "PNL":
            output["pnl_rank"] = row["rank"]
            output["in_pnl_leaderboard"] = True
        elif row["order_by"] == "VOL":
            output["volume_rank"] = row["rank"]
            output["in_volume_leaderboard"] = True
    return sorted(merged.values(), key=lambda row: (row["category"], row["time_period"], row["pnl_rank"] or 10**12, row["volume_rank"] or 10**12, row["proxy_wallet"]))


def make_trader_rows(
    long_rows: Iterable[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    snapshot_id: str,
) -> list[dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    for row in long_rows:
        wallet = (row.get("proxy_wallet") or "").lower()
        if not wallet:
            continue
        trader = aggregates.setdefault(
            wallet,
            {
                "snapshot_id": snapshot_id,
                "proxy_wallet": wallet,
                "user_name": row.get("user_name"),
                "profile_image": row.get("profile_image"),
                "x_username": row.get("x_username"),
                "verified_badge": row.get("verified_badge"),
                "leaderboard_appearances": 0,
                "categories_seen": set(),
                "time_periods_seen": set(),
                "orderings_seen": set(),
                "best_pnl_rank": None,
                "best_volume_rank": None,
                "max_reported_pnl": None,
                "max_reported_volume": None,
            },
        )
        trader["leaderboard_appearances"] += 1
        trader["categories_seen"].add(row["category"])
        trader["time_periods_seen"].add(row["time_period"])
        trader["orderings_seen"].add(row["order_by"])
        for field in ("user_name", "profile_image", "x_username", "verified_badge"):
            if row.get(field) not in (None, ""):
                trader[field] = row[field]
        rank_field = "best_pnl_rank" if row["order_by"] == "PNL" else "best_volume_rank"
        trader[rank_field] = _minimum(trader[rank_field], row.get("rank"))
        trader["max_reported_pnl"] = _maximum(trader["max_reported_pnl"], row.get("pnl"))
        trader["max_reported_volume"] = _maximum(trader["max_reported_volume"], row.get("volume"))

    output: list[dict[str, Any]] = []
    for wallet, trader in aggregates.items():
        profile_wrapper = profiles.get(wallet)
        profile = profile_wrapper["data"] if profile_wrapper else {}
        if profile:
            trader["user_name"] = profile.get("name") or trader["user_name"]
            trader["profile_image"] = profile.get("profileImage") or trader["profile_image"]
            trader["x_username"] = profile.get("xUsername") or trader["x_username"]
            if profile.get("verifiedBadge") is not None:
                trader["verified_badge"] = profile["verifiedBadge"]
        trader.update(
            {
                "profile_created_at": profile.get("createdAt"),
                "display_username_public": profile.get("displayUsernamePublic"),
                "bio": profile.get("bio"),
                "pseudonym": profile.get("pseudonym"),
                "profile_name": profile.get("name"),
                "associated_users_json": json.dumps(profile.get("users"), ensure_ascii=False, separators=(",", ":")) if profile.get("users") is not None else None,
                "profile_fetched_at": profile_wrapper.get("fetched_at") if profile_wrapper else None,
                "categories_seen": json.dumps(sorted(trader["categories_seen"]), separators=(",", ":")),
                "time_periods_seen": json.dumps(sorted(trader["time_periods_seen"]), separators=(",", ":")),
                "orderings_seen": json.dumps(sorted(trader["orderings_seen"]), separators=(",", ":")),
                "profile_json": json.dumps(profile, ensure_ascii=False, separators=(",", ":")) if profile else None,
            }
        )
        output.append(trader)
    return sorted(output, key=lambda row: (row["best_pnl_rank"] or 10**12, row["proxy_wallet"]))


def _integer(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _minimum(current: Any, candidate: Any) -> Any:
    if candidate is None:
        return current
    return candidate if current is None or candidate < current else current


def _maximum(current: Any, candidate: Any) -> Any:
    if candidate is None:
        return current
    return candidate if current is None or candidate > current else current
