"""Atomic snapshot writers for raw JSON, CSV, JSONL, and SQLite."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

from .collector import QueryResult


LONG_FIELDS = (
    "snapshot_id", "category", "time_period", "order_by", "rank", "proxy_wallet", "user_name",
    "volume", "pnl", "profile_image", "x_username", "verified_badge", "page_offset",
    "page_fetched_at", "source_url", "raw_json",
)
WIDE_FIELDS = (
    "snapshot_id", "category", "time_period", "proxy_wallet", "user_name", "profile_image",
    "x_username", "verified_badge", "pnl", "volume", "pnl_rank", "volume_rank",
    "in_pnl_leaderboard", "in_volume_leaderboard",
)
TRADER_FIELDS = (
    "snapshot_id", "proxy_wallet", "user_name", "profile_name", "pseudonym", "bio",
    "profile_created_at", "display_username_public", "profile_image", "x_username", "verified_badge",
    "associated_users_json", "profile_fetched_at", "leaderboard_appearances", "categories_seen",
    "time_periods_seen", "orderings_seen", "best_pnl_rank", "best_volume_rank",
    "max_reported_pnl", "max_reported_volume", "profile_json",
)


def write_snapshot(
    snapshot_dir: Path,
    *,
    results: Sequence[QueryResult],
    profiles: dict[str, dict[str, Any]],
    long_rows: Sequence[dict[str, Any]],
    wide_rows: Sequence[dict[str, Any]],
    trader_rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = snapshot_dir / "raw" / "leaderboards"
    raw_dir.mkdir(parents=True)
    for result in results:
        write_json_atomic(
            raw_dir / f"{result.query.key}.json",
            {
                "category": result.query.category,
                "time_period": result.query.time_period,
                "order_by": result.query.order_by,
                "reached_offset_ceiling": result.reached_offset_ceiling,
                "possibly_truncated": result.possibly_truncated,
                "pages": result.pages,
            },
        )
    profiles_dir = snapshot_dir / "raw" / "profiles"
    profiles_dir.mkdir(parents=True)
    for wallet, profile in profiles.items():
        write_json_atomic(profiles_dir / f"{wallet}.json", profile)

    write_csv_atomic(snapshot_dir / "leaderboard.csv", long_rows, LONG_FIELDS)
    write_jsonl_atomic(snapshot_dir / "leaderboard.jsonl", long_rows)
    write_csv_atomic(snapshot_dir / "leaderboard_wide.csv", wide_rows, WIDE_FIELDS)
    write_csv_atomic(snapshot_dir / "traders.csv", trader_rows, TRADER_FIELDS)
    write_sqlite_atomic(snapshot_dir / "leaderboard.sqlite", manifest, long_rows, wide_rows, trader_rows)
    write_json_atomic(snapshot_dir / "manifest.json", manifest)


def write_latest(output_dir: Path, snapshot_id: str) -> None:
    _atomic_text(output_dir / "latest.txt", snapshot_id + "\n")


def write_json_atomic(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    _atomic_text(path, content)


def write_csv_atomic(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_sqlite_atomic(
    path: Path,
    manifest: dict[str, Any],
    long_rows: Sequence[dict[str, Any]],
    wide_rows: Sequence[dict[str, Any]],
    trader_rows: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        connection = sqlite3.connect(temporary)
        try:
            connection.executescript(
                """
                PRAGMA journal_mode = DELETE;
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
                CREATE TABLE leaderboard (
                    snapshot_id TEXT NOT NULL, category TEXT NOT NULL, time_period TEXT NOT NULL,
                    order_by TEXT NOT NULL, rank INTEGER, proxy_wallet TEXT, user_name TEXT,
                    volume REAL, pnl REAL, profile_image TEXT, x_username TEXT, verified_badge INTEGER,
                    page_offset INTEGER NOT NULL, page_fetched_at TEXT NOT NULL, source_url TEXT NOT NULL,
                    raw_json TEXT NOT NULL
                );
                CREATE INDEX idx_leaderboard_slice ON leaderboard(category, time_period, order_by, rank);
                CREATE INDEX idx_leaderboard_wallet ON leaderboard(proxy_wallet);
                CREATE TABLE leaderboard_wide (
                    snapshot_id TEXT NOT NULL, category TEXT NOT NULL, time_period TEXT NOT NULL,
                    proxy_wallet TEXT NOT NULL, user_name TEXT, profile_image TEXT, x_username TEXT,
                    verified_badge INTEGER, pnl REAL, volume REAL, pnl_rank INTEGER, volume_rank INTEGER,
                    in_pnl_leaderboard INTEGER NOT NULL, in_volume_leaderboard INTEGER NOT NULL,
                    PRIMARY KEY(snapshot_id, category, time_period, proxy_wallet)
                );
                CREATE INDEX idx_wide_pnl ON leaderboard_wide(category, time_period, pnl_rank);
                CREATE INDEX idx_wide_volume ON leaderboard_wide(category, time_period, volume_rank);
                CREATE TABLE traders (
                    snapshot_id TEXT NOT NULL, proxy_wallet TEXT NOT NULL, user_name TEXT, profile_name TEXT,
                    pseudonym TEXT, bio TEXT, profile_created_at TEXT, display_username_public INTEGER,
                    profile_image TEXT, x_username TEXT, verified_badge INTEGER, associated_users_json TEXT,
                    profile_fetched_at TEXT, leaderboard_appearances INTEGER NOT NULL, categories_seen TEXT NOT NULL,
                    time_periods_seen TEXT NOT NULL, orderings_seen TEXT NOT NULL, best_pnl_rank INTEGER,
                    best_volume_rank INTEGER, max_reported_pnl REAL, max_reported_volume REAL, profile_json TEXT,
                    PRIMARY KEY(snapshot_id, proxy_wallet)
                );
                """
            )
            connection.executemany(
                "INSERT INTO metadata(key, value_json) VALUES (?, ?)",
                [(key, json.dumps(value, ensure_ascii=False, separators=(",", ":"))) for key, value in manifest.items()],
            )
            _insert_dicts(connection, "leaderboard", LONG_FIELDS, long_rows)
            _insert_dicts(connection, "leaderboard_wide", WIDE_FIELDS, wide_rows)
            _insert_dicts(connection, "traders", TRADER_FIELDS, trader_rows)
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _insert_dicts(connection: sqlite3.Connection, table: str, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    quoted = ", ".join(f'"{field}"' for field in fields)
    placeholders = ", ".join("?" for _ in fields)
    statement = f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders})'
    connection.executemany(statement, [tuple(_sqlite_value(row.get(field)) for field in fields) for row in rows])


def _sqlite_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    return value


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

