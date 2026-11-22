"""Small dependency-free client for Polymarket's public JSON APIs."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .constants import LEADERBOARD_URL, PROFILE_URL


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class JsonResponse:
    url: str
    fetched_at: str
    status: int
    headers: dict[str, str]
    data: Any


class ApiError(RuntimeError):
    """Raised after a public API request exhausts its retry budget."""


class NotFound(ApiError):
    """Raised when a public profile does not exist."""


class PolymarketClient:
    def __init__(
        self,
        *,
        timeout: float = 30.0,
        retries: int = 5,
        backoff: float = 0.75,
        user_agent: str = "polymarket-leaderboard-dataset/0.1",
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.user_agent = user_agent
        self._opener = opener

    def leaderboard(
        self,
        *,
        category: str,
        time_period: str,
        order_by: str,
        limit: int,
        offset: int,
    ) -> JsonResponse:
        response = self._get_json(
            LEADERBOARD_URL,
            {
                "category": category,
                "timePeriod": time_period,
                "orderBy": order_by,
                "limit": limit,
                "offset": offset,
            },
        )
        if not isinstance(response.data, list):
            raise ApiError(f"Expected a JSON array from {response.url}")
        return response

    def profile(self, address: str) -> JsonResponse:
        try:
            response = self._get_json(PROFILE_URL, {"address": address})
        except HTTPError as exc:
            if exc.code == 404:
                raise NotFound(f"No public profile for {address}") from exc
            raise
        if not isinstance(response.data, dict):
            raise ApiError(f"Expected a JSON object from {response.url}")
        return response

    def _get_json(self, base_url: str, params: Mapping[str, object]) -> JsonResponse:
        url = f"{base_url}?{urlencode(params)}"
        last_error: BaseException | None = None
        for attempt in range(self.retries + 1):
            request = Request(
                url,
                headers={"Accept": "application/json", "User-Agent": self.user_agent},
                method="GET",
            )
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    body = response.read()
                    return JsonResponse(
                        url=response.geturl(),
                        fetched_at=utc_now(),
                        status=int(response.status),
                        headers=_useful_headers(dict(response.headers.items())),
                        data=json.loads(body.decode("utf-8")),
                    )
            except HTTPError as exc:
                if exc.code == 404:
                    raise
                last_error = exc
                if exc.code not in {408, 425, 429, 500, 502, 503, 504} or attempt >= self.retries:
                    break
                delay = _retry_delay(exc.headers, self.backoff, attempt)
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                delay = self.backoff * (2**attempt) + random.uniform(0, 0.25)
            time.sleep(delay)
        raise ApiError(f"GET {url} failed after {self.retries + 1} attempts: {last_error}") from last_error


def _useful_headers(headers: Mapping[str, str]) -> dict[str, str]:
    wanted = {"date", "etag", "last-modified", "content-type", "cf-ray", "cache-control"}
    return {key.lower(): value for key, value in headers.items() if key.lower() in wanted}


def _retry_delay(headers: Mapping[str, str] | None, backoff: float, attempt: int) -> float:
    retry_after = headers.get("Retry-After") if headers else None
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                now = datetime.now(retry_at.tzinfo or timezone.utc)
                return max(0.0, (retry_at - now).total_seconds())
            except (TypeError, ValueError):
                pass
    return backoff * (2**attempt) + random.uniform(0, 0.25)

