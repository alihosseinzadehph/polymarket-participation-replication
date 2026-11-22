from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import REPO_ROOT, load_yaml


def normalize_tag(value: str) -> str:
    value = value.casefold().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


@dataclass(frozen=True)
class CategoryRule:
    category_id: str
    label: str
    legacy: str
    tags: frozenset[str]
    patterns: tuple[re.Pattern[str], ...]


class Taxonomy:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or REPO_ROOT / "config" / "taxonomy.yaml"
        config = load_yaml(self.path)
        self.version = str(config["taxonomy_version"])
        self.rules: list[CategoryRule] = []
        for item in config["categories"]:
            patterns = tuple(re.compile(p, re.IGNORECASE) for p in item.get("text", []))
            self.rules.append(
                CategoryRule(
                    category_id=str(item["id"]),
                    label=str(item["label"]),
                    legacy=str(item["legacy"]),
                    tags=frozenset(normalize_tag(str(x)) for x in item.get("tags", [])),
                    patterns=patterns,
                )
            )
        if not self.rules or self.rules[-1].category_id != "other":
            raise ValueError("The final taxonomy rule must be the 'other' fallback")

    def classify(
        self,
        title: str | None,
        slug: str | None,
        tags: Iterable[Any] | None,
    ) -> dict[str, Any]:
        normalized_tags: set[str] = set()
        for item in tags or []:
            if isinstance(item, dict):
                for field in ("slug", "label"):
                    if item.get(field):
                        normalized_tags.add(normalize_tag(str(item[field])))
            elif item:
                normalized_tags.add(normalize_tag(str(item)))

        text = f"{title or ''} {slug or ''}".casefold()
        # v3: generic metadata tags must not veto a more specific outcome title.
        # Geographic and contract-form evidence is required for narrow labels.
        by_id = {rule.category_id: rule for rule in self.rules}
        crypto = bool(normalized_tags & {"crypto", "bitcoin", "ethereum", "solana", "xrp"}) or bool(
            re.search(r"\b(bitcoin|btc|ethereum|eth|solana|xrp|dogecoin)\b", text)
        )
        intraday = crypto and (
            bool(normalized_tags & {"5m", "15m", "1h", "4h", "hourly"})
            or bool(re.search(r"\b(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)\s*[-–]\s*\d{1,2}(?::\d{2})?\s*(?:am|pm))\b", text))
            or bool(re.search(r"\b(?:updown|up-or-down)-(?:5m|15m|1h|4h)-", text))
            or bool(re.search(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\s*[-–]\s*(?:[01]?\d|2[0-3]):[0-5]\d\s*(?:utc|et|est|edt)\b", text))
            or bool(re.search(r"\bprice at\b.*\b(?:am|pm|utc)\b", text))
        )
        us_context = bool(normalized_tags & {"us-politics", "united-states", "us-election", "us-presidential-election", "usa-election", "trump-presidency", "nyc", "california-midterm", "us-state-elections", "us-local-politics", "jan-6", "january-6", "deprec-us-election", "deprec-usa-election", "trump", "biden", "kamala", "harris", "los-angeles", "california"}) or bool(
            re.search(r"\b(?:trump|biden|harris|white house|united states|american|democratic presidential|republican presidential|nyc|new york city|california|texas|florida|wisconsin|federal reserve|fed governor)\b", text)
        ) or bool(re.search(r"\bUS\b|\bU\.S\.", f"{title or ''}"))
        political_outcome = bool(normalized_tags & {"politics", "elections", "primaries", "us-politics", "us-election", "us-presidential-election", "usa-election", "trump-presidency"}) or bool(
            re.search(r"\b(?:president|presidential|nominee|election|congress|senate|senator|governor|mayor|gubernatorial|white house|impeach|approval rating|shutdown|government|supreme court)\b", text)
        )
        overrides: list[tuple[str, str]] = []
        # A quoted asset-price outcome is classified by that outcome, not its
        # possible geopolitical driver; names alone are not enough.
        price_question = bool(re.search(r"\b(?:price|hit|reach|settle|close|above|below|higher|lower)\b", text))
        medal_outcome = bool(re.search(r"\b(?:gold|silver|bronze)\s+medals?\b", text))
        metal_proper_name = bool(re.search(r"\bsilver[- ]bulletin\b", text))
        military_target = bool(re.search(r"\b(?:terminal|refinery|island|pipeline|tanker)\b", text)) and bool(
            re.search(r"\b(?:hit|struck|attacked|attack|bomb|destroyed)\b", text)
        ) and not bool(re.search(r"\bprices?\b", text))
        mentions_outcome = bool(normalized_tags & {"mentions", "tweet-markets"}) or bool(
            re.search(r"\b(?:say|mention|mentions|tweet|tweets|post on x)\b", text)
        )
        if mentions_outcome:
            overrides.append(("meta_mentions", "outcome_text"))
        elif bool(normalized_tags & {"cricket", "rugby", "poker"}) or (
            bool(re.search(r"\b(?:cricket|rugby|poker|t20|icc|ipl)\b", text)) and "sports" in normalized_tags
        ):
            overrides.append(("sports_other", "outcome_text"))
        elif intraday:
            overrides.append(("crypto_short_horizon", "contract_form"))
        elif military_target:
            overrides.append(("politics_global", "outcome_text"))
        elif price_question and not (medal_outcome or metal_proper_name) and re.search(r"\b(?:crude oil|oil|gold|silver|natural gas)\b", text):
            overrides.append(("commodities", "outcome_text"))
        elif crypto and price_question:
            major = bool(normalized_tags & {"bitcoin", "ethereum", "solana", "xrp"}) or bool(re.search(r"\b(?:bitcoin|btc|ethereum|eth|solana|xrp)\b", text))
            overrides.append(("crypto_majors" if major else "crypto_alts", "outcome_text"))
        # Cross-border conflict questions concern geopolitics even when a US
        # office-holder/actor is mentioned. Elections retain geographic guards.
        elif re.search(r"\b(?:ceasefire|invasion|peace deal|strikes? iran|forces enter|invade|war with|military attack)\b", text) or (
            bool(normalized_tags & {"geopolitics", "middle-east"})
            and bool(re.search(r"\b(?:strike|strikes|attack|attacks|airstrike|bomb|military)\b", text))
        ):
            overrides.append(("politics_global", "outcome_text"))
        candidates: list[CategoryRule] = []
        bases: list[str] = []
        for rule in self.rules[:-1]:
            if rule.category_id == "crypto_short_horizon" and not intraday:
                continue
            if rule.category_id == "commodities" and (medal_outcome or metal_proper_name):
                continue
            if rule.category_id == "politics_us_local" and re.search(r"\b(?:fed|federal reserve)\b", text):
                continue
            if rule.category_id.startswith("politics_us_") and not (us_context and political_outcome):
                continue
            tag_match = bool(rule.tags.intersection(normalized_tags))
            text_match = any(pattern.search(text) for pattern in rule.patterns)
            if tag_match or text_match:
                candidates.append(rule)
                bases.append("tag" if tag_match else "text")
        if overrides:
            primary = by_id[overrides[0][0]]
            basis = "contract_form" if overrides[0][1] == "contract_form" else "outcome_rule"
            candidates = [primary] + [r for r in candidates if r.category_id != primary.category_id]
        else:
            primary = candidates[0] if candidates else self.rules[-1]
            basis = bases[0] if bases else "fallback"
            # Generic politics is considered only after specific subject rules.
            # Other/global includes political events with unspecified geography.
            if primary.category_id == "other" and (
                bool(normalized_tags & {"politics", "presidency", "president", "congress", "senate"})
                or political_outcome
            ):
                primary = by_id["politics_us_federal" if us_context and political_outcome else "politics_global"]
                basis = "political_fallback"
                candidates = [primary]
        return {
            "taxonomy_version": self.version,
            "category_id": primary.category_id,
            "category": primary.label,
            "legacy_category": primary.legacy,
            "classification_basis": basis,
            "all_category_ids": [rule.category_id for rule in candidates],
            "normalized_tags": sorted(normalized_tags),
        }
