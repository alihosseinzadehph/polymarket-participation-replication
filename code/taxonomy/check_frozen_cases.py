"""Recheck 20 retained event examples without downloading or relabelling data.

This regression check is not an independent annotation-accuracy assessment.
It reads the frozen classifier, rule file, and already reviewed event metadata.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from polymarket_research.taxonomy import Taxonomy


def main() -> None:
    taxonomy = Taxonomy(ROOT / "config" / "taxonomy.yaml")
    record = json.loads(
        (ROOT / "tests" / "population_cases_verified_results.json").read_text(
            encoding="utf-8"
        )
    )
    failures = []
    for case in record["cases"]:
        result = taxonomy.classify(
            case["title"], case["slug"], case["original_tags"]
        )
        if result["category_id"] not in case["acceptable_categories"]:
            failures.append(
                {
                    "event_id": case["event_id"],
                    "expected": case["acceptable_categories"],
                    "actual": result["category_id"],
                }
            )
    print(
        json.dumps(
            {
                "taxonomy_version": taxonomy.version,
                "cases": len(record["cases"]),
                "passed": len(record["cases"]) - len(failures),
                "failures": failures,
                "scope": "Retained targeted regression cases; not independent human validation.",
            },
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
