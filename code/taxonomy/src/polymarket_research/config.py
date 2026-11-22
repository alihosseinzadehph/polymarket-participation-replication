from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


@dataclass(frozen=True)
class AnalysisConfig:
    path: Path
    values: dict[str, Any]

    @property
    def repo_root(self) -> Path:
        return REPO_ROOT

    @property
    def derived_dir(self) -> Path:
        return self.repo_root / "data" / "derived"

    @property
    def tables_dir(self) -> Path:
        return self.repo_root / "results" / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.repo_root / "results" / "figures"

    @property
    def logs_dir(self) -> Path:
        return self.repo_root / "results" / "logs"

    @property
    def reports_dir(self) -> Path:
        return self.repo_root / "reports" / "generated"

    def source_path(self, key: str) -> Path:
        return Path(self.values["paths"][key])

    def trade_globs(self) -> list[str]:
        root = self.source_path("trade_root").as_posix()
        return [
            f"{root}/polymarket_v1/daily_aligned/*.parquet",
            f"{root}/polymarket_v1/daily_aligned_multi/*.parquet",
            f"{root}/polymarket_v2/daily_aligned/*.parquet",
            f"{root}/polymarket_v2/daily_aligned_multi/*.parquet",
        ]

    def ensure_output_dirs(self) -> None:
        for path in (
            self.derived_dir,
            self.tables_dir,
            self.figures_dir,
            self.logs_dir,
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> AnalysisConfig:
    config_path = Path(path) if path else REPO_ROOT / "config" / "analysis.yaml"
    config_path = config_path.resolve()
    config = AnalysisConfig(config_path, load_yaml(config_path))
    config.ensure_output_dirs()
    return config


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
