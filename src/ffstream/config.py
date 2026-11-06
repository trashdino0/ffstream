"""Project paths and configuration loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
CONFIG_DIR = ROOT / "config"


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR, CONFIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_scoring(path: str | Path | None = None) -> dict[str, Any]:
    """Load scoring rules. Defaults to config/scoring.json."""
    scoring_path = Path(path) if path else CONFIG_DIR / "scoring.json"
    with open(scoring_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_scoring(scoring: dict[str, Any], path: str | Path | None = None) -> Path:
    scoring_path = Path(path) if path else CONFIG_DIR / "scoring.json"
    scoring_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scoring_path, "w", encoding="utf-8") as f:
        json.dump(scoring, f, indent=2)
    return scoring_path
