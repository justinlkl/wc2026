"""Load and cache international match results."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from wc2026.config import (
    COMPETITIVE_TOURNAMENT_PATTERNS,
    DATA_DIR,
    MIN_DATE,
    RESULTS_URL,
)

logger = logging.getLogger(__name__)


def download_results(dest: Path | None = None, force: bool = False) -> Path:
    """Download results.csv from martj42/international_results."""
    dest = dest or DATA_DIR / "results.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        logger.info("Results already cached at %s", dest)
        return dest

    logger.info("Downloading international results from GitHub...")
    resp = requests.get(RESULTS_URL, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    logger.info("Saved %s (%d bytes)", dest, len(resp.content))
    return dest


def _is_competitive(tournament: str) -> bool:
    if pd.isna(tournament):
        return False
    t = str(tournament)
    return any(pat in t for pat in COMPETITIVE_TOURNAMENT_PATTERNS)


def load_results(
    path: Path | None = None,
    min_date: str = MIN_DATE,
    competitive_only: bool = True,
) -> pd.DataFrame:
    """Load results, filter to modern competitive window."""
    path = path or DATA_DIR / "results.csv"
    if not path.exists():
        download_results(path)

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "neutral" in df.columns:
        df["neutral"] = df["neutral"].map(
            {True: 1, False: 0, "True": 1, "False": 0}
        ).fillna(0).astype(int)
    else:
        df["neutral"] = 0

    df = df[df["date"] >= pd.Timestamp(min_date)]

    if competitive_only:
        mask = df["tournament"].apply(_is_competitive)
        df = df[mask].copy()

    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()

    # Drop incomplete rows
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    return df


def encode_result(home_score: int, away_score: int) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if home_score > away_score:
        return 0
    if home_score < away_score:
        return 2
    return 1
