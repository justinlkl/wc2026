"""WC 2026 xG source wiring.

This module converts `data/matches_detailed.csv` (per-match xG for the
entire WC 2026 tournament so far) into the same long format used by
StatsBomb xG features.

Expected input columns (best-effort, with fallbacks):
- date (or match_date)
- home_team, away_team (or homeTeam/awayTeam)
- home_xg_for, away_xg_for (or homeXgFor/awayXgFor)

and optionally:
- home_xg_against, away_xg_against

If against is not present, it is derived from the opponent's `xg_for`.

The output long format matches:
  date, team, opponent, xg_for, xg_against, is_home, tournament
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from wc2026.config import DATA_DIR, WC_SEASON
from wc2026.team_names import canonical_team

logger = logging.getLogger(__name__)


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def load_wc2026_match_xg_long() -> pd.DataFrame:
    """Load WC 2026 match xG as long format."""

    path = DATA_DIR / "matches_detailed.csv"
    if not path.exists():
        logger.warning("WC 2026 xG file not found: %s", path)
        return pd.DataFrame(
            columns=["date", "team", "opponent", "xg_for", "xg_against", "is_home", "tournament"]
        )

    df = pd.read_csv(path)

    date_col = _first_existing_col(df, ["date", "match_date", "kickoff_time"])
    home_col = _first_existing_col(df, ["home_team", "homeTeam", "HomeTeam", "home_team_name"])
    away_col = _first_existing_col(df, ["away_team", "awayTeam", "AwayTeam", "away_team_name"])


    # This dataset uses `home_xg` / `away_xg`.
    home_xg_for_col = _first_existing_col(
        df,
        [
            "home_xg",
            "homeXg",
            "home_xg_for",
            "homeXgFor",
        ],
    )
    away_xg_for_col = _first_existing_col(
        df,
        [
            "away_xg",
            "awayXg",
            "away_xg_for",
            "awayXgFor",
        ],
    )




    if not all([date_col, home_col, away_col, home_xg_for_col, away_xg_for_col]):
        logger.warning(
            "matches_detailed.csv missing required columns. Found columns=%s",
            list(df.columns),
        )
        return pd.DataFrame(
            columns=["date", "team", "opponent", "xg_for", "xg_against", "is_home", "tournament"]
        )

    df[date_col] = _to_timestamp(df[date_col])
    df = df.dropna(subset=[date_col])

    # Canonicalize team names
    df[home_col] = df[home_col].astype(str).map(canonical_team)
    df[away_col] = df[away_col].astype(str).map(canonical_team)

    df[home_xg_for_col] = pd.to_numeric(df[home_xg_for_col], errors="coerce")
    df[away_xg_for_col] = pd.to_numeric(df[away_xg_for_col], errors="coerce")

    # Optional against columns
    # This dataset only provides per-side xG for; derive xG-against.
    # home_xg_against = away_xg
    # away_xg_against = home_xg
    df["_home_xg_against"] = pd.to_numeric(df[away_xg_for_col], errors="coerce")
    df["_away_xg_against"] = pd.to_numeric(df[home_xg_for_col], errors="coerce")
    home_xg_against_col = "_home_xg_against"
    away_xg_against_col = "_away_xg_against"



    rows: list[dict] = []
    # Use iloc-based access to avoid `itertuples` name mangling for columns
    # starting with `_`.
    for _, row in df.iterrows():
        date = row[date_col]
        home = row[home_col]
        away = row[away_col]
        hxg_for = float(row[home_xg_for_col])
        hxg_against = float(row[home_xg_against_col])
        axg_for = float(row[away_xg_for_col])
        axg_against = float(row[away_xg_against_col])




        rows.append(
            {
                "date": pd.Timestamp(date),
                "team": home,
                "opponent": away,
                "xg_for": hxg_for,
                "xg_against": hxg_against,
                "is_home": 1,
                "tournament": f"FIFA World Cup {WC_SEASON}",
            }
        )
        rows.append(
            {
                "date": pd.Timestamp(date),
                "team": away,
                "opponent": home,
                "xg_for": axg_for,
                "xg_against": axg_against,
                "is_home": 0,
                "tournament": f"FIFA World Cup {WC_SEASON}",
            }
        )

    out = pd.DataFrame(rows).sort_values(["team", "date"]).reset_index(drop=True)
    logger.info("Loaded WC 2026 match xG long rows=%d from %s", len(out), path)
    return out

