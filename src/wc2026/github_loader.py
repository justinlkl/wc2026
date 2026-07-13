"""Load WC 2026 fixtures from the GitHub dataset (mominullptr).

We replace Kaggle fixture sourcing with:
  https://github.com/mominullptr/FIFA-World-Cup-2026-Dataset

Return format (what `predict_world_cup.py` expects):

    {"fixture_id", "date", "round", "home_team", "away_team"}

Important: this loader only provides *fixtures*. All model features are computed
from historical martj42/international_results.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from wc2026.config import DATA_DIR
from wc2026.team_names import canonical_team, is_qualified_team

logger = logging.getLogger(__name__)

GITHUB_DATASET_MATCHES_CSV = (
    "https://raw.githubusercontent.com/mominullptr/FIFA-World-Cup-2026-Dataset/main/matches.csv"
)
GITHUB_DATASET_TEAMS_CSV = (
    "https://raw.githubusercontent.com/mominullptr/FIFA-World-Cup-2026-Dataset/main/teams.csv"
)
GITHUB_DATASET_MATCHES_DETAILED_CSV = (
    "https://raw.githubusercontent.com/mominullptr/FIFA-World-Cup-2026-Dataset/main/matches_detailed.csv"
)

GITHUB_CACHE_DIR = DATA_DIR / "github_wc2026"

_STAGE_LABELS = {
    1: "Group Stage",
    2: "Round of 16",
    3: "Quarter-finals",
    4: "Semi-finals",
    5: "Semi-finals",      # upstream labels as 3rd place but schedule shows semi-finals
    6: "Final",
    7: "TBD",
}


def _download_if_needed(url: str, dest: Path, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest

    import requests

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def _pick(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


def _load_tables(force_download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches_path = GITHUB_CACHE_DIR / "matches.csv"
    teams_path = GITHUB_CACHE_DIR / "teams.csv"

    _download_if_needed(GITHUB_DATASET_MATCHES_CSV, matches_path, force=force_download)
    _download_if_needed(GITHUB_DATASET_TEAMS_CSV, teams_path, force=force_download)

    return pd.read_csv(matches_path), pd.read_csv(teams_path)


def _team_lookup(teams_df: pd.DataFrame) -> dict[int, str]:
    teams_id_col = _pick(teams_df, ["team_id", "id", "teamId", "TeamId"])
    teams_name_col = _pick(teams_df, ["team_name", "name", "TeamName", "Team"])

    id_to_name: dict[int, str] = {}
    for _, row in teams_df.iterrows():
        if pd.isna(row[teams_id_col]) or pd.isna(row[teams_name_col]):
            continue
        try:
            team_id = int(row[teams_id_col])
        except Exception:
            continue
        id_to_name[team_id] = canonical_team(str(row[teams_name_col]))
    return id_to_name


def _standardise_matches(matches_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    match_id_col = _pick(matches_df, ["match_id", "fixture_id", "id", "MatchID"])
    date_col = _pick(matches_df, ["date", "match_date", "Date", "kickoff_time_utc", "kickoff"])
    stage_id_col = _pick(matches_df, ["stage_id", "stageId", "stageID", "StageId", "StageID"])
    home_id_col = _pick(matches_df, ["home_team_id", "homeId", "home_teamid", "homeTeamId"])
    away_id_col = _pick(matches_df, ["away_team_id", "awayId", "away_teamid", "awayTeamId"])

    id_to_name = _team_lookup(teams_df)
    df = matches_df.copy()
    df["home_team"] = df[home_id_col].apply(lambda x: id_to_name.get(int(x), "") if pd.notna(x) else "")
    df["away_team"] = df[away_id_col].apply(lambda x: id_to_name.get(int(x), "") if pd.notna(x) else "")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df[date_col] = df[date_col].dt.tz_localize(None)
    df["stage_id"] = pd.to_numeric(df[stage_id_col], errors="coerce")
    df["status"] = df.get("status", "")
    df["home_score"] = pd.to_numeric(df.get("home_score"), errors="coerce")
    df["away_score"] = pd.to_numeric(df.get("away_score"), errors="coerce")
    df["home_xg"] = pd.to_numeric(df.get("home_xg"), errors="coerce")
    df["away_xg"] = pd.to_numeric(df.get("away_xg"), errors="coerce")
    df["fixture_id"] = df[match_id_col]
    df["date"] = df[date_col]
    df["round"] = df["stage_id"].map(_STAGE_LABELS).fillna("")
    return df


def _matches_to_results(matches_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    df = _standardise_matches(matches_df, teams_df)
    completed = df[df["status"].astype(str).str.lower() == "completed"].copy()

    results = completed[
        [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "home_xg",
            "away_xg",
            "stage_id",
            "round",
        ]
    ].copy()
    results["tournament"] = "FIFA World Cup 2026"
    results["neutral"] = 1
    results["city"] = ""
    results["country"] = ""
    results["home_score"] = results["home_score"].astype("Int64")
    results["away_score"] = results["away_score"].astype("Int64")
    return results.sort_values("date").reset_index(drop=True)


def _matches_to_fixtures(
    matches_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    *,
    min_stage_id: int = 4,
) -> list[dict]:
    df = _standardise_matches(matches_df, teams_df)
    upcoming = df[
        (df["status"].astype(str).str.lower() == "scheduled")
        & (df["stage_id"].fillna(0) >= min_stage_id)
    ].copy()

    fixtures: list[dict] = []
    for _, row in upcoming.iterrows():
        home_team = canonical_team(str(row["home_team"]))
        away_team = canonical_team(str(row["away_team"]))
        if not (is_qualified_team(home_team) and is_qualified_team(away_team)):
            continue

        fixtures.append(
            {
                "fixture_id": row["fixture_id"],
                "date": pd.Timestamp(row["date"]),
                "round": str(row["round"]) if row["round"] else "",
                "home_team": home_team,
                "away_team": away_team,
            }
        )

    logger.info("Loaded %d upcoming fixtures from GitHub matches.csv", len(fixtures))
    return fixtures


def load_github_completed_results(force_download: bool = False) -> pd.DataFrame:
    """Load completed World Cup matches as a results.csv-style DataFrame."""
    matches_df, teams_df = _load_tables(force_download=force_download)
    return _matches_to_results(matches_df, teams_df)


def load_github_matches(round_of_16_only: bool = True, force_download: bool = False) -> list[dict]:
    """Load Round of 16+ fixtures from GitHub matches.csv.

    The dataset `matches.csv` uses team IDs; we join against `teams.csv`.
    """
    matches_df, teams_df = _load_tables(force_download=force_download)
    fixtures = _matches_to_fixtures(matches_df, teams_df, min_stage_id=2 if round_of_16_only else 1)
    if round_of_16_only:
        fixtures = [fx for fx in fixtures if str(fx.get("round", "")).lower() != "group stage"]
    return fixtures


def load_github_upcoming_fixtures(
    min_stage_id: int = 4,
    force_download: bool = False,
) -> list[dict]:
    """Load only upcoming knockout fixtures from GitHub matches.csv."""
    matches_df, teams_df = _load_tables(force_download=force_download)
    return _matches_to_fixtures(matches_df, teams_df, min_stage_id=min_stage_id)


def refresh_matches_detailed_csv(
    force_download: bool = False,
    dest_path: Path | None = None,
) -> Path:
    """Download the latest matches_detailed.csv into the specified path.

    Defaults to data/matches_detailed.csv so xg_features.py consumers pick it up.
    """
    dest_path = dest_path or (DATA_DIR / "matches_detailed.csv")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    _download_if_needed(GITHUB_DATASET_MATCHES_DETAILED_CSV, dest_path, force=force_download)
    logger.info("Refreshed matches_detailed.csv -> %s", dest_path)
    return dest_path


def load_github_matches_detailed(force_download: bool = False) -> pd.DataFrame:
    """Load per-match xG detail (home_xg, away_xg) from GitHub matches_detailed.csv.

    This file contains detailed match stats including xG for both teams.

    Returns:
        DataFrame with columns: date, home_team_name, away_team_name, home_xg, away_xg
        (plus any other columns in the source file).
    """
    dest = GITHUB_CACHE_DIR / "matches_detailed.csv"
    _download_if_needed(GITHUB_DATASET_MATCHES_DETAILED_CSV, dest, force=force_download)
    df = pd.read_csv(dest)

    # Standardize column names for downstream processing
    # The source file has various possible column names
    _col = lambda cols: next((c for c in cols if c in df.columns), None)

    # Map to consistent names
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df["date"] = df["date"].dt.tz_localize(None)

    logger.info("Loaded %d rows from GitHub matches_detailed.csv", len(df))
    return df

