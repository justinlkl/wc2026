"""Load WC 2026 fixtures from Kaggle datasets.

The loader normalizes Kaggle fixture exports into the same list[dict] shape used
by the prediction pipeline and filters them down to matches involving the 48
qualified teams.

Notes about Kaggle downloading:
- Prefer `kagglehub` if available (does not require `kaggle.json`).
- Fall back to Kaggle CLI **only if** the user has authenticated via
  `~/.kaggle/kaggle.json` (or env-based auth supported by their CLI setup).
- We intentionally avoid `python -m kaggle ...` because some kaggle package
  versions do not provide an importable `kaggle.__main__`.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from wc2026.config import DATA_DIR
from wc2026.team_names import canonical_team, is_qualified_team

logger = logging.getLogger(__name__)

KAGGLE_DATASET_SCHEDULE = "areezvisram12/fifa-world-cup-2026-match-data-unofficial"

KAGGLE_CACHE_DIR = DATA_DIR / "kaggle_wc2026"

_DATE_COLS = ["date", "Date", "match_date"]
_HOME_COLS = ["home_team", "Home Team", "home", "team1"]
_AWAY_COLS = ["away_team", "Away Team", "away", "team2"]
_ROUND_COLS = ["stage", "round", "Round", "Stage", "match_type"]
_ID_COLS = ["match_id", "fixture_id", "id", "MatchID"]


def _col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first candidate column name that exists in df."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


def _download_kaggle(dataset_slug: str, dest: Path, force: bool = False) -> Path:
    """Download a Kaggle dataset into dest.

    Tries, in order:
    1) `kagglehub` (preferred)
    2) Kaggle CLI executable `kaggle` (requires authentication)
    """
    dest.mkdir(parents=True, exist_ok=True)
    if not force and any(dest.glob("*.csv")):
        logger.info("Kaggle cache hit: %s", dest)
        return dest

    # 1) kagglehub
    try:
        import kagglehub  # type: ignore

        path = kagglehub.dataset_download(
            dataset_slug,
            path=str(dest),
            force_download=force,
        )
        logger.info("kagglehub downloaded to %s", path)
        return Path(path)
    except Exception:
        # If kagglehub isn't installed or fails, we fall back to CLI.
        pass

    # 2) Kaggle CLI
    kaggle_exe = shutil.which("kaggle")
    if not kaggle_exe:
        raise RuntimeError(
            "Kaggle CLI executable not found on PATH. "
            "Install with: pip install kaggle\n"
            "Then re-run."
        )

    cmd = [
        kaggle_exe,
        "datasets",
        "download",
        "-d",
        dataset_slug,
        "-p",
        str(dest),
        "--unzip",
    ]
    if force:
        cmd.append("--force")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle CLI failed:\n{result.stderr}\n\n"
            "Fix by authenticating the Kaggle CLI (creates ~/.kaggle/kaggle.json).\n"
            "Typical setup: `kaggle config set -n <username> -p <api_key>`.\n"
        )

    logger.info("kaggle CLI downloaded to %s", dest)
    return dest


def _find_matches_csv(folder: Path) -> Path:
    """Find the most likely matches CSV inside a downloaded Kaggle folder."""
    priority = ["matches.csv", "schedule.csv", "fixtures.csv", "wc_2026_matches.csv"]
    for name in priority:
        candidate = folder / name
        if candidate.exists():
            return candidate

    for candidate in sorted(folder.glob("**/*.csv")):
        if not any(skip in candidate.stem.lower() for skip in ("team", "venue", "city", "player")):
            return candidate

    raise FileNotFoundError(f"No matches CSV found in {folder}")


def _normalise_fixtures(df: pd.DataFrame) -> list[dict]:
    """Convert a raw Kaggle fixtures DataFrame into the standard list[dict]."""
    date_col = _col(df, _DATE_COLS)
    home_col = _col(df, _HOME_COLS)
    away_col = _col(df, _AWAY_COLS)
    round_col = _col(df, _ROUND_COLS) if any(col in df.columns for col in _ROUND_COLS) else None
    id_col = _col(df, _ID_COLS) if any(col in df.columns for col in _ID_COLS) else None

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")

    fixtures: list[dict] = []
    for index, row in df.iterrows():
        home_team = canonical_team(str(row[home_col]))
        away_team = canonical_team(str(row[away_col]))
        if not (is_qualified_team(home_team) and is_qualified_team(away_team)):
            continue

        match_date = row[date_col]
        if pd.isna(match_date):
            date_value = pd.Timestamp("2026-06-11")
        else:
            # Make naive timestamp for the rest of the pipeline.
            date_value = pd.Timestamp(match_date).tz_localize(None)

        fixtures.append(
            {
                "fixture_id": row[id_col] if id_col else index,
                "date": date_value,
                "round": str(row[round_col]) if round_col and not pd.isna(row[round_col]) else "",
                "home_team": home_team,
                "away_team": away_team,
            }
        )

    return fixtures


def download_kaggle_fixtures(force: bool = False) -> Path:
    """Download the WC 2026 schedule dataset and return the cache folder."""
    return _download_kaggle(KAGGLE_DATASET_SCHEDULE, KAGGLE_CACHE_DIR, force=force)


def load_kaggle_fixtures(
    path: Path | None = None,
    force_download: bool = False,
) -> list[dict]:
    """Load WC 2026 fixtures from Kaggle and return normalized fixture dicts."""
    if path is not None:
        csv_path = path
    else:
        folder = download_kaggle_fixtures(force=force_download)
        csv_path = _find_matches_csv(folder)
        logger.info("Reading Kaggle fixtures from %s", csv_path)

    df = pd.read_csv(csv_path)
    fixtures = _normalise_fixtures(df)
    logger.info("Loaded %d WC 2026 fixtures from Kaggle", len(fixtures))
    return fixtures

