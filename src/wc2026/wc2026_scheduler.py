"""Auto-update scheduler for WC 2026 predictions.

Runs as a background loop (default: every 3 hours). Each cycle:
- re-downloads upstream CSVs (GitHub dataset + international_results)
- regenerates predictions CSV via inference only (no retraining)

Usage:
  python -m wc2026.wc2026_scheduler          # runs indefinitely
  python -m wc2026.wc2026.wc2026_scheduler --once   # single run then exit
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from wc2026.config import DATA_DIR, OUTPUTS_DIR
from wc2026.data_loading import download_results, load_results
from wc2026.features import build_modeling_dataset
from wc2026.github_loader import (
    GITHUB_CACHE_DIR,
    GITHUB_DATASET_MATCHES_CSV,
    GITHUB_DATASET_MATCHES_DETAILED_CSV,
    _download_if_needed,
    refresh_matches_detailed_csv,
)
from wc2026.model_train import train_models
from wc2026.predict_world_cup import predict_fixtures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("wc2026.scheduler")

POLL_INTERVAL_SECONDS = 3 * 60 * 60  # 3 hours


_STAGE_LABELS = {
    1: "Group Stage",
    2: "Round of 16",
    3: "Quarter-finals",
    4: "Semi-finals",
    5: "Semi-finals",      # upstream labels these as 3rd place but schedule shows semi-finals
    6: "Final",
    7: "TBD",
}


def _rebuild_fixtures_csv_from_github_matches(
    *,
    force_download: bool,
    min_stage_id: int = 4,
    fixtures_path: Path | None = None,
) -> Path:
    """Create data/fixtures.csv from GitHub dataset matches.csv.

    The upstream dataset contains both completed and scheduled matches.
    We output fixtures only for matches with status == scheduled and stage >= min_stage_id
    (consistent with predict_world_cup's inference feature pipeline).
    """

    fixtures_path = fixtures_path or (DATA_DIR / "fixtures.csv")

    GITHUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    matches_path = GITHUB_CACHE_DIR / "matches.csv"

    # Download raw matches CSV
    _download_if_needed(GITHUB_DATASET_MATCHES_CSV, matches_path, force=force_download)

    matches_df = pd.read_csv(matches_path)

    # Heuristics for schema
    def pick(cols: list[str]) -> str:
        for c in cols:
            if c in matches_df.columns:
                return c
        raise KeyError(f"None of {cols} found in matches.csv columns: {list(matches_df.columns)}")

    match_id_col = pick(["match_id", "fixture_id", "id", "MatchID"])
    date_col = pick(["date", "match_date", "Date", "kickoff_time_utc", "kickoff"])
    stage_id_col = pick(["stage_id", "stageId", "stageID", "StageId", "StageID"])
    home_id_col = pick(["home_team_id", "homeId", "home_teamid", "homeTeamId"])
    away_id_col = pick(["away_team_id", "awayId", "away_teamid", "awayTeamId"])

    status_col = "status" if "status" in matches_df.columns else None

    # We also need teams mapping from teams.csv
    teams_path = GITHUB_CACHE_DIR / "teams.csv"
    from wc2026.github_loader import GITHUB_DATASET_TEAMS_CSV

    _download_if_needed(GITHUB_DATASET_TEAMS_CSV, teams_path, force=force_download)
    teams_df = pd.read_csv(teams_path)

    team_id_col = None
    for c in ["team_id", "id", "teamId", "TeamId"]:
        if c in teams_df.columns:
            team_id_col = c
            break
    if team_id_col is None:
        raise KeyError(f"No team id column found in teams.csv: {list(teams_df.columns)}")

    team_name_col = None
    for c in ["team_name", "name", "TeamName", "Team"]:
        if c in teams_df.columns:
            team_name_col = c
            break
    if team_name_col is None:
        raise KeyError(f"No team name column found in teams.csv: {list(teams_df.columns)}")

    id_to_name: dict[int, str] = {}
    from wc2026.team_names import canonical_team

    for _, row in teams_df.iterrows():
        if pd.isna(row[team_id_col]) or pd.isna(row[team_name_col]):
            continue
        try:
            tid = int(row[team_id_col])
        except Exception:
            continue
        id_to_name[tid] = canonical_team(str(row[team_name_col]))

    # Standardize
    df = matches_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None)
    df[stage_id_col] = pd.to_numeric(df[stage_id_col], errors="coerce")

    if status_col:
        scheduled = df[status_col].astype(str).str.lower() == "scheduled"
    else:
        # If no status column exists, fall back to stage-based filtering only.
        scheduled = pd.Series([True] * len(df), index=df.index)

    df = df[scheduled & (df[stage_id_col].fillna(0) >= min_stage_id)].copy()

    df["home_team"] = df[home_id_col].apply(lambda x: id_to_name.get(int(x), "") if pd.notna(x) else "")
    df["away_team"] = df[away_id_col].apply(lambda x: id_to_name.get(int(x), "") if pd.notna(x) else "")

    df = df[df["home_team"] != ""]
    df = df[df["away_team"] != ""]

    df["stage_id_int"] = df[stage_id_col].astype(int)
    df["round"] = df["stage_id_int"].map(_STAGE_LABELS).fillna("TBD")
    fixtures_out = pd.DataFrame(
        {
            "fixture_id": df[match_id_col],
            "date": df[date_col].dt.date.astype(str),
            "round": df["round"],
            "home_team": df["home_team"],
            "away_team": df["away_team"],
        }
    )

    fixtures_out["date"] = pd.to_datetime(fixtures_out["date"]).dt.date

    fixtures_path.parent.mkdir(parents=True, exist_ok=True)
    fixtures_out.to_csv(fixtures_path, index=False)

    logger.info(
        "Rebuilt fixtures CSV: %s (rows=%d, min_stage_id=%d)",
        fixtures_path,
        len(fixtures_out),
        min_stage_id,
    )
    return fixtures_path


def run_update(*, force_download: bool, fixtures_min_stage_id: int) -> None:
    logger.info("=== WC 2026 update cycle starting at %s ===", datetime.now(timezone.utc).isoformat())

    # 1) Refresh international_results backbone
    download_results(force=force_download)

    # 2) Refresh GitHub dataset tables and rebuild fixtures.csv from matches.csv
    _rebuild_fixtures_csv_from_github_matches(
        force_download=force_download,
        min_stage_id=fixtures_min_stage_id,
        fixtures_path=DATA_DIR / "fixtures.csv",
    )

    # 3) Refresh matches_detailed.csv (xG source) daily and retrain
    #    so new rows update Elo + xG features before inference
    refresh_matches_detailed_csv(force_download=force_download, dest_path=DATA_DIR / "matches_detailed.csv")

    results = load_results()
    matches = build_modeling_dataset(results, include_xg=True)
    train_models(matches)

    # 4) Run inference to produce updated predictions
    out_path = OUTPUTS_DIR / "wc2026_predictions.csv"
    predict_fixtures(
        fixtures_csv=DATA_DIR / "fixtures.csv",
        output_path=out_path,
        print_top_outcome=False,
        source="csv",
        force_kaggle_download=False,
    )

    logger.info("Update complete: predictions saved to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="WC 2026 prediction auto-updater")
    parser.add_argument("--once", action="store_true", help="Run one update cycle then exit")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL_SECONDS, help="Polling interval in seconds")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of upstream CSVs (GitHub dataset + international_results)",
    )
    parser.add_argument(
        "--fixtures-min-stage-id",
        type=int,
        default=4,
        help="Only scheduled fixtures with stage_id >= this value are written to fixtures.csv",
    )
    args = parser.parse_args()

    if args.once:
        run_update(force_download=args.force_download, fixtures_min_stage_id=args.fixtures_min_stage_id)
        return

    logger.info("Scheduler started. Polling every %d seconds.", args.interval)
    while True:
        try:
            run_update(force_download=args.force_download, fixtures_min_stage_id=args.fixtures_min_stage_id)
        except Exception as exc:
            logger.error("Update cycle failed: %s", exc, exc_info=True)
        logger.info("Sleeping %d s until next check...", args.interval)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

