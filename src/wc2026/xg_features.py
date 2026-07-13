"""StatsBomb open-data xG aggregation and rolling team features."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import requests

from wc2026.config import DATA_DIR, STATSBOMB_WC_COMPETITIONS, XG_WINDOWS
from wc2026.team_names import canonical_team
from wc2026.wc2026_xg_source import load_wc2026_match_xg_long


logger = logging.getLogger(__name__)

_SHOOTOUT_PERIOD = 5
_RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master"


def statsbomb_data_dir() -> Path:
    return DATA_DIR / "statsbomb" / "open-data"


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def download_statsbomb_wc_data(force: bool = False) -> Path:
    """
    Download only World Cup 2018/2022 matches and event files (not the full repo).
    """
    root = statsbomb_data_dir()
    if force:
        import shutil

        if root.exists():
            shutil.rmtree(root)

    matches_root = root / "data" / "matches"
    events_root = root / "data" / "events"

    for comp_id in STATSBOMB_WC_COMPETITIONS:
        comp_dir = matches_root / str(comp_id)
        if not comp_dir.exists():
            season_files = []
            # Discover season JSON files via GitHub API (lightweight listing)
            api_url = (
                f"https://api.github.com/repos/statsbomb/open-data/contents/"
                f"data/matches/{comp_id}"
            )
            resp = requests.get(api_url, timeout=30)
            if resp.status_code == 200:
                season_files = [item["name"] for item in resp.json()]
            else:
                # Fallback season ids commonly used in open-data
                season_files = ["1.json", "3.json", "27.json"]

            for season_file in season_files:
                url = f"{_RAW_BASE}/data/matches/{comp_id}/{season_file}"
                dest = comp_dir / season_file
                try:
                    _download_file(url, dest)
                except requests.HTTPError:
                    logger.debug("No match file at %s", url)

        match_ids: set[int] = set()
        for match_file in comp_dir.glob("*.json"):
            with open(match_file) as f:
                batch = json.load(f)
            for match in batch:
                match_ids.add(match["match_id"])

        logger.info(
            "Competition %d: downloading %d event files",
            comp_id,
            len(match_ids),
        )
        for mid in match_ids:
            url = f"{_RAW_BASE}/data/events/{mid}.json"
            dest = events_root / f"{mid}.json"
            try:
                _download_file(url, dest)
            except requests.HTTPError:
                logger.warning("Missing events for match %s", mid)

    return root


def clone_statsbomb_data(force: bool = False) -> Path:
    """Download StatsBomb WC data (targeted, not full git clone)."""
    return download_statsbomb_wc_data(force=force)


def team_xg(events: list, team: str) -> tuple[float, float]:
    """Sum shot xG for and against a given team from StatsBomb events."""
    xg_for = 0.0
    xg_against = 0.0
    for ev in events:
        if ev.get("type", {}).get("name") != "Shot":
            continue
        if ev.get("period") == _SHOOTOUT_PERIOD:
            continue
        shot = ev.get("shot", {})
        xg = shot.get("statsbomb_xg")
        if xg is None:
            continue
        ev_team = ev.get("team", {}).get("name", "")
        if ev_team == team:
            xg_for += float(xg)
        else:
            xg_against += float(xg)
    return xg_for, xg_against


def _load_matches(competition_id: int) -> list[dict]:
    root = statsbomb_data_dir()
    matches_path = root / "data" / "matches" / str(competition_id)
    if not matches_path.exists():
        download_statsbomb_wc_data()
    matches: list[dict] = []
    for fp in matches_path.glob("*.json"):
        with open(fp) as f:
            batch = json.load(f)
        matches.extend(batch)
    return matches


def build_match_xg_table() -> pd.DataFrame:
    """Build team-match xG rows from StatsBomb (2018/2022) and WC 2026 CSV."""

    rows: list[dict] = []

    # 1) StatsBomb World Cup events (2018/2022)
    root = statsbomb_data_dir()

    if not (root / "data" / "matches").exists():
        download_statsbomb_wc_data()

    for comp_id, year in STATSBOMB_WC_COMPETITIONS.items():
        matches = _load_matches(comp_id)
        for match in matches:
            match_id = match["match_id"]
            date = pd.Timestamp(match["match_date"])
            home_raw = match["home_team"]["home_team_name"]
            away_raw = match["away_team"]["away_team_name"]
            # Defensive: skip if StatsBomb changed format and field is not a string
            if not isinstance(home_raw, str) or not isinstance(away_raw, str):
                logger.warning(
                    "StatsBomb match %d: non-string team name skipped (home=%r away=%r)",
                    match_id,
                    home_raw,
                    away_raw,
                )
                continue
            home = canonical_team(home_raw)
            away = canonical_team(away_raw)

            events_path = root / "data" / "events" / f"{match_id}.json"
            if not events_path.exists():
                logger.warning("Missing events for match %s", match_id)
                continue

            with open(events_path) as f:
                events = json.load(f)

            hxg, hxg_a = team_xg(events, home)
            axg, axg_a = team_xg(events, away)

            rows.append(
                {
                    "date": date,
                    "team": home,
                    "opponent": away,
                    "xg_for": hxg,
                    "xg_against": hxg_a,
                    "is_home": 1,
                    "tournament": f"FIFA World Cup {year}",
                }
            )
            rows.append(
                {
                    "date": date,
                    "team": away,
                    "opponent": home,
                    "xg_for": axg,
                    "xg_against": axg_a,
                    "is_home": 0,
                    "tournament": f"FIFA World Cup {year}",
                }
            )

    # 2) WC 2026 xG from matches_detailed.csv
    wc2026_long = load_wc2026_match_xg_long()
    if not wc2026_long.empty:
        rows.extend(wc2026_long.to_dict(orient="records"))

    if not rows:
        return pd.DataFrame(
            columns=["date", "team", "opponent", "xg_for", "xg_against", "is_home"]
        )

    df = pd.DataFrame(rows).sort_values(["team", "date"]).reset_index(drop=True)
    return df





def rolling_xg_features(xg_long: pd.DataFrame) -> pd.DataFrame:
    """
    Per team-match row, compute rolling xG averages (excluding current match).
    """
    if xg_long.empty:
        return xg_long

    records: list[dict] = []
    for team, grp in xg_long.groupby("team", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        for i, row in grp.iterrows():
            past = grp.iloc[:i]
            feat: dict = {
                "date": row["date"],
                "team": team,
                "opponent": row["opponent"],
                "is_home": row["is_home"],
            }
            for w in XG_WINDOWS:
                tail = past.tail(w)
                if len(tail) == 0:
                    feat[f"xg_for_avg{w}"] = None
                    feat[f"xg_against_avg{w}"] = None
                else:
                    feat[f"xg_for_avg{w}"] = tail["xg_for"].mean()
                    feat[f"xg_against_avg{w}"] = tail["xg_against"].mean()
            records.append(feat)

    return pd.DataFrame(records)


def lookup_xg_features(
    xg_rolling: pd.DataFrame,
    team: str,
    as_of: pd.Timestamp,
) -> dict[str, float | None]:
    """Latest rolling xG features for a team strictly before as_of."""
    if xg_rolling.empty:
        return {f"xg_for_avg{w}": None for w in XG_WINDOWS} | {
            f"xg_against_avg{w}": None for w in XG_WINDOWS
        }

    subset = xg_rolling[(xg_rolling["team"] == team) & (xg_rolling["date"] <= as_of)]

    if subset.empty:
        return {f"xg_for_avg{w}": None for w in XG_WINDOWS} | {
            f"xg_against_avg{w}": None for w in XG_WINDOWS
        }

    latest = subset.sort_values("date").iloc[-1]
    out: dict[str, float | None] = {}
    for w in XG_WINDOWS:
        out[f"xg_for_avg{w}"] = latest.get(f"xg_for_avg{w}")
        out[f"xg_against_avg{w}"] = latest.get(f"xg_against_avg{w}")
    return out
