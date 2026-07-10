"""Form, H2H, tournament context, and match-level feature matrix."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from wc2026.config import (
    DEFAULT_TOURNAMENT_WEIGHT,
    FEATURE_COLS,
    FORM_WINDOWS,
    IN_TOURNAMENT_FORM_WINDOWS,
    TOURNAMENT_WEIGHTS,
    WC2026_START,
    XG_WINDOWS,
)
from wc2026.data_loading import encode_result
from wc2026.elo import compute_elo
from wc2026.xg_features import build_match_xg_table, lookup_xg_features, rolling_xg_features

logger = logging.getLogger(__name__)


def _tournament_weight(tournament: str) -> float:
    if pd.isna(tournament):
        return DEFAULT_TOURNAMENT_WEIGHT
    t = str(tournament)
    for key, weight in TOURNAMENT_WEIGHTS.items():
        if key in t:
            return weight
    return DEFAULT_TOURNAMENT_WEIGHT


def _is_world_cup(tournament: str) -> int:
    if pd.isna(tournament):
        return 0
    t = str(tournament)
    return 1 if "FIFA World Cup" in t and "qualification" not in t.lower() else 0


def _team_result_points(gf: int, ga: int) -> tuple[float, float]:
    """Win = 1, draw = 0.5, loss = 0; gd = gf - ga."""
    gd = gf - ga
    if gf > ga:
        return 1.0, gd
    if gf < ga:
        return 0.0, gd
    return 0.5, gd


def build_team_long(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, match) with result, goals, opponent."""
    home = results.assign(
        team=results["home_team"],
        opponent=results["away_team"],
        gf=results["home_score"],
        ga=results["away_score"],
        is_home=1,
    )
    away = results.assign(
        team=results["away_team"],
        opponent=results["home_team"],
        gf=results["away_score"],
        ga=results["home_score"],
        is_home=0,
    )
    cols = ["date", "team", "opponent", "gf", "ga", "is_home", "tournament", "neutral"]
    long = pd.concat([home[cols], away[cols]], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    pts, gd = zip(*[_team_result_points(int(r.gf), int(r.ga)) for r in long.itertuples()])
    long["points"] = pts
    long["gd"] = gd
    return long


def _rolling_form(team_long: pd.DataFrame) -> pd.DataFrame:
    """Rolling win rate, goal diff, gf/ga averages before each team-match."""
    records: list[dict] = []

    for team, grp in team_long.groupby("team", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        for i, row in grp.iterrows():
            past = grp.iloc[:i]
            feat: dict = {
                "date": row["date"],
                "team": team,
                "opponent": row["opponent"],
                "is_home": row["is_home"],
            }
            if len(past) > 0:
                feat["rest_days"] = (row["date"] - past.iloc[-1]["date"]).days
            else:
                feat["rest_days"] = None

            for w in FORM_WINDOWS:
                tail = past.tail(w)
                if len(tail) == 0:
                    feat[f"win{w}"] = None
                    feat[f"gd{w}"] = None
                    feat[f"gf{w}"] = None
                    feat[f"ga{w}"] = None
                else:
                    feat[f"win{w}"] = tail["points"].mean()
                    feat[f"gd{w}"] = tail["gd"].mean()
                    feat[f"gf{w}"] = tail["gf"].mean()
                    feat[f"ga{w}"] = tail["ga"].mean()
            records.append(feat)

    return pd.DataFrame(records)


def _lookup_form(
    form_df: pd.DataFrame,
    team: str,
    as_of: pd.Timestamp,
) -> dict:
    defaults = {
        "rest_days": None,
        **{f"win{w}": None for w in FORM_WINDOWS},
        **{f"gd{w}": None for w in FORM_WINDOWS},
        **{f"gf{w}": None for w in FORM_WINDOWS},
        **{f"ga{w}": None for w in FORM_WINDOWS},
    }
    subset = form_df[(form_df["team"] == team) & (form_df["date"] < as_of)]
    if subset.empty:
        return defaults
    latest = subset.sort_values("date").iloc[-1]
    out = defaults.copy()
    for k in out:
        if k in latest.index:
            out[k] = latest[k]
    return out


def _rolling_form_scoped(
    team_long: pd.DataFrame,
    start_date: str,
    windows: tuple[int, ...],
) -> pd.DataFrame:
    """
    Rolling win rate / goal diff limited to matches on/after start_date.

    Output columns are prefixed with `inwc_` later by caller logic.
    """
    scoped = team_long[team_long["date"] >= pd.Timestamp(start_date)].copy()
    records: list[dict] = []

    for team, grp in scoped.groupby("team", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        for i, row in grp.iterrows():
            past = grp.iloc[:i]
            feat: dict = {
                "date": row["date"],
                "team": team,
                "opponent": row["opponent"],
                "is_home": row["is_home"],
            }
            if len(past) > 0:
                feat["rest_days"] = (row["date"] - past.iloc[-1]["date"]).days
            else:
                feat["rest_days"] = None

            for w in windows:
                tail = past.tail(w)
                if len(tail) == 0:
                    feat[f"win{w}"] = None
                    feat[f"gd{w}"] = None
                else:
                    feat[f"win{w}"] = tail["points"].mean()
                    feat[f"gd{w}"] = tail["gd"].mean()
            records.append(feat)

    return pd.DataFrame(records)


def _lookup_inwc_form(
    form_df: pd.DataFrame,
    team: str,
    as_of: pd.Timestamp,
) -> dict:
    defaults = {
        "rest_days": None,
        **{f"win{w}": None for w in IN_TOURNAMENT_FORM_WINDOWS},
        **{f"gd{w}": None for w in IN_TOURNAMENT_FORM_WINDOWS},
    }
    subset = form_df[(form_df["team"] == team) & (form_df["date"] < as_of)]
    if subset.empty:
        return defaults
    latest = subset.sort_values("date").iloc[-1]
    out = defaults.copy()
    for k in out:
        if k in latest.index:
            out[k] = latest[k]
    return out


def _pair_key(team_a: str, team_b: str) -> tuple[str, str]:
    return tuple(sorted([team_a, team_b]))


def _h2h_features(results: pd.DataFrame) -> pd.DataFrame:
    """Head-to-head stats from home team's perspective before each match."""
    pair_history: dict[tuple[str, str], list[tuple[float, float]]] = {}
    records: list[dict] = []

    for i, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        key = _pair_key(home, away)
        ref = key[0]
        hist = pair_history.get(key, [])

        if not hist:
            records.append(
                {
                    "index": i,
                    "h2h_n": 0,
                    "h2h_home_winrate": None,
                    "h2h_home_gd": None,
                }
            )
        else:
            if home == ref:
                pts = [p for p, _ in hist]
                gds = [g for _, g in hist]
            else:
                pts = [1.0 - p if p in (0.0, 1.0) else 0.5 for p, _ in hist]
                gds = [-g for _, g in hist]
            records.append(
                {
                    "index": i,
                    "h2h_n": len(hist),
                    "h2h_home_winrate": float(np.mean(pts)),
                    "h2h_home_gd": float(np.mean(gds)),
                }
            )

        if home == ref:
            pts, gd = _team_result_points(int(row["home_score"]), int(row["away_score"]))
        else:
            pts, gd = _team_result_points(int(row["away_score"]), int(row["home_score"]))
            if pts == 0.0:
                pts = 1.0
            elif pts == 1.0:
                pts = 0.0
            gd = -gd

        pair_history.setdefault(key, []).append((pts, gd))

    return pd.DataFrame(records).set_index("index")


def _lookup_h2h(
    results: pd.DataFrame,
    home: str,
    away: str,
    as_of: pd.Timestamp,
) -> dict:
    past = results[results["date"] < as_of]
    key = _pair_key(home, away)
    ref = key[0]

    hist_pts: list[float] = []
    hist_gds: list[float] = []
    for _, m in past.iterrows():
        m_home = m["home_team"]
        m_away = m["away_team"]
        if _pair_key(m_home, m_away) != key:
            continue
        if m_home == ref:
            pts, gd = _team_result_points(int(m["home_score"]), int(m["away_score"]))
        else:
            pts, gd = _team_result_points(int(m["away_score"]), int(m["home_score"]))
            if pts == 0.0:
                pts = 1.0
            elif pts == 1.0:
                pts = 0.0
            gd = -gd

        if home == ref:
            hist_pts.append(pts)
            hist_gds.append(gd)
        else:
            hist_pts.append(1.0 - pts if pts in (0.0, 1.0) else 0.5)
            hist_gds.append(-gd)

    if not hist_pts:
        return {"h2h_n": 0, "h2h_home_winrate": None, "h2h_home_gd": None}

    return {
        "h2h_n": len(hist_pts),
        "h2h_home_winrate": float(np.mean(hist_pts)),
        "h2h_home_gd": float(np.mean(hist_gds)),
    }


def build_modeling_dataset(
    results: pd.DataFrame,
    include_xg: bool = False,
) -> pd.DataFrame:
    """
    Full match-level feature matrix with targets.

    All features are computed using only information available before kickoff.
    """
    results = results.sort_values("date").reset_index(drop=True)
    results = compute_elo(results)

    # compute_elo now provides split attack/defence Elo features.
    # home_att_elo/home_def_elo/away_att_elo/away_def_elo/att_vs_def_* are
    # expected to match config.FEATURE_COLS.


    team_long = build_team_long(results)
    form_df = _rolling_form(team_long)
    inwc_form_df = _rolling_form_scoped(
        team_long,
        WC2026_START,
        IN_TOURNAMENT_FORM_WINDOWS,
    )
    h2h_df = _h2h_features(results)

    xg_rolling = pd.DataFrame()
    if include_xg:
        try:
            xg_long = build_match_xg_table()
            if not xg_long.empty:
                xg_rolling = rolling_xg_features(xg_long)
                logger.info("Loaded xG features for %d team-match rows", len(xg_rolling))
        except Exception as exc:
            logger.warning("xG features unavailable: %s", exc)


    matches = results.copy()
    matches["tournament_weight"] = matches["tournament"].map(_tournament_weight)
    matches["is_world_cup"] = matches["tournament"].map(_is_world_cup)

    form_home_records = []
    form_away_records = []
    inwc_form_home_records = []
    inwc_form_away_records = []
    for i, row in matches.iterrows():
        hf = _lookup_form(form_df, row["home_team"], row["date"])
        af = _lookup_form(form_df, row["away_team"], row["date"])
        form_home_records.append(hf)
        form_away_records.append(af)

        ihf = _lookup_inwc_form(inwc_form_df, row["home_team"], row["date"])
        iaf = _lookup_inwc_form(inwc_form_df, row["away_team"], row["date"])
        inwc_form_home_records.append(ihf)
        inwc_form_away_records.append(iaf)

    for w in FORM_WINDOWS:
        matches[f"home_win{w}"] = [r[f"win{w}"] for r in form_home_records]
        matches[f"home_gd{w}"] = [r[f"gd{w}"] for r in form_home_records]
        matches[f"home_gf{w}"] = [r[f"gf{w}"] for r in form_home_records]
        matches[f"home_ga{w}"] = [r[f"ga{w}"] for r in form_home_records]
        matches[f"away_win{w}"] = [r[f"win{w}"] for r in form_away_records]
        matches[f"away_gd{w}"] = [r[f"gd{w}"] for r in form_away_records]
        matches[f"away_gf{w}"] = [r[f"gf{w}"] for r in form_away_records]
        matches[f"away_ga{w}"] = [r[f"ga{w}"] for r in form_away_records]

    matches["home_rest_days"] = [r["rest_days"] for r in form_home_records]
    matches["away_rest_days"] = [r["rest_days"] for r in form_away_records]

    # WC2026-only in-tournament rolling form (no leakage; as-of uses strict <)
    for w in IN_TOURNAMENT_FORM_WINDOWS:
        matches[f"home_inwc_win{w}"] = [r[f"win{w}"] for r in inwc_form_home_records]
        matches[f"home_inwc_gd{w}"] = [r[f"gd{w}"] for r in inwc_form_home_records]
        matches[f"away_inwc_win{w}"] = [r[f"win{w}"] for r in inwc_form_away_records]
        matches[f"away_inwc_gd{w}"] = [r[f"gd{w}"] for r in inwc_form_away_records]

    matches = matches.join(h2h_df)

    if not xg_rolling.empty:
        for i, row in matches.iterrows():
            hx = lookup_xg_features(xg_rolling, row["home_team"], row["date"])
            ax = lookup_xg_features(xg_rolling, row["away_team"], row["date"])
            for w in XG_WINDOWS:
                matches.at[i, f"home_xg_for_avg{w}"] = hx[f"xg_for_avg{w}"]
                matches.at[i, f"home_xg_against_avg{w}"] = hx[f"xg_against_avg{w}"]
                matches.at[i, f"away_xg_for_avg{w}"] = ax[f"xg_for_avg{w}"]
                matches.at[i, f"away_xg_against_avg{w}"] = ax[f"xg_against_avg{w}"]

            # Per-side gap signal (consistent naming with FEATURE_COLS)
            if 5 in XG_WINDOWS:
                matches.at[i, "home_xg_diff_5"] = (
                    None if hx["xg_for_avg5"] is None or hx["xg_against_avg5"] is None
                    else hx["xg_for_avg5"] - hx["xg_against_avg5"]
                )
                matches.at[i, "away_xg_diff_5"] = (
                    None if ax["xg_for_avg5"] is None or ax["xg_against_avg5"] is None
                    else ax["xg_for_avg5"] - ax["xg_against_avg5"]
                )


    matches["result_encoded"] = matches.apply(
        lambda r: encode_result(int(r["home_score"]), int(r["away_score"])),
        axis=1,
    )
    matches["y_home_goals"] = matches["home_score"].astype(int)
    matches["y_away_goals"] = matches["away_score"].astype(int)

    return matches


def build_feature_row(
    results: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
    neutral: int = 0,
    tournament: str = "FIFA World Cup",
    form_df: pd.DataFrame | None = None,
    xg_rolling: pd.DataFrame | None = None,
    elo_snapshot: dict[str, float] | None = None,
) -> dict:
    """
    Build a single feature row for inference at any future date.

    Uses historical results strictly before match_date.
    """
    from wc2026.config import ELO_START
    from wc2026.elo import get_elo_snapshot

    past = results[results["date"] < match_date].sort_values("date")
    if elo_snapshot is None:
        elo_snapshot = get_elo_snapshot(past, match_date)

    home_att_elo = elo_snapshot.get(f"{home_team}:att", ELO_START)
    home_def_elo = elo_snapshot.get(f"{home_team}:def", ELO_START)
    away_att_elo = elo_snapshot.get(f"{away_team}:att", ELO_START)
    away_def_elo = elo_snapshot.get(f"{away_team}:def", ELO_START)


    if form_df is None:
        team_long = build_team_long(past)
        form_df = _rolling_form(team_long)

    # WC2026-only in-tournament rolling form (scoped + strict < as-of inside lookup)
    team_long_inwc = build_team_long(past)
    inwc_form_df = _rolling_form_scoped(
        team_long_inwc,
        WC2026_START,
        IN_TOURNAMENT_FORM_WINDOWS,
    )

    hf = _lookup_form(form_df, home_team, match_date)
    af = _lookup_form(form_df, away_team, match_date)
    ihf = _lookup_inwc_form(inwc_form_df, home_team, match_date)
    iaf = _lookup_inwc_form(inwc_form_df, away_team, match_date)
    h2h = _lookup_h2h(past, home_team, away_team, match_date)

    row: dict = {
        "date": match_date,
        "home_team": home_team,
        "away_team": away_team,
        "neutral": neutral,
        "tournament_weight": _tournament_weight(tournament),
        "is_world_cup": _is_world_cup(tournament),
        "home_att_elo": home_att_elo,
        "home_def_elo": home_def_elo,
        "away_att_elo": away_att_elo,
        "away_def_elo": away_def_elo,
        "att_vs_def_home": home_att_elo - away_def_elo,
        "att_vs_def_away": away_att_elo - home_def_elo,

        "home_rest_days": hf["rest_days"],
        "away_rest_days": af["rest_days"],
        **h2h,
    }

    for w in FORM_WINDOWS:
        row[f"home_win{w}"] = hf[f"win{w}"]
        row[f"home_gd{w}"] = hf[f"gd{w}"]
        row[f"home_gf{w}"] = hf[f"gf{w}"]
        row[f"home_ga{w}"] = hf[f"ga{w}"]
        row[f"away_win{w}"] = af[f"win{w}"]
        row[f"away_gd{w}"] = af[f"gd{w}"]
        row[f"away_gf{w}"] = af[f"gf{w}"]
        row[f"away_ga{w}"] = af[f"ga{w}"]

    for w in IN_TOURNAMENT_FORM_WINDOWS:
        row[f"home_inwc_win{w}"] = ihf[f"win{w}"]
        row[f"home_inwc_gd{w}"] = ihf[f"gd{w}"]
        row[f"away_inwc_win{w}"] = iaf[f"win{w}"]
        row[f"away_inwc_gd{w}"] = iaf[f"gd{w}"]

    if xg_rolling is not None and not xg_rolling.empty:
        hx = lookup_xg_features(xg_rolling, home_team, match_date)
        ax = lookup_xg_features(xg_rolling, away_team, match_date)
        for w in XG_WINDOWS:
            row[f"home_xg_for_avg{w}"] = hx[f"xg_for_avg{w}"]
            row[f"home_xg_against_avg{w}"] = hx[f"xg_against_avg{w}"]
            row[f"away_xg_for_avg{w}"] = ax[f"xg_for_avg{w}"]
            row[f"away_xg_against_avg{w}"] = ax[f"xg_against_avg{w}"]

        # Per-side gap signal (consistent naming with FEATURE_COLS)
        if 5 in XG_WINDOWS:
            home_for = row.get("home_xg_for_avg5")
            home_against = row.get("home_xg_against_avg5")
            away_for = row.get("away_xg_for_avg5")
            away_against = row.get("away_xg_against_avg5")

            row["home_xg_diff_5"] = (
                None if home_for is None or home_against is None else home_for - home_against
            )
            row["away_xg_diff_5"] = (
                None if away_for is None or away_against is None else away_for - away_against
            )


    return row




def build_inference_context(results: pd.DataFrame, include_xg: bool = False) -> dict:

    """Precompute form and (optionally) xG rolling tables for fast batch inference."""
    team_long = build_team_long(results)
    form_df = _rolling_form(team_long)
    xg_rolling = pd.DataFrame()
    if include_xg:
        try:
            from wc2026.xg_features import build_match_xg_table, rolling_xg_features

            xg_long = build_match_xg_table()
            if not xg_long.empty:
                xg_rolling = rolling_xg_features(xg_long)
        except Exception as exc:
            logger.warning("xG context unavailable: %s", exc)

    return {"form_df": form_df, "xg_rolling": xg_rolling}



def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract model feature columns, filling NaNs with column medians.

    Gracefully handles missing columns (e.g., xG features when StatsBomb data unavailable).
    Missing columns are filled with 0.0 as the global fallback.
    """
    available_cols = [col for col in FEATURE_COLS if col in df.columns]
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        logger.warning(
            "Feature columns not in dataset (will use 0.0): %s",
            missing_cols,
        )

    X = df[available_cols].apply(pd.to_numeric, errors="coerce")
    for col in X.columns:
        median = X[col].median()
        X[col] = X[col].fillna(median if pd.notna(median) else 0.0)

    # Add missing columns as 0-filled
    for col in missing_cols:
        X[col] = 0.0

    # Return columns in the same order as FEATURE_COLS
    return X[FEATURE_COLS]
