"""Dynamic split (attack/defence) Elo ratings for international teams."""

from __future__ import annotations

import math

import pandas as pd

from wc2026.config import (
    DEFAULT_TOURNAMENT_WEIGHT,
    ELO_GOAL_DIFF_MULT,
    ELO_HOME_BONUS,
    ELO_K,
    ELO_RECENCY_PULL,
    ELO_START,
    TOURNAMENT_WEIGHTS,
)


def _expected_goals(att_rating: float, def_rating: float, base: float = 1.3) -> float:
    """Elo-style multiplicative expected goals.

    Returns an expected goals rate proxy (not a win probability).
    """
    return base * (10 ** ((att_rating - def_rating) / 400.0))



def _goal_diff_multiplier(home_score: int, away_score: int) -> float:
    gd = abs(home_score - away_score)
    if gd <= 1:
        return 1.0
    return 1.0 + ELO_GOAL_DIFF_MULT * math.log(gd)


def _tournament_weight(tournament: str) -> float:
    if pd.isna(tournament):
        return DEFAULT_TOURNAMENT_WEIGHT
    t = str(tournament)
    for key, w in TOURNAMENT_WEIGHTS.items():
        if key in t:
            return w
    return DEFAULT_TOURNAMENT_WEIGHT


def _recency_pull_factor(last_year: int, current_year: int) -> float:
    """Blend coefficient to apply when moving from last_year to current_year.

    Each Jan 1, ratings are blended toward ELO_START:
      rating = rating * (1 - pull) + ELO_START * pull

    For delta years, compound the decay.
    """
    years = max(0, current_year - last_year)
    # After n years with per-year pull p: rating *= (1-p)^n + ...
    return (1.0 - ELO_RECENCY_PULL) ** years


def compute_elo(results: pd.DataFrame) -> pd.DataFrame:
    """Add pre-match split Elo columns.

    Adds:
      home_att_elo, home_def_elo, away_att_elo, away_def_elo,
      att_vs_def_home, att_vs_def_away

    Attack and defence ratings are updated after each match.

    Simplification:
      - Attack updates based on the match outcome perspective for attack.
      - Defence updates based on inverted perspective.

    This keeps the model shape consistent while still separating offence/defence.
    """

    # initialise rating dicts
    attack_rating: dict[str, float] = {}
    defence_rating: dict[str, float] = {}

    last_year_by_team: dict[str, int] = {}

    # pre-match snapshots
    home_att: list[float] = []
    home_def: list[float] = []
    away_att: list[float] = []
    away_def: list[float] = []

    for _, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        neutral = int(row.get("neutral", 0))

        match_year = pd.Timestamp(row["date"]).year

        def _apply_recency(team: str) -> tuple[float, float]:
            a = attack_rating.get(team, ELO_START)
            d = defence_rating.get(team, ELO_START)
            last_year = last_year_by_team.get(team, match_year)
            if match_year > last_year:
                pull_mul = _recency_pull_factor(last_year, match_year)
                # rating = rating*pull_mul + ELO_START*(1-pull_mul)
                a = a * pull_mul + ELO_START * (1.0 - pull_mul)
                d = d * pull_mul + ELO_START * (1.0 - pull_mul)
            last_year_by_team[team] = match_year
            return a, d

        h_att, h_def = _apply_recency(home)
        a_att, a_def = _apply_recency(away)

        home_att.append(h_att)
        home_def.append(h_def)
        away_att.append(a_att)
        away_def.append(a_def)

        # Expected goals proxy (used for independent attack/defence updates)
        # Neutral games remove home advantage from the attacking side only.
        home_adv = 0.0 if neutral else ELO_HOME_BONUS

        actual_home_goals = int(row["home_score"])
        actual_away_goals = int(row["away_score"])

        exp_goals_home = _expected_goals(h_att + home_adv, a_def)
        exp_goals_away = _expected_goals(a_att, h_def)

        mult = _goal_diff_multiplier(actual_home_goals, actual_away_goals)
        k_eff = ELO_K * mult * _tournament_weight(row.get("tournament", None))

        # Attack updates toward actual goals scored vs expectation
        attack_rating[home] = h_att + k_eff * (actual_home_goals - exp_goals_home)
        # Defence: separate signal based on what the opponent was expected to concede.
        defence_rating[home] = h_def + k_eff * (exp_goals_away - actual_away_goals)

        attack_rating[away] = a_att + k_eff * (actual_away_goals - exp_goals_away)
        defence_rating[away] = a_def + k_eff * (exp_goals_home - actual_home_goals)


    out = results.copy()
    out["home_att_elo"] = home_att
    out["home_def_elo"] = home_def
    out["away_att_elo"] = away_att
    out["away_def_elo"] = away_def
    out["att_vs_def_home"] = out["home_att_elo"] - out["away_def_elo"]
    out["att_vs_def_away"] = out["away_att_elo"] - out["home_def_elo"]
    return out


def get_elo_snapshot(
    results: pd.DataFrame,
    as_of: pd.Timestamp,
) -> dict[str, float]:
    """Compute split Elo ratings using all matches strictly before as_of."""
    subset = results[results["date"] < as_of].copy()
    if subset.empty:
        return {}

    attack_rating: dict[str, float] = {}
    defence_rating: dict[str, float] = {}
    last_year_by_team: dict[str, int] = {}

    for _, row in subset.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        neutral = int(row.get("neutral", 0))
        match_year = pd.Timestamp(row["date"]).year

        def _apply_recency(team: str) -> tuple[float, float]:
            a = attack_rating.get(team, ELO_START)
            d = defence_rating.get(team, ELO_START)
            last_year = last_year_by_team.get(team, match_year)
            if match_year > last_year:
                pull_mul = _recency_pull_factor(last_year, match_year)
                a = a * pull_mul + ELO_START * (1.0 - pull_mul)
                d = d * pull_mul + ELO_START * (1.0 - pull_mul)
            last_year_by_team[team] = match_year
            return a, d

        h_att, h_def = _apply_recency(home)
        a_att, a_def = _apply_recency(away)

        home_adv = 0.0 if neutral else ELO_HOME_BONUS

        actual_home_goals = int(row["home_score"])
        actual_away_goals = int(row["away_score"])

        exp_goals_home = _expected_goals(h_att + home_adv, a_def)
        exp_goals_away = _expected_goals(a_att, h_def)

        mult = _goal_diff_multiplier(actual_home_goals, actual_away_goals)
        k_eff = ELO_K * mult * _tournament_weight(row.get("tournament", None))

        attack_rating[home] = h_att + k_eff * (actual_home_goals - exp_goals_home)
        defence_rating[home] = h_def + k_eff * (exp_goals_away - actual_away_goals)

        attack_rating[away] = a_att + k_eff * (actual_away_goals - exp_goals_away)
        defence_rating[away] = a_def + k_eff * (exp_goals_home - actual_home_goals)


    snapshot: dict[str, float] = {}
    for team, a in attack_rating.items():
        snapshot[f"{team}:att"] = a
    for team, d in defence_rating.items():
        snapshot[f"{team}:def"] = d
    return snapshot

