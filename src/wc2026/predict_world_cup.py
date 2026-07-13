from __future__ import annotations


import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from wc2026.config import DATA_DIR, OUTPUTS_DIR
from wc2026.data_loading import load_results
from wc2026.features import build_feature_row, build_inference_context
from wc2026.kaggle_loader import load_kaggle_fixtures
from wc2026.github_loader import load_github_completed_results, load_github_upcoming_fixtures

from wc2026.model_train import load_models, predict_match
from wc2026.team_names import canonical_team, is_qualified_team
from wc2026.score_matrix import generate_score_matrix

logger = logging.getLogger(__name__)

DEFAULT_FIXTURES_CSV = DATA_DIR / "fixtures.csv"
HISTORICAL_BACKBONE_END = pd.Timestamp("2024-12-31")


def _format_top_score(
    home: str,
    away: str,
    exp_home: float,
    exp_away: float,
    rank: int,
) -> str:
    """Format a top score prediction like '1-1 (12.5%)'."""
    sp = generate_score_matrix(exp_home, exp_away)
    if rank < len(sp.top_scores):
        away_goals, home_goals = sp.top_scores[rank][0]
        prob = sp.top_scores[rank][1] * 100
        return f"{home_goals}-{away_goals} ({prob:.1f}%)"
    return ""


def _historical_backbone() -> pd.DataFrame:
    """Load the historical training backbone capped at 2024-12-31."""
    results = load_results()
    return results[results["date"] <= HISTORICAL_BACKBONE_END].copy().reset_index(drop=True)


def _combined_results_with_github() -> pd.DataFrame:
    """Historical backbone plus completed matches from the GitHub WC 2026 dataset."""
    backbone = _historical_backbone()
    completed_2026 = load_github_completed_results()

    if completed_2026.empty:
        return backbone

    combined = pd.concat([backbone, completed_2026], ignore_index=True, sort=False)
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined


def load_fixtures_from_csv(path: Path | None = None) -> list[dict]:
    path = path or DEFAULT_FIXTURES_CSV
    if not path.exists():
        raise FileNotFoundError(

            f"Fixtures CSV not found: {path}\n"
            "Place your fixtures CSV at data/fixtures.csv with columns:\n"
            "  fixture_id, date, round, home_team, away_team"
        )
    df = pd.read_csv(path, parse_dates=["date"])
    fixtures = []
    for _, row in df.iterrows():
        home_team = canonical_team(str(row["home_team"]))
        away_team = canonical_team(str(row["away_team"]))
        if not (is_qualified_team(home_team) and is_qualified_team(away_team)):
            continue
        fixtures.append({
            "fixture_id": row["fixture_id"],
            "date": pd.Timestamp(row["date"]),
            "round": row.get("round", ""),
            "home_team": home_team,
            "away_team": away_team,
        })
    logger.info("Loaded %d fixtures from %s", len(fixtures), path)
    return fixtures


def load_fixtures_from_kaggle(force_download: bool = False) -> list[dict]:
    """Load fixtures from Kaggle when a local CSV is not available."""
    return load_kaggle_fixtures(force_download=force_download)


def predict_fixtures(
    fixtures: list[dict] | None = None,
    fixtures_csv: Path | None = None,
    output_path: Path | None = None,
    print_top_outcome: bool = False,
    source: Literal["csv", "kaggle", "github"] = "github",
    results_source: Literal["github", "historical"] = "github",

    kaggle_path: Path | None = None,
    force_kaggle_download: bool = False,
) -> pd.DataFrame:
    """Score all WC 2026 fixtures and write predictions CSV."""

    if results_source == "github" or source == "github":
        # Use combined results (historical + completed WC 2026 matches)
        results = _combined_results_with_github()
    else:
        results = _historical_backbone()
    context = build_inference_context(results, include_xg=True)

    models = load_models()

    if fixtures is None:
        if source == "kaggle":
            fixtures = load_kaggle_fixtures(
                path=kaggle_path,
                force_download=force_kaggle_download,
            )
        elif source == "github":
            fixtures = load_github_upcoming_fixtures(
                min_stage_id=4,
                force_download=force_kaggle_download,
            )
        elif fixtures_csv is not None:
            fixtures = load_fixtures_from_csv(fixtures_csv)
        else:
            fixtures = load_fixtures_from_csv()


    rows = []
    for fix in fixtures:
        home = fix["home_team"]
        away = fix["away_team"]
        date = pd.Timestamp(fix["date"])

        feat = build_feature_row(
            results=results,
            home_team=home,
            away_team=away,
            match_date=date,
            neutral=1,
            tournament="FIFA World Cup",
            form_df=context["form_df"],
            xg_rolling=context.get("xg_rolling"),
        )
        pred = predict_match(feat, models)

        if print_top_outcome:
            probs = {
                "home_win": pred["p_home_win"],
                "draw": pred["p_draw"],
                "away_win": pred["p_away_win"],
            }
            top_label = max(probs, key=probs.get)
            top_prob = probs[top_label] * 100.0
            if top_label == "home_win":
                winner = home
            elif top_label == "away_win":
                winner = away
            else:
                winner = "Draw"
            print(f">>> Predicted outcome: {winner} ({top_prob:.1f}%)")

        rows.append({

            "fixture_id": fix.get("fixture_id"),
            "date": date.date().isoformat(),
            "round": fix.get("round", ""),
            "home_team": home,
            "away_team": away,
            "p_home_win": round(pred["p_home_win"], 4),
            "p_draw":     round(pred["p_draw"],     4),
            "p_away_win": round(pred["p_away_win"], 4),
            "exp_home_goals": round(pred["exp_home_goals"], 2),
            "exp_away_goals": round(pred["exp_away_goals"], 2),
            # Score matrix
            "score1": _format_top_score(home, away, pred["exp_home_goals"], pred["exp_away_goals"], 0),
            "score2": _format_top_score(home, away, pred["exp_home_goals"], pred["exp_away_goals"], 1),
            "score3": _format_top_score(home, away, pred["exp_home_goals"], pred["exp_away_goals"], 2),
        })

    df = pd.DataFrame(rows).sort_values("date")
    out_path = output_path or OUTPUTS_DIR / "wc2026_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved %d predictions → %s", len(df), out_path)
    return df