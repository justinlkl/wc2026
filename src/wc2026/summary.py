"""WC 2026 Tournament Summary and Analysis"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from wc2026.config import MODELS_DIR, OUTPUTS_DIR
from wc2026.github_loader import (
    _load_tables,
    _standardise_matches,
    _matches_to_results,
)
from wc2026.team_names import canonical_team
from wc2026.score_matrix import generate_score_matrix


def tournament_standings() -> pd.DataFrame:
    """Compute group stage standings from completed matches."""
    matches_df, teams_df = _load_tables()
    df = _standardise_matches(matches_df, teams_df)
    completed = df[df["status"] == "Completed"]

    # Initialize standings
    standings: dict[str, dict] = {}

    for _, match in completed.iterrows():
        home = canonical_team(str(match["home_team"]))
        away = canonical_team(str(match["away_team"]))
        h_score = int(match["home_score"]) if pd.notna(match["home_score"]) else 0
        a_score = int(match["away_score"]) if pd.notna(match["away_score"]) else 0

        # Initialize teams if needed
        for team in [home, away]:
            if team and team not in standings:
                standings[team] = {
                    "team": team,
                    "played": 0,
                    "won": 0,
                    "drawn": 0,
                    "lost": 0,
                    "gf": 0,
                    "ga": 0,
                    "gd": 0,
                    "points": 0,
                }

        if home and home in standings:
            standings[home]["played"] += 1
            standings[home]["gf"] += h_score
            standings[home]["ga"] += a_score
            standings[home]["gd"] += h_score - a_score

            if h_score > a_score:
                standings[home]["won"] += 1
                standings[home]["points"] += 3
            elif h_score == a_score:
                standings[home]["drawn"] += 1
                standings[home]["points"] += 1
            else:
                standings[home]["lost"] += 1

        if away and away in standings:
            standings[away]["played"] += 1
            standings[away]["gf"] += a_score
            standings[away]["ga"] += h_score
            standings[away]["gd"] += a_score - h_score

            if a_score > h_score:
                standings[away]["won"] += 1
                standings[away]["points"] += 3
            elif a_score == h_score:
                standings[away]["drawn"] += 1
                standings[away]["points"] += 1
            else:
                standings[away]["lost"] += 1

    # Convert to DataFrame
    df_standings = pd.DataFrame(standings.values())
    df_standings = df_standings.sort_values(["points", "gd", "gf"], ascending=[False, False, False])
    return df_standings


def model_performance() -> dict:
    """Load and display model training metadata."""
    meta_path = MODELS_DIR / "training_meta.json"
    if not meta_path.exists():
        return {"error": "Model not trained yet. Run: python -m wc2026.cli train"}

    with open(meta_path) as f:
        meta = json.load(f)

    metrics = meta.get("metrics", [])
    features = meta.get("feature_cols", [])

    return {
        "metrics": metrics,
        "num_features": len(features),
        "top_features": meta.get("top_features", [])[:10],
    }


def predictions_summary() -> pd.DataFrame | None:
    """Load latest predictions."""
    pred_path = OUTPUTS_DIR / "wc2026_predictions.csv"
    if not pred_path.exists():
        return None
    return pd.read_csv(pred_path)


def print_tournament_summary() -> None:
    """Print a comprehensive WC 2026 summary."""
    print("=" * 80)
    print("FIFA WORLD CUP 2026 PREDICTION MODEL SUMMARY")
    print("=" * 80)
    print()

    # Model Performance
    print("MODEL PERFORMANCE")
    print("-" * 40)
    perf = model_performance()
    if "error" in perf:
        print(f"  {perf['error']}")
    else:
        print(f"  Number of features: {perf['num_features']}")
        print()
        for m in perf["metrics"]:
            split = m["split"].upper()
            acc = m["accuracy"] * 100
            ll = m["log_loss"]
            baseline = m["baseline_log_loss"]
            home_mae = m["home_mae"]
            away_mae = m["away_mae"]
            print(
                f"  {split:>6}: Accuracy={acc:.1f}% | LogLoss={ll:.3f} "
                f"(baseline={baseline:.3f}) | "
                f"Home MAE={home_mae:.2f} | Away MAE={away_mae:.2f}"
            )
        print()
        print("  Top 5 Features by Importance:")
        for i, feat in enumerate(perf["top_features"][:5], 1):
            print(f"    {i}. {feat['feature']}: {feat['gain']:.2f}")

    print()

    # Tournament Progress
    print("TOURNAMENT PROGRESS")
    print("-" * 40)
    matches_df, teams_df = _load_tables()
    df = _standardise_matches(matches_df, teams_df)

    completed = df[df["status"] == "Completed"]
    scheduled = df[df["status"] == "Scheduled"]

    print(f"  Matches completed: {len(completed)}")
    print(f"  Matches remaining: {len(scheduled)}")
    print()

    # Recent Results (last 5)
    print("  Recent Results:")
    recent = completed.sort_values("date").tail(5)
    for _, row in recent.iterrows():
        home = canonical_team(str(row["home_team"]))
        away = canonical_team(str(row["away_team"]))
        h_score = int(row["home_score"]) if pd.notna(row["home_score"]) else 0
        a_score = int(row["away_score"]) if pd.notna(row["away_score"]) else 0
        print(f"    {row['date'].strftime('%Y-%m-%d')}: {home} {h_score}-{a_score} {away}")

    print()

    # Upcoming Predictions
    preds = predictions_summary()
    if preds is not None and len(preds) > 0:
        print("UPCOMING MATCH PREDICTIONS")
        print("-" * 40)
        for _, row in preds.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            p_home = row["p_home_win"] * 100
            p_away = row["p_away_win"] * 100
            p_draw = row["p_draw"] * 100
            exp_h = row["exp_home_goals"]
            exp_a = row["exp_away_goals"]

            # Determine winner
            probs = {"H": p_home, "D": p_draw, "A": p_away}
            winner = max(probs, key=probs.get)
            if winner == "H":
                pick = home
            elif winner == "A":
                pick = away
            else:
                pick = "Draw"

            stage_map = {
                4: "Semi-final",
                5: "Third Place",
                6: "Final",
            }
            stage = row.get("round", "Match")
            print(f"  {row['date']} | {row['round']}")
            print(f"    {home} vs {away}")
            print(f"    Pick: {pick} ({probs[winner]:.1f}%)")

            # Generate and display top 3 most likely scores
            sp = generate_score_matrix(exp_h, exp_a)
            print(f"    Most Likely Scores:")
            for rank, ((away_goals, home_goals), prob) in enumerate(sp.top_scores[:3], 1):
                print(f"      {rank}. {home} {home_goals}-{away_goals} {away} ({prob*100:.1f}%)")
            print(f"    Probabilities: {home} {p_home:.1f}% | Draw {p_draw:.1f}% | {away} {p_away:.1f}%")
            print(f"    Expected Goals: {home} {exp_h:.2f} | {away} {exp_a:.2f}")
            print()
    else:
        print("  No predictions available. Run: python -m wc2026.cli predict")

    print("=" * 80)


def print_standings() -> None:
    """Print full tournament standings."""
    import math

    standings = tournament_standings()

    if standings.empty:
        print("No completed matches yet.")
        return

    print("=" * 80)
    print("GROUP STAGE STANDINGS")
    print("=" * 80)
    print()
    print(
        f"{'Team':<20} {'P':>3} {'W':>3} {'D':>3} {'L':>3} {'GF':>4} {'GA':>4} {'GD':>4} {'Pts':>4}"
    )
    print("-" * 70)

    for _, row in standings.iterrows():
        team = row["team"]
        if team:
            print(
                f"{team:<20} "
                f"{int(row['played']):>3} "
                f"{int(row['won']):>3} "
                f"{int(row['drawn']):>3} "
                f"{int(row['lost']):>3} "
                f"{int(row['gf']):>4} "
                f"{int(row['ga']):>4} "
                f"{int(row['gd']):>4} "
                f"{int(row['points']):>4}"
            )
    print()


if __name__ == "__main__":
    print_tournament_summary()
    print()
    print_standings()