"""Visualization and formatted prediction display for WC 2026."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

from wc2026.config import OUTPUTS_DIR
from wc2026.score_matrix import generate_score_matrix, format_top_scores

logger = logging.getLogger(__name__)


def _format_prob(p: float) -> str:
    """Format probability as a percentage with color indicator."""
    pct = p * 100
    if pct >= 50:
        return f"[green]{pct:.1f}%[/green]"
    elif pct >= 33:
        return f"[yellow]{pct:.1f}%[/yellow]"
    else:
        return f"{pct:.1f}%"


def _expected_score(home_goals: float, away_goals: float) -> tuple[int, int]:
    """Round expected goals to nearest integer for implied final score."""
    import math
    # Use Poisson-like rounding: round to nearest, but floor for low expected
    if home_goals < 0.5:
        h = 0
    elif home_goals < 1.0:
        h = 1 if home_goals > 0.75 else 0
    else:
        h = int(math.floor(home_goals + 0.5))

    if away_goals < 0.5:
        a = 0
    elif away_goals < 1.0:
        a = 1 if away_goals > 0.75 else 0
    else:
        a = int(math.floor(away_goals + 0.5))

    return h, a


def format_predictions_table(df: pd.DataFrame) -> str:
    """Format a predictions DataFrame as a readable table."""
    lines = []
    lines.append("=" * 100)
    lines.append("WC 2026 MATCH PREDICTIONS")
    lines.append("=" * 100)
    lines.append("")

    for _, row in df.iterrows():
        round_name = row.get("round", "Unknown Round")
        date = row.get("date", "TBD")
        home = row.get("home_team", "?")
        away = row.get("away_team", "?")

        p_home = row.get("p_home_win", 0)
        p_draw = row.get("p_draw", 0)
        p_away = row.get("p_away_win", 0)
        exp_h = row.get("exp_home_goals", 0)
        exp_a = row.get("exp_away_goals", 0)

        # Determine predicted winner
        probs = {"H": p_home, "D": p_draw, "A": p_away}
        top_outcome = max(probs, key=probs.get)
        if top_outcome == "H":
            winner = home
        elif top_outcome == "A":
            winner = away
        else:
            winner = "Draw"

        imp_h, imp_a = _expected_score(exp_h, exp_a)

        lines.append(f"  {round_name}  |  {date}  |  {home} vs {away}")
        lines.append(f"  Implied Score: {home} {imp_h}-{imp_a} {away}")
        lines.append(f"  Probabilities: {home} {_format_prob(p_home)}  |  Draw {format(p_draw*100, '.1f')}%  |  {away} {_format_prob(p_away)}")
        lines.append(f"  Expected Goals: {home} {exp_h:.2f} - {exp_a:.2f} {away}")
        lines.append(f"  Predicted Winner: {winner} ({probs[top_outcome]*100:.1f}%)")

        # Show top 3 most likely scores
        sp = generate_score_matrix(exp_h, exp_a)
        scores = sp.top_scores[:3]
        score_strs = [f"{home_goals}-{away_goals} ({prob*100:.1f}%)" for (away_goals, home_goals), prob in scores]
        lines.append(f"  Most Likely Scores: {' | '.join(score_strs)}")
        lines.append("-" * 100)
        lines.append("")

    return "\n".join(lines)


def format_group_stage(df: pd.DataFrame) -> str:
    """Format predictions by group stage with team matchups."""
    lines = []
    lines.append("=" * 100)
    lines.append("WC 2026 GROUP STAGE PREDICTIONS")
    lines.append("=" * 100)
    lines.append("")

    group_matches = df[df.get("round", pd.Series([""] * len(df))).str.contains("Group", case=False, na=False)]

    if group_matches.empty:
        lines.append("No group stage predictions available.")
        return "\n".join(lines)

    current_group = None
    for _, row in group_matches.iterrows():
        round_name = row.get("round", "")

        # Extract group letter if present
        parts = round_name.split()
        if len(parts) >= 2:
            group_id = parts[1]
        else:
            group_id = round_name

        if group_id != current_group:
            if current_group is not None:
                lines.append("")
            lines.append(f"  ╔════════════════════════════════════════════════╗")
            lines.append(f"  ║  GROUP {group_id}                                     ║")
            lines.append(f"  ╚════════════════════════════════════════════════╝")
            lines.append("")
            current_group = group_id

        home = row.get("home_team", "?")
        away = row.get("away_team", "?")
        p_home = row.get("p_home_win", 0)
        p_draw = row.get("p_draw", 0)
        p_away = row.get("p_away_win", 0)
        exp_h = row.get("exp_home_goals", 0)
        exp_a = row.get("exp_away_goals", 0)

        # Find predicted winner
        probs = {"H": p_home, "D": p_draw, "A": p_away}
        top = max(probs, key=probs.get)
        if top == "H":
            winner = home
        elif top == "A":
            winner = away
        else:
            winner = "×"

        imp_h, imp_a = _expected_score(exp_h, exp_a)

        lines.append(f"    {home:15} [{p_home*100:5.1f}%]  vs  [{p_away*100:5.1f}%] {away:15}  → {imp_h}-{imp_a}")

    lines.append("")
    return "\n".join(lines)


def suggest_stakes(df: pd.DataFrame, min_prob: float = 0.60) -> list[dict]:
    """Identify high-confidence predictions for possible stakes.

    Returns matches where a team has > min_prob probability of winning.
    """
    stakes = []
    for _, row in df.iterrows():
        home = row.get("home_team", "?")
        away = row.get("away_team", "?")
        p_home = row.get("p_home_win", 0)
        p_away = row.get("p_away_win", 0)

        if p_home >= min_prob:
            stakes.append({
                "match": f"{home} vs {away}",
                "pick": home,
                "probability": p_home,
                "confidence": "HIGH" if p_home >= 0.70 else "MEDIUM",
            })
        if p_away >= min_prob:
            stakes.append({
                "match": f"{home} vs {away}",
                "pick": away,
                "probability": p_away,
                "confidence": "HIGH" if p_away >= 0.70 else "MEDIUM",
            })

    # Sort by probability descending
    stakes.sort(key=lambda x: x["probability"], reverse=True)
    return stakes


def format_stakes(df: pd.DataFrame) -> str:
    """Format high-confidence picks for informational purposes."""
    stakes = suggest_stakes(df, min_prob=0.60)

    lines = []
    lines.append("=" * 100)
    lines.append("HIGH-CONFIDENCE PICKS (60%+ win probability)")
    lines.append("=" * 100)
    lines.append("")

    if not stakes:
        lines.append("  No high-confidence picks found.")
    else:
        for s in stakes:
            conf_emoji = "🟢" if s["confidence"] == "HIGH" else "🟡"
            lines.append(
                f"  {conf_emoji} {s['match']}: {s['pick']} "
                f"({s['probability']*100:.1f}%) [{s['confidence']}]"
            )

    lines.append("")
    return "\n".join(lines)


def display_predictions(df: pd.DataFrame | Path | str, verbose: bool = False) -> None:
    """Display predictions in a formatted view.

    Args:
        df: DataFrame or path to predictions CSV
        verbose: Include group stage breakdown and stakes analysis
    """
    if isinstance(df, (Path, str)):
        df = pd.read_csv(df)

    # Console output
    table = format_predictions_table(df)
    print(table)

    if verbose:
        group_stage = format_group_stage(df)
        print(group_stage)

        stakes = format_stakes(df)
        print(stakes)