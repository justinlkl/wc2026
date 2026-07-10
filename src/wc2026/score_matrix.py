"""Score probability matrix using Poisson distribution.

Generates probabilities for specific scorelines based on expected goals,
using a Poisson model with correlation adjustment for home/away goals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# Maximum goals to show in the matrix
MAX_GOALS = 4


@dataclass
class ScorePrediction:
    """Container for score prediction results."""
    home_exp: float
    away_exp: float
    matrix: np.ndarray  # shape (MAX_GOALS+1, MAX_GOALS+1), [away, home]
    prob_0v0: float
    prob_1v0: float
    prob_0v1: float
    prob_1v1: float
    prob_2v0: float
    prob_0v2: float
    prob_2v1: float
    prob_1v2: float
    prob_2v2: float
    prob_3v0: float
    prob_0v3: float

    @property
    def top_scores(self) -> list[tuple[tuple[int, int], float]]:
        """Return top 3 most probable scores as ((away, home), probability)."""
        scores = []
        for away in range(MAX_GOALS + 1):
            for home in range(MAX_GOALS + 1):
                prob = self.matrix[away, home]
                scores.append(((away, home), prob))

        # Sort by probability descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:3]


def _poisson_pmf(lam: float, k: int) -> float:
    """Compute Poisson probability P(X=k) with rate parameter lambda."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    # Use log to avoid overflow
    log_prob = k * np.log(lam) - lam - _log_factorial(k)
    return float(np.exp(log_prob))


def _log_factorial(n: int) -> float:
    """Compute log(n!) using Stirling's approximation or lookup table."""
    if n <= 1:
        return 0.0
    if n < 20:
        # Use exact values for small n
        log_sum = 0.0
        for i in range(2, n + 1):
            log_sum += np.log(i)
        return log_sum
    # Stirling's approximation for larger n
    return n * np.log(n) - n + 0.5 * np.log(2 * np.pi * n)


def _estimate_correlation(home_exp: float, away_exp: float) -> float:
    """
    Estimate the correlation between home and away goals based on expected values.

    In football, there's typically a weak positive correlation - high-scoring games
    tend to have goals for both teams. We estimate this from the data pattern.
    """
    # Base correlation coefficient (typically 0.1-0.2 in football)
    base_corr = 0.15

    # Scale down for extreme values
    avg_exp = (home_exp + away_exp) / 2
    if avg_exp < 0.5:
        scale = 0.5
    elif avg_exp > 3:
        scale = 0.8
    else:
        scale = 1.0

    return base_corr * scale


def _bivariate_poisson_prob(
    home_lambda: float,
    away_lambda: float,
    home_goals: int,
    away_goals: int,
) -> float:
    """
    Compute joint probability using a bivariate Poisson-like distribution.

    Uses Copula approach: P(X=h, Y=a) = P(X=h) * P(Y=a) * (1 + corr * adjustment)
    where adjustment accounts for the correlation between goals.
    """
    # Base probabilities
    p_home = _poisson_pmf(home_lambda, home_goals)
    p_away = _poisson_pisson_pmf(away_lambda, away_goals)

    if p_home == 0 or p_away == 0:
        return 0.0

    # Correlation factor
    corr = _estimate_correlation(home_lambda, away_lambda)

    # Adjustment factor: if both teams scored, increase probability slightly
    # If one scored and other didn't, decrease slightly
    z_h = (home_goals - home_lambda) / np.sqrt(home_lambda + 0.001)
    z_a = (away_goals - away_lambda) / np.sqrt(away_lambda + 0.001)

    # Copula adjustment for correlation
    copula_adj = 1 + corr * min(2, z_h * z_a)

    # Ensure valid probability after adjustment
    prob = p_home * p_away * copula_adj

    # Clamp to [0, 1]
    return max(0.0, min(1.0, prob))


def _poisson_pisson_pmf(lam: float, k: int) -> float:
    """Wrapper for Poisson PMF."""
    return _poisson_pmf(lam, k)


def generate_score_matrix(
    home_exp_goals: float,
    away_exp_goals: float,
    home_name: Optional[str] = None,
    away_name: Optional[str] = None,
) -> ScorePrediction:
    """
    Generate a probability matrix for all possible scorelines.

    Args:
        home_exp_goals: Expected goals for home team
        away_exp_goals: Expected goals for away team
        home_name: Optional home team name for display
        away_name: Optional away team name for display

    Returns:
        ScorePrediction object containing the matrix and key probabilities
    """
    # Create matrix
    matrix = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))

    # Calculate total probability for normalization
    total = 0.0
    for away in range(MAX_GOALS + 1):
        for home in range(MAX_GOALS + 1):
            prob = _bivariate_poisson_prob(home_exp_goals, away_exp_goals, home, away)
            matrix[away, home] = prob
            total += prob

    # Normalize if needed
    if total > 0 and abs(total - 1.0) > 0.01:
        matrix = matrix / total

    return ScorePrediction(
        home_exp=home_exp_goals,
        away_exp=away_exp_goals,
        matrix=matrix,
        prob_0v0=matrix[0, 0],
        prob_1v0=matrix[1, 0],
        prob_0v1=matrix[0, 1],
        prob_1v1=matrix[1, 1],
        prob_2v0=matrix[2, 0],
        prob_0v2=matrix[0, 2],
        prob_2v1=matrix[2, 1],
        prob_1v2=matrix[1, 2],
        prob_2v2=matrix[2, 2],
        prob_3v0=matrix[3, 0],
        prob_0v3=matrix[0, 3],
    )


def format_score_matrix(sp: ScorePrediction, home_name: str = "Home", away_name: str = "Away") -> str:
    """Format the score probability matrix as a nice ASCII table."""
    lines = []

    # Header
    home_w = max(len(home_name), 4)
    away_w = max(len(away_name), 4)
    goal_w = 8

    lines.append(f"{'':>{away_w}} \\ {'':>{home_w}} | " + " ".join(f"{'H' + str(g):>{goal_w}}" for g in range(MAX_GOALS + 1)))
    lines.append("-" * (away_w + home_w + 2 + (MAX_GOALS + 1) * (goal_w + 1)))

    # Data rows
    for away in range(MAX_GOALS + 1):
        row = f"{'A' + str(away):>{away_w}} |"
        for home in range(MAX_GOALS + 1):
            prob = sp.matrix[away, home] * 100
            score = f"{home}-{away}"
            row += f" {prob:>{goal_w}.1f}%"
        lines.append(row)

    lines.append("")
    lines.append(f"(Home goals shown horizontally, Away goals shown vertically)")

    return "\n".join(lines)


def format_top_scores(
    home_exp_goals: float,
    away_exp_goals: float,
    home_name: str,
    away_name: str,
    n: int = 3,
) -> str:
    """Format the top N most likely scorelines."""
    sp = generate_score_matrix(home_exp_goals, away_exp_goals)

    lines = []
    lines.append(f"Most Likely Scorelines:")
    lines.append("-" * 40)

    for rank, ((away_goals, home_goals), prob) in enumerate(sp.top_scores[:n], 1):
        pct = prob * 100
        score_str = f"{home_name} {home_goals}-{away_goals} {away_name}"
        lines.append(f"  {rank}. {score_str} ({pct:.1f}%)")

    return "\n".join(lines)


def add_score_predictions_to_results(df: dict) -> dict:
    """
    Add score matrix to prediction result dictionary.

    Args:
        df: Dictionary with prediction results (modified in place)

    Returns:
        Updated dictionary with score matrix data
    """
    sp = generate_score_matrix(
        df.get("exp_home_goals", 1.0),
        df.get("exp_away_goals", 1.0),
        df.get("home_team"),
        df.get("away_team"),
    )

    # Add matrix (flattened) and key probabilities
    df["top_scores"] = sp.top_scores

    # Add key scoreline probabilities
    for away in range(MAX_GOALS + 1):
        for home in range(MAX_GOALS + 1):
            df[f"prob_{home}v{away}"] = sp.matrix[away, home]

    return df