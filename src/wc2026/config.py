"""Project-wide constants and paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
# Add after DATA_DIR definition:
KAGGLE_CACHE_DIR = DATA_DIR / "kaggle_wc2026"
KAGGLE_DATASET_SCHEDULE = "areezvisram12/fifa-world-cup-2026-match-data-unofficial"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
STATSBOMB_REPO = "https://github.com/statsbomb/open-data.git"

# Elo (single-rating baseline; replaced by split attack/defence in elo.py)
ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_BONUS = 100.0
ELO_GOAL_DIFF_MULT = 0.5  # extra K scaling for goal difference

# Elo recency / pull-toward-mean
# Each Jan 1, ratings are blended toward ELO_START by this factor.
ELO_RECENCY_PULL = 0.03


# Date filters
MIN_DATE = "2006-01-01"
TRAIN_END = "2018-06-30"
VAL_END = "2022-06-30"

# Rolling windows
FORM_WINDOWS = (5, 10)
XG_WINDOWS = (5, 10)

# WC 2026-only rolling form scope
WC2026_START = "2026-06-11"
IN_TOURNAMENT_FORM_WINDOWS = (3, 5)

# Competitive tournaments (substring match on tournament column)
COMPETITIVE_TOURNAMENT_PATTERNS = [
    "FIFA World Cup",
    "World Cup qualification",
    "UEFA Euro",
    "Euro qualification",
    "Copa América",
    "Copa America",
    "African Cup of Nations",
    "AFC Asian Cup",
    "CONCACAF Gold Cup",
    "Nations League",
    "Confederations Cup",
]

# Tournament importance weights
TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.0,

    "World Cup qualification": 0.85,
    "UEFA Euro": 0.95,
    "Euro qualification": 0.8,
    "Copa América": 0.9,
    "Copa America": 0.9,
    "African Cup of Nations": 0.85,
    "AFC Asian Cup": 0.85,
    "CONCACAF Gold Cup": 0.75,
    "Nations League": 0.7,
    "Confederations Cup": 0.85,
}
DEFAULT_TOURNAMENT_WEIGHT = 0.5

# StatsBomb World Cup competition IDs (men's)
STATSBOMB_WC_COMPETITIONS = {
    43: 2018,
    11: 2022,
}

# API-Football
APIFOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
WC_LEAGUE_ID = 1
WC_SEASON = 2026

# Feature columns used by models (order matters for inference)
FEATURE_COLS = [
    "neutral",
    "tournament_weight",
    "is_world_cup",
    # Split attack/defence Elo
    "home_att_elo",
    "home_def_elo",
    "away_att_elo",
    "away_def_elo",
    "att_vs_def_home",
    "att_vs_def_away",
    # Targets / form
    "home_win5",
    "home_gd5",
    "home_win10",
    "home_gd10",
    "home_rest_days",
    "home_gf5",
    "home_ga5",
    "away_win5",
    "away_gd5",
    "away_win10",
    "away_gd10",
    "away_rest_days",
    "away_gf5",
    "away_ga5",

    # WC2026-only in-tournament rolling form (home/away perspective)
    "home_inwc_win3",
    "home_inwc_gd3",
    "home_inwc_win5",
    "home_inwc_gd5",
    "away_inwc_win3",
    "away_inwc_gd3",
    "away_inwc_win5",
    "away_inwc_gd5",

    # H2H
    "h2h_n",
    "h2h_home_winrate",
    "h2h_home_gd",

    # xG-derived (per-side)
    "home_xg_for_avg5",
    "home_xg_against_avg5",
    "home_xg_for_avg10",
    "home_xg_against_avg10",
    "away_xg_for_avg5",
    "away_xg_against_avg5",
    "away_xg_for_avg10",
    "away_xg_against_avg10",
    # Direct gap signal (per-side)
    "home_xg_diff_5",
    "away_xg_diff_5",
]



RESULT_LABELS = ["home_win", "draw", "away_win"]


