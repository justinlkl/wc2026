# FIFA World Cup 2026 Predictor

A machine learning model for predicting FIFA World Cup 2026 match outcomes and scorelines, built on international results, Elo ratings, form analysis, head-to-head history, and xG features.

## Architecture

```
international_results (2006+)
         │
         ▼
    Elo ratings (dynamic, split attack/defense)
         │
         ▼
    Form features (rolling 5/10: win rate, gd, gf, ga, rest days)
         │
         ▼
    H2H features (meeting count, home win rate, avg gd)
         │
         ▼
    Tournament context (weight, is_world_cup, neutral)
         │
         ▼
    ┌───────────────────────────────────────────┐
    │  XGBClassifier (home / draw / away)       │
    │  XGBRegressor × 2 (home goals, away goals)│
    │  Poisson Score Matrix (scoreline probs)    │
    └───────────────────────────────────────────┘
         │
         ▼
    GitHub WC 2026 fixtures → predictions
```

## Quick Start

```bash
cd /Users/lamjustin/Downloads/wc2026
source .venv/bin/activate

# Download historical data
python -m wc2026.cli download

# Train models (chronological split: train ≤2018, val ≤2022, test >2022)
python -m wc2026.cli train

# Predict WC 2026 fixtures
python -m wc2026.cli predict --source github

# Show detailed predictions with top 3 scorelines
python -m wc2026.cli summary
```

## Commands

| Command | Description |
|---------|-------------|
| `download` | Download historical matches from martj42/international_results |
| `train` | Train XGBoost models with chronological splits |
| `predict` | Predict upcoming WC 2026 match outcomes + top 3 scorelines |
| `summary` | Display tournament standings and model predictions |
| `update` | Update fixtures from GitHub dataset |

### Predict Options

```bash
# Use GitHub fixtures (default, no API key needed)
python -m wc2026.cli predict --source github

# Use local CSV fixtures
python -m wc2026.cli predict --source csv --from-csv data/fixtures.csv

# Verbose output with group stage and stakes analysis
python -m wc2026.cli predict --source github --verbose
```

## Features

### Match Outcome Features

| Feature | Description |
|---------|-------------|
| `home_att_elo` / `home_def_elo` | Split Elo (attack/defense) ratings for home team |
| `away_att_elo` / `away_def_elo` | Split Elo (attack/defense) ratings for away team |
| `att_vs_def_home` | Home attack vs Away defense differential |
| `att_vs_def_away` | Away attack vs Home defense differential |

### Form Features (rolling 5/10 games)

| Feature | Description |
|---------|-------------|
| `home_win5/w10` | Home team win rate (last 5/10 games) |
| `home_gd5/gd10` | Home team goal differential |
| `home_gf5/gf10` | Home team goals scored |
| `home_ga5/ga10` | Home team goals conceded |
| `away_*` | Same as above for away team |

### Head-to-Head Features

| Feature | Description |
|---------|-------------|
| `h2h_n` | Number of previous meetings |
| `h2h_home_winrate` | Win rate for home team in H2H |
| `h2h_home_gd` | Average goal differential in H2H |

### Tournament Context

| Feature | Description |
|---------|-------------|
| `neutral` | 1 if neutral venue |
| `tournament_weight` | Tournament importance (0.5-1.0) |
| `is_world_cup` | 1 if World Cup match |

### xG Features (when StatsBomb data available)

| Feature | Description |
|---------|-------------|
| `home_xg_for_avg5/10` | Rolling xG for (last 5/10 matches) |
| `home_xg_against_avg5/10` | Rolling xG against |
| `away_xg_for/against` | Same as above for away team |

## Model Performance

| Split | Accuracy | Log Loss | Baseline | Home MAE | Away MAE |
|-------|----------|----------|----------|----------|----------|
| Train | ~62.4%   | ~0.818   | 1.041    | ~0.92    | ~0.75    |
| Val   | ~58.0%   | ~0.914   | 1.050    | ~1.08    | ~0.87    |
| Test  | ~57.2%   | ~0.924   | 1.056    | ~1.08    | ~0.89    |

**Top Features by Importance:**
1. `att_vs_def_away` - Away team attack vs home defense differential
2. `att_vs_def_home` - Home team attack vs away defense differential
3. `away_gd10` - Away team 10-game goal difference
4. `home_gd10` - Home team 10-game goal difference
5. `away_win10` - Away team 10-game win rate

## Score Matrix

The model uses a **Poisson distribution** to generate probability matrices for specific scorelines:

```
Spain vs Belgium (Semi-finals)
Most Likely Scores:
  1. Spain 1-1 Belgium (12.4%)
  2. Spain 0-1 Belgium (9.2%)
  3. Spain 1-0 Belgium (8.9%)
```

The matrix accounts for:
- Expected goals for each team
- Weak correlation between goals (high-scoring games tend to have goals for both sides)
- Normalization to ensure probabilities sum to 100%

## Project Layout

```
wc2026/
├── src/wc2026/
│   ├── config.py           # Constants, feature list, date splits
│   ├── data_loading.py     # Download & filter international results
│   ├── elo.py              # Dynamic Elo with split attack/defense
│   ├── features.py         # Form, H2H, tournament, match-level matrix
│   ├── xg_features.py      # StatsBomb xG aggregation & rolling averages
│   ├── score_matrix.py     # Poisson-based scoreline probabilities
│   ├── model_train.py      # XGBoost training, metrics, feature importance
│   ├── predict_world_cup.py # GitHub fixtures + inference
│   ├── summary.py          # Tournament standings and summary
│   ├── visualization.py    # Formatted prediction display
│   ├── team_names.py       # Team name normalization (46 qualified teams)
│   ├── github_loader.py   # Load fixtures from GitHub WC 2026 dataset
│   └── kaggle_loader.py   # Load fixtures from Kaggle (optional)
├── data/
│   ├── results.csv          # Historical international results
│   ├── github_wc2026/      # GitHub WC 2026 fixtures & results
│   └── matches_detailed.csv # WC 2026 match details with xG
├── models/
│   ├── outcome_classifier.joblib
│   ├── home_goals_regressor.joblib
│   ├── away_goals_regressor.joblib
│   ├── feature_medians.joblib
│   └── training_meta.json
└── outputs/
    └── wc2026_predictions.csv
```

## Data Sources

- [martj42/international_results](https://github.com/martj42/international_results) — historical national team results
- [StatsBomb open-data](https://github.com/statsbomb/open-data) — xG for WC 2018 & 2022
- [mominullptr/FIFA-World-Cup-2026-Dataset](https://github.com/mominullptr/FIFA-World-Cup-2026-Dataset) — WC 2026 fixtures and results

## Understanding the Predictions

### Why Expected Goals ≠ Match Winner

The expected goals (xG) from the goal regressors show the *average* number of goals expected for each team. However, the outcome predictions come from a **separate XGBoost classifier** that learns from ALL features simultaneously:

- Belgium's higher xG (1.44 vs 1.41) means they're expected to score slightly more goals on average
- Spain wins in the model (40.5% vs 32.8%) because:
  - Spain has superior Elo ratings (attack/defense split)
  - Better recent form in high-stakes matches
  - Historical dominance in head-to-head matchups
  - Stronger defensive record (0 goals conceded in group stage)

The classifier considers many factors beyond raw expected goals, making it more nuanced than just comparing xG values.

### Probabilities Don't Sum to Win%

The probability of "Spain winning" (40.5%) doesn't mean Spain scores more goals than Belgium. It means:

- **P(Spain wins)** = probability of any scoreline where Spain has more goals (e.g., 1-0, 2-0, 2-1, etc.)
- **P(Draw)** = probability of equal goals (0-0, 1-1, 2-2, etc.)
- **P(Belgium wins)** = probability of any scoreline where Belgium has more goals

The most likely scoreline (1-1 at 12.4%) is actually a draw!

## Extending

- Add more xG sources (xgclient, TheStatsAPI) in `xg_features.py`
- Switch goal models to `reg:poisson` for Poisson goal distributions
- Parse knockout/group rounds from fixtures for `is_knockout` features
- Expand `team_names.py` aliases as you discover mapping gaps
- Calibrate probabilities using isotonic regression
- Add Elo prediction accuracy weights

## API-Football (Optional)

If you want to use API-Football for live fixtures:

1. Get an API key at [api-football.com](https://www.api-football.com/)
2. Add to `.env`:
   ```
   APIFOOTBALL_KEY=your_key_here
   ```
3. World Cup uses `league=1`, `season=2026`

Never commit API keys.