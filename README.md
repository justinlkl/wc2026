# FIFA World Cup 2026 Predictor

Match outcome and goals model for FIFA World Cup 2026, built on international results, Elo, form, head-to-head, and xG features, with XGBoost classifiers/regressors and API-Football inference.

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

# Show detailed predictions with analysis
python -m wc2026.cli summary
```

## Commands

| Command | Description |
|---------|-------------|
| `download` | Download historical matches from martj42/international_results |
| `train` | Train XGBoost models with chronological splits |
| `predict` | Predict upcoming WC 2026 match outcomes |
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

## Project Layout

| Path | Purpose |
| ---- | ------- |
| `src/wc2026/config.py` | Constants, feature list, date splits |
| `src/wc2026/data_loading.py` | Download & filter international results |
| `src/wc2026/elo.py` | Dynamic Elo with split attack/defense ratings |
| `src/wc2026/features.py` | Form, H2H, tournament, match-level matrix |
| `src/wc2026/xg_features.py` | StatsBomb xG aggregation & rolling averages |
| `src/wc2026/model_train.py` | XGBoost training, metrics, feature importance |
| `src/wc2026/predict_world_cup.py` | GitHub fixtures + inference |
| `src/wc2026/team_names.py` | Team name normalization |
| `src/wc2026/visualization.py` | Formatted prediction display |
| `src/wc2026/summary.py` | Tournament standings and model summary |
| `data/` | Cached results, StatsBomb, GitHub dataset |
| `models/` | Trained `.joblib` artifacts + `training_meta.json` |
| `outputs/` | Prediction CSVs |

## Features

**Strength:**
- `home_att_elo` / `home_def_elo` (split Elo)
- `away_att_elo` / `away_def_elo`
- `att_vs_def_home` (home attack vs away defense)
- `att_vs_def_away` (away attack vs home defense)

**Form (rolling 5/10):**
- Win rate, goal diff, goals for/against, rest days

**H2H:**
- Meeting count, home win rate, avg goal diff

**Context:**
- `neutral`, `tournament_weight`, `is_world_cup`

**xG (when StatsBomb data available):**
- Rolling xG for/against averages (5/10 game windows)

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

## Data Sources

- [martj42/international_results](https://github.com/martj42/international_results) — historical national team results
- [StatsBomb open-data](https://github.com/statsbomb/open-data) — xG for WC 2018 & 2022
- [mominullptr/FIFA-World-Cup-2026-Dataset](https://github.com/mominullptr/FIFA-World-Cup-2026-Dataset) — WC 2026 fixtures and results

## Extending

- Add more xG sources (xgclient, TheStatsAPI) in `xg_features.py`
- Switch goal models to `reg:poisson` for Poisson goal distributions
- Parse knockout/group rounds from fixtures for `is_knockout` features
- Expand `team_names.py` aliases as you discover mapping gaps

## API-Football (Optional)

If you want to use API-Football for live fixtures:

1. Get an API key at [api-football.com](https://www.api-football.com/)
2. Add to `.env`:
   ```
   APIFOOTBALL_KEY=your_key_here
   ```
3. World Cup uses `league=1`, `season=2026`

Never commit API keys.# wc2026
