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
    │  Poisson Score Matrix (scoreline probs)   │
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

## Running Predictions

### Local one-off prediction

```bash
python -m wc2026.cli predict --source github --force
```

This re-downloads all upstream data (`international_results`, GitHub fixtures, xG), rebuilds `data/fixtures.csv`, retrains models with the fresh xG features, and runs inference.

### Understanding the output

```
>>> Predicted outcome: France (43.0%)
>>> Predicted outcome: Argentina (40.1%)
 fixture_id       date       round                home_team  away_team  p_home_win  p_draw  p_away_win  exp_home_goals  exp_away_goals      score1         score2         score3
        101  2026-07-14  Semi-finals              France      Spain       0.4129   0.2684     0.3187            1.35            1.88  1-1 (10.9%)  1-2 (10.0%)  0-1 (8.8%)
        102  2026-07-15  Semi-finals             England  Argentina    0.2620   0.2230     0.5151            1.20            1.45  1-1 (12.8%)  0-1 (11.2%)  1-2 (9.0%)
        103  2026-07-18  Third Place Playoff      France     England     0.4303   0.2665     0.3032            1.45            1.10  1-1 (12.8%)  1-0 (12.3%)  0-0 (9.5%)
        104  2026-07-19  Final                   Spain   Argentina      0.2945   0.3043     0.4011            1.27            2.00  1-1 (10.6%)  1-2 (10.3%)  0-1 (9.1%)
```

**Columns:**
| Column | Description |
|--------|-------------|
| `p_home_win` / `p_draw` / `p_away_win` | Outcome probabilities from XGBClassifier |
| `exp_home_goals` / `exp_away_goals` | Expected goals from separate XGBRegressor per team |
| `score1–3` | Top 3 most likely exact scorelines (Poisson matrix, normalised) |

### Why expected goals ≠ win probability

France has a *higher* win probability (41.3%) than Spain (31.9%) yet a *lower* expected goals (1.35 vs 1.88). This is normal — they are separate models:

- **Win probability** = how often this team wins across all possible outcomes (including draws and penalty shootouts in aggregate)
- **Expected goals** = the average number of goals this team scores per match

France can win more often by scoring efficiently in tight 1-0 wins and grinding out 0-0 draws that go to penalties. Spain can outshoot France and average more goals overall while losing more of those close games. The most likely scoreline for France vs Spain is 1-1 (10.9%) — a draw — which contributes to France's win probability coming from lower-scoring wins rather than shootouts.

### Workflow / CI (GitHub Actions)

The `.github/workflows/wc2026_update.yml` runs daily at 06:00 UTC and:

1. Refreshes `international_results` and `matches_detailed.csv`
2. Retrains models with fresh xG features (`include_xg=True`)
3. Re-generates `outputs/wc2026_predictions.csv`
4. Commits updated data, model artifacts, and predictions

To trigger manually: go to the Actions tab → "WC 2026 — Auto Update Predictions" → "Run workflow".

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

The model produces two independent outputs for each match:

1. **Outcome probabilities** from an XGBClassifier (home win / draw / away win) — learns from all features simultaneously: Elo ratings, form, H2H, and xG rolling averages
2. **Expected goals** from separate XGBRegressors — directly predicts goals scored per team, then converts to a Poisson scoreline probability matrix

These are trained separately, so a team can have higher win probability but lower expected goals, and both can be simultaneously correct. See the explanation in *Running Predictions → Understanding the output*.

### Probabilities Don't Sum to Win%

The probability of "France winning" (41.3%) doesn't mean France scores more goals than Spain. It means:

- **P(Fx wins)** = probability of any scoreline where France has more goals (e.g., 1-0, 2-0, 2-1, etc.)
- **P(Draw)** = probability of equal goals (0-0, 1-1, 2-2, etc.)
- **P(Spain wins)** = probability of any scoreline where Spain has more goals

The most likely scoreline (1-1 at 10.9%) is actually a draw — France's win probability comes from winning the marginal distribution. This does not conflict with Spain having higher expected goals.

## Extending

- Add more xG sources (xgclient, TheStatsAPI) in `xg_features.py`
- Switch goal models to `reg:poisson` for Poisson goal distributions
- Expand `team_names.py` aliases as you discover mapping gaps
- Calibrate probabilities using isotonic regression
- Adjust the daily cron in `.github/workflows/wc2026_update.yml` if match data refreshes at different times

## API-Football (Optional)

If you want to use API-Football for live fixtures:

1. Get an API key at [api-football.com](https://www.api-football.com/)
2. Add to `.env`:
   ```
   APIFOOTBALL_KEY=your_key_here
   ```
3. World Cup uses `league=1`, `season=2026`

Never commit API keys.