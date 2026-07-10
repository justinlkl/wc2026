"""XGBoost training, evaluation, and model persistence."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

from wc2026.config import (
    FEATURE_COLS,
    MODELS_DIR,
    RESULT_LABELS,
    TRAIN_END,
    VAL_END,
)
from wc2026.data_loading import load_results
from wc2026.features import build_modeling_dataset, feature_matrix

logger = logging.getLogger(__name__)


@dataclass
class TrainMetrics:
    split: str
    accuracy: float
    log_loss: float
    baseline_log_loss: float
    home_mae: float
    away_mae: float


def _chronological_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["date"] <= pd.Timestamp(TRAIN_END)]
    val = df[(df["date"] > pd.Timestamp(TRAIN_END)) & (df["date"] <= pd.Timestamp(VAL_END))]
    test = df[df["date"] > pd.Timestamp(VAL_END)]
    return train, val, test


def _baseline_log_loss(y: np.ndarray) -> float:
    """Log-loss from predicting constant class frequencies."""
    counts = np.bincount(y, minlength=3)
    probs = counts / counts.sum()
    return log_loss(y, np.tile(probs, (len(y), 1)))


def _default_classifier() -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        max_depth=4,
        learning_rate=0.05,
        n_estimators=800,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )


def _default_regressor() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        max_depth=4,
        learning_rate=0.05,
        n_estimators=800,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )


def evaluate_split(
    clf: XGBClassifier,
    reg_home: XGBRegressor,
    reg_away: XGBRegressor,
    df: pd.DataFrame,
    split_name: str,
    feature_medians: dict[str, float],
) -> TrainMetrics:
    X = feature_matrix(df)
    y = df["result_encoded"].values

    probs = clf.predict_proba(X)
    acc = accuracy_score(y, probs.argmax(axis=1))
    ll = log_loss(y, probs)
    baseline = _baseline_log_loss(y)

    pred_home = reg_home.predict(X)
    pred_away = reg_away.predict(X)
    home_mae = mean_absolute_error(df["y_home_goals"], pred_home)
    away_mae = mean_absolute_error(df["y_away_goals"], pred_away)

    logger.info(
        "[%s] accuracy=%.3f log_loss=%.3f baseline=%.3f improvement=%.3f "
        "home_mae=%.2f away_mae=%.2f n=%d",
        split_name,
        acc,
        ll,
        baseline,
        baseline - ll,
        home_mae,
        away_mae,
        len(df),
    )

    return TrainMetrics(
        split=split_name,
        accuracy=acc,
        log_loss=ll,
        baseline_log_loss=baseline,
        home_mae=home_mae,
        away_mae=away_mae,
    )


def feature_importance(clf: XGBClassifier) -> pd.DataFrame:
    booster = clf.get_booster()
    importance = booster.get_score(importance_type="gain")
    df = (
        pd.Series(importance)
        .sort_values(ascending=False)
        .rename_axis("feature")
        .reset_index(name="gain")
    )
    return df


def train_models(
    matches: pd.DataFrame | None = None,
    models_dir: Path | None = None,
) -> dict:
    """Train outcome classifier and goal regressors; save artifacts."""
    models_dir = models_dir or MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    if matches is None:
        results = load_results()
        matches = build_modeling_dataset(results)

    train, val, test = _chronological_splits(matches)
    logger.info(
        "Splits — train=%d val=%d test=%d",
        len(train),
        len(val),
        len(test),
    )

    X_train = feature_matrix(train)
    y_train = train["result_encoded"].values
    X_val = feature_matrix(val) if len(val) > 0 else X_train
    y_val = val["result_encoded"].values if len(val) > 0 else y_train

    clf = _default_classifier()
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    reg_home = _default_regressor()
    reg_home.fit(
        X_train,
        train["y_home_goals"],
        eval_set=[(X_val, val["y_home_goals"] if len(val) > 0 else train["y_home_goals"])],
        verbose=False,
    )

    reg_away = _default_regressor()
    reg_away.fit(
        X_train,
        train["y_away_goals"],
        eval_set=[(X_val, val["y_away_goals"] if len(val) > 0 else train["y_away_goals"])],
        verbose=False,
    )

    medians = {col: float(X_train[col].median()) for col in FEATURE_COLS}

    metrics = []
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        if len(split_df) == 0:
            continue
        metrics.append(evaluate_split(clf, reg_home, reg_away, split_df, name, medians))

    imp = feature_importance(clf)
    logger.info("Top features:\n%s", imp.head(15).to_string(index=False))

    joblib.dump(clf, models_dir / "outcome_classifier.joblib")
    joblib.dump(reg_home, models_dir / "home_goals_regressor.joblib")
    joblib.dump(reg_away, models_dir / "away_goals_regressor.joblib")
    joblib.dump(medians, models_dir / "feature_medians.joblib")

    meta = {
        "feature_cols": FEATURE_COLS,
        "result_labels": RESULT_LABELS,
        "metrics": [
            {
                "split": m.split,
                "accuracy": m.accuracy,
                "log_loss": m.log_loss,
                "baseline_log_loss": m.baseline_log_loss,
                "home_mae": m.home_mae,
                "away_mae": m.away_mae,
            }
            for m in metrics
        ],
        "top_features": imp.head(20).to_dict(orient="records"),
    }
    with open(models_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "classifier": clf,
        "reg_home": reg_home,
        "reg_away": reg_away,
        "medians": medians,
        "metrics": metrics,
        "importance": imp,
        "matches": matches,
    }


def load_models(models_dir: Path | None = None) -> dict:
    models_dir = models_dir or MODELS_DIR
    return {
        "classifier": joblib.load(models_dir / "outcome_classifier.joblib"),
        "reg_home": joblib.load(models_dir / "home_goals_regressor.joblib"),
        "reg_away": joblib.load(models_dir / "away_goals_regressor.joblib"),
        "medians": joblib.load(models_dir / "feature_medians.joblib"),
    }


def predict_match(
    feature_row: dict,
    models: dict,
) -> dict:
    """Return outcome probabilities and expected goals for one match."""
    medians = models["medians"]
    row = {k: feature_row.get(k, medians.get(k, 0.0)) for k in FEATURE_COLS}
    for k, v in row.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            row[k] = medians.get(k, 0.0)

    X = pd.DataFrame([row])[FEATURE_COLS]
    probs = models["classifier"].predict_proba(X)[0]
    exp_home = float(models["reg_home"].predict(X)[0])
    exp_away = float(models["reg_away"].predict(X)[0])

    return {
        "p_home_win": probs[0],
        "p_draw": probs[1],
        "p_away_win": probs[2],
        "exp_home_goals": max(0.0, exp_home),
        "exp_away_goals": max(0.0, exp_away),
    }
