"""Production models, walk-forward validation, and weekly prediction.

Walk-forward testing showed that a regularized linear model on a curated
feature set is more reliable than a large gradient-boosted model for weekly
kicker scoring. Team defense has a stronger signal and benefits from the full
rolling feature set with Ridge regularization.
"""

from __future__ import annotations

from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODEL_DIR, ensure_dirs

RANDOM_STATE = 42
Z90 = 1.2815515655446004

KICKER_ALPHA = 0.1
KICKER_L1 = 0.5
DST_ALPHA = 500.0

# Shrink the kicker model toward the market-implied team total. This was more
# stable in the 2026 early-season holdout without hurting 2021-2025.
KICKER_BLEND = 0.5
DST_BLEND = 1.0


def _clean_features(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    X = df.reindex(columns=list(features))
    return X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _mean_model(target: str) -> Pipeline:
    if target == "kicker_pts":
        estimator = ElasticNet(
            alpha=KICKER_ALPHA,
            l1_ratio=KICKER_L1,
            max_iter=20000,
            random_state=RANDOM_STATE,
        )
    elif target == "dst_pts":
        estimator = Ridge(alpha=DST_ALPHA, random_state=RANDOM_STATE)
    else:
        estimator = Ridge(alpha=50.0, random_state=RANDOM_STATE)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def _volatility_column(target: str) -> str:
    return "kicker_pts_std_last8" if target == "kicker_pts" else "dst_pts_std_last8"


def _boom_threshold(target: str) -> float:
    return 10.0 if target == "kicker_pts" else 12.0


def _vegas_model(
    train: pd.DataFrame, target: str
) -> tuple[LinearRegression | None, str, float]:
    feature = "team_implied" if target == "kicker_pts" else "opp_implied"
    frame = train[[feature, target]].dropna()
    if len(frame) < 20 or frame[feature].nunique() < 3:
        return None, feature, 0.0
    model = LinearRegression().fit(
        frame[[feature]].to_numpy(dtype=float), frame[target].to_numpy(dtype=float)
    )
    return model, feature, float(frame[feature].mean())


def _blend_weight(target: str) -> float:
    return KICKER_BLEND if target == "kicker_pts" else DST_BLEND


def train_models(
    train_df: pd.DataFrame,
    target: str,
    features: Iterable[str],
    season: int | None = None,
    week: int | None = None,
) -> dict[str, Any]:
    features = [c for c in dict.fromkeys(features)]
    train = train_df[train_df[target].notna()].copy()
    if train.empty:
        raise ValueError(f"no training rows for {target}")

    X = _clean_features(train, features)
    y = train[target].astype(float)
    mean_model = _mean_model(target).fit(X, y)
    model_fitted = mean_model.predict(X)

    vegas_model, vegas_feature, vegas_fill = _vegas_model(train, target)
    blend_weight = _blend_weight(target)
    if vegas_model is not None and blend_weight < 1.0:
        values = pd.to_numeric(train[vegas_feature], errors="coerce")
        filled = values.fillna(vegas_fill).to_numpy(dtype=float).reshape(-1, 1)
        vegas_fitted = vegas_model.predict(filled)
        fitted = blend_weight * model_fitted + (1.0 - blend_weight) * vegas_fitted
    else:
        fitted = model_fitted

    residual_std = float(np.std(y - fitted, ddof=1))
    residual_std = max(residual_std, 0.5)

    vol_column = _volatility_column(target)
    if vol_column in train.columns:
        vol = pd.to_numeric(train[vol_column], errors="coerce")
        vol_median = float(vol.median())
        if not np.isfinite(vol_median) or vol_median <= 0:
            vol_median = 1.0
    else:
        vol_median = 1.0

    return {
        "features": features,
        "target": target,
        "model": mean_model,
        "vegas_model": vegas_model,
        "vegas_feature": vegas_feature,
        "vegas_fill": vegas_fill,
        "blend_weight": blend_weight,
        "residual_std": residual_std,
        "vol_column": vol_column,
        "vol_median": vol_median,
        "boom_threshold": _boom_threshold(target),
        "train_rows": int(len(train)),
        "train_seasons": (int(train["season"].min()), int(train["season"].max())),
        "season": season,
        "week": week,
    }


def predict_models(models: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    X = _clean_features(df, models["features"])
    model_pred = models["model"].predict(X)
    mean = model_pred

    vegas_model = models.get("vegas_model")
    blend_weight = float(models.get("blend_weight", 1.0))
    if vegas_model is not None and blend_weight < 1.0:
        feature = models["vegas_feature"]
        values = pd.to_numeric(df.get(feature), errors="coerce")
        filled = values.fillna(models.get("vegas_fill", 0.0)).to_numpy(dtype=float)
        market = vegas_model.predict(filled.reshape(-1, 1))
        market = np.where(values.isna().to_numpy(), model_pred, market)
        mean = blend_weight * model_pred + (1.0 - blend_weight) * market

    out = pd.DataFrame({"proj_mean": mean, "model_mean": model_pred}, index=df.index)

    vol_column = models.get("vol_column")
    vol_median = float(models.get("vol_median", 1.0) or 1.0)
    if vol_column and vol_column in df.columns:
        vol = pd.to_numeric(df[vol_column], errors="coerce")
    else:
        vol = pd.Series(np.nan, index=df.index)
    vol_scale = (vol / vol_median).clip(0.6, 1.8).fillna(1.0)
    sigma = models["residual_std"] * vol_scale

    out["proj_sigma"] = sigma
    out["proj_p10"] = out["proj_mean"] - Z90 * sigma
    out["proj_p90"] = out["proj_mean"] + Z90 * sigma
    out["boom_prob"] = norm.sf((models["boom_threshold"] - out["proj_mean"]) / sigma)
    # A light ceiling tilt. The mean projection remains the primary estimate.
    upside = (out["proj_p90"] - out["proj_mean"]).clip(lower=0)
    out["stream_score"] = out["proj_mean"] + 0.12 * upside + 1.0 * out["boom_prob"]
    return out


def add_prediction_columns(
    pred: pd.DataFrame, models: dict[str, Any], target: str
) -> pd.DataFrame:
    out = pred.copy()
    model_pred = predict_models(models, out)
    for column in model_pred.columns:
        out[column] = model_pred[column]
    out["prediction"] = out["proj_mean"]
    out["_target"] = target
    return out


def _safe_corr(func, x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return np.nan
    return float(func(x[mask], y[mask])[0])


def ranking_metrics(pred: pd.DataFrame, target: str, k: int) -> dict[str, float]:
    rows = []
    for (_, _), group in pred.groupby(["season", "week"], sort=False):
        group = group.dropna(subset=[target, "prediction"])
        if len(group) < k:
            continue
        ordered = group.sort_values("prediction", ascending=False)
        actual = group[target].to_numpy(dtype=float)
        topk = ordered.head(k)[target].mean()
        oracle = np.sort(actual)[-k:].mean()
        league = actual.mean()
        predicted_ids = set(ordered.head(k).index)
        actual_ids = set(group.nlargest(k, target).index)
        rows.append(
            {
                "topk_actual": topk,
                "league_avg": league,
                "oracle_topk": oracle,
                "hit_rate": len(predicted_ids & actual_ids) / k,
                "gain_vs_league": topk - league,
            }
        )
    if not rows:
        return {
            f"top{k}_actual": np.nan,
            f"top{k}_hit": np.nan,
            f"top{k}_gain": np.nan,
            f"top{k}_oracle": np.nan,
        }
    frame = pd.DataFrame(rows)
    return {
        f"top{k}_actual": float(frame["topk_actual"].mean()),
        f"top{k}_hit": float(frame["hit_rate"].mean()),
        f"top{k}_gain": float(frame["gain_vs_league"].mean()),
        f"top{k}_oracle": float(frame["oracle_topk"].mean()),
    }


def score_predictions(pred: pd.DataFrame, target: str) -> dict[str, float]:
    pred = pred.dropna(subset=[target, "prediction"]).copy()
    if pred.empty:
        return {}
    y = pred[target].to_numpy(dtype=float)
    p = pred["prediction"].to_numpy(dtype=float)
    metrics = {
        "n": int(len(pred)),
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "spearman": _safe_corr(spearmanr, p, y),
        "pearson": _safe_corr(pearsonr, p, y),
        "bias": float(np.mean(p - y)),
    }
    for k in (1, 3, 5, 10):
        metrics.update(ranking_metrics(pred, target, k))
    return metrics


def _baseline_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    kind: str,
) -> pd.Series:
    """Simple benchmarks used to verify the model adds signal."""
    if kind == "vegas":
        feature = "team_implied" if target == "kicker_pts" else "opp_implied"
        tr = train[[feature, target]].dropna()
        te = test[[feature]].copy()
        if len(tr) < 20 or te[feature].notna().sum() == 0:
            return pd.Series(np.nan, index=test.index)
        regression = LinearRegression().fit(
            tr[[feature]].to_numpy(dtype=float), tr[target].to_numpy(dtype=float)
        )
        values = te[feature].to_numpy(dtype=float)
        filled = np.nan_to_num(values, nan=float(tr[feature].mean())).reshape(-1, 1)
        pred = pd.Series(regression.predict(filled), index=test.index)
        pred[te[feature].isna()] = np.nan
        return pred

    if kind == "form":
        column = (
            "kicker_kicker_pts_last4" if target == "kicker_pts" else "dst_pts_last4"
        )
        if column not in test.columns:
            return pd.Series(np.nan, index=test.index)
        return pd.to_numeric(test[column], errors="coerce")

    if kind == "vegas_plus_form":
        vegas = _baseline_predictions(train, test, target, "vegas")
        form = _baseline_predictions(train, test, target, "form")
        fill_v = float(vegas.mean()) if vegas.notna().any() else 0.0
        fill_f = float(form.mean()) if form.notna().any() else 0.0
        return 0.6 * vegas.fillna(fill_v) + 0.4 * form.fillna(fill_f)

    raise ValueError(f"unknown baseline kind: {kind}")


def walk_forward(
    df: pd.DataFrame,
    target: str,
    features: Iterable[str],
    test_seasons: Iterable[int],
) -> tuple[pd.DataFrame, dict[str, float]]:
    frames = []
    for season in sorted(int(s) for s in test_seasons):
        train = df[(df["season"] < season) & df[target].notna()].copy()
        test = df[(df["season"] == season) & df[target].notna()].copy()
        if train.empty or test.empty:
            continue
        models = train_models(train, target, features)
        scored = add_prediction_columns(test, models, target)
        frames.append(scored)
    if not frames:
        return pd.DataFrame(), {}
    pred = pd.concat(frames, ignore_index=True)
    return pred, score_predictions(pred, target)


def within_season_backtest(
    df: pd.DataFrame,
    target: str,
    features: Iterable[str],
    season: int,
    weeks: Iterable[int],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Train before each week and predict that week, for a realistic weekly run."""
    frames = []
    for week in sorted(int(w) for w in weeks):
        train = df[
            (df["season"] < season) | ((df["season"] == season) & (df["week"] < week))
        ]
        train = train[train[target].notna()]
        test = df[
            (df["season"] == season) & (df["week"] == week) & df[target].notna()
        ].copy()
        if train.empty or test.empty:
            continue
        models = train_models(train, target, features)
        scored = add_prediction_columns(test, models, target)
        frames.append(scored)
    if not frames:
        return pd.DataFrame(), {}
    pred = pd.concat(frames, ignore_index=True)
    return pred, score_predictions(pred, target)


def baseline_scores(
    df: pd.DataFrame,
    target: str,
    test_seasons: Iterable[int],
    kinds: Iterable[str] = ("vegas", "form", "vegas_plus_form"),
) -> pd.DataFrame:
    rows = []
    for kind in kinds:
        frames = []
        for season in sorted(int(s) for s in test_seasons):
            train = df[(df["season"] < season) & df[target].notna()].copy()
            test = df[(df["season"] == season) & df[target].notna()].copy()
            if train.empty or test.empty:
                continue
            test = test.copy()
            test["prediction"] = _baseline_predictions(train, test, target, kind)
            frames.append(test)
        if not frames:
            continue
        pred = pd.concat(frames, ignore_index=True)
        metrics = score_predictions(pred, target)
        metrics["baseline"] = kind
        metrics["target"] = target
        rows.append(metrics)
    return pd.DataFrame(rows)


def feature_importance(models: dict[str, Any], top_n: int = 40) -> pd.DataFrame:
    estimator = models["model"].named_steps["model"]
    if not hasattr(estimator, "coef_"):
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])
    importance = pd.DataFrame(
        {
            "feature": models["features"],
            "coefficient": np.asarray(estimator.coef_).ravel(),
        }
    )
    importance["abs_coefficient"] = importance["coefficient"].abs()
    return (
        importance.sort_values("abs_coefficient", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def train_final(
    df: pd.DataFrame,
    target: str,
    features: Iterable[str],
    season: int,
    week: int,
) -> dict[str, Any]:
    train = df[
        (df["season"] < season) | ((df["season"] == season) & (df["week"] < week))
    ]
    train = train[train[target].notna()].copy()
    return train_models(train, target, features, season=season, week=week)


def save_models(models: dict[str, Any], name: str) -> str:
    ensure_dirs()
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(models, path)
    return str(path)


def load_models(name: str) -> dict[str, Any]:
    return joblib.load(MODEL_DIR / f"{name}.joblib")
