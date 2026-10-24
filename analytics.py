"""Daily aggregation, time-series forecasting, anomaly detection and goals."""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

CATEGORIES = ["Household", "Vehicle", "Industrial"]
MODEL_DIR = Path("models")
MIN_TRAIN_DAYS = 14


def daily_series(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Aggregate multiple entries per date; unlogged dates remain missing, not zero."""
    if df.empty:
        return pd.DataFrame(columns=["date", "liters", "cost", "entry_count"])
    sub = df.loc[df["category"] == category, ["date", "liters", "cost"]].copy()
    if sub.empty:
        return pd.DataFrame(columns=["date", "liters", "cost", "entry_count"])
    sub["date"] = pd.to_datetime(sub["date"]).dt.normalize()
    out = sub.groupby("date", as_index=False).agg(liters=("liters", "sum"), cost=("cost", "sum"), entry_count=("liters", "size"))
    return out.sort_values("date").reset_index(drop=True)


def _origin_date(dates: pd.Series) -> pd.Timestamp:
    return pd.to_datetime(dates).min().normalize()


def make_features(dates, origin) -> np.ndarray:
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    origin = pd.Timestamp(origin).normalize()
    day_index = (dates.normalize() - origin).days.to_numpy(dtype=float)
    weekday = dates.dayofweek.to_numpy(dtype=float)
    day_of_year = dates.dayofyear.to_numpy(dtype=float)
    return np.column_stack([
        day_index,
        np.sin(2 * np.pi * weekday / 7),
        np.cos(2 * np.pi * weekday / 7),
        np.sin(2 * np.pi * day_of_year / 365.25),
        np.cos(2 * np.pi * day_of_year / 365.25),
    ])


def data_fingerprint(daily: pd.DataFrame) -> str:
    if daily.empty:
        return "empty"
    stable = daily[["date", "liters", "cost"]].copy()
    stable["date"] = stable["date"].astype(str)
    payload = stable.to_json(orient="records", double_precision=8)
    return hashlib.sha256(payload.encode()).hexdigest()


def _model_path(user_id: int, category: str) -> Path:
    safe_category = "".join(c for c in category if c.isalnum())
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR / f"model_u{int(user_id)}_{safe_category}.joblib"


def train_evaluate(df: pd.DataFrame, user_id: int, category: str) -> dict:
    """Chronological holdout; compare Ridge to a prior-7-observation mean baseline."""
    daily = daily_series(df, category)
    if len(daily) < MIN_TRAIN_DAYS:
        raise ValueError(f"At least {MIN_TRAIN_DAYS} distinct dated observations are required; found {len(daily)}.")
    origin = _origin_date(daily["date"])
    X = make_features(daily["date"], origin)
    y = daily["liters"].to_numpy(dtype=float)
    holdout_n = max(3, int(np.ceil(len(y) * 0.20)))
    if len(y) - holdout_n < 7:
        holdout_n = len(y) - 7
    split = len(y) - holdout_n
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    validation_model = Ridge(alpha=1.0).fit(X_train, y_train)
    pred = np.clip(validation_model.predict(X_test), 0, None)
    baseline = np.array([np.mean(y[max(0, i - 7):i]) for i in range(split, len(y))])
    model_mae = float(mean_absolute_error(y_test, pred))
    baseline_mae = float(mean_absolute_error(y_test, baseline))
    score = float(r2_score(y_test, pred)) if len(y_test) >= 2 else float("nan")

    final_model = Ridge(alpha=1.0).fit(X, y)
    bundle = {
        "model": final_model,
        "origin": origin.isoformat(),
        "trained_through": pd.Timestamp(daily["date"].max()).isoformat(),
        "sample_count": int(len(daily)),
        "fingerprint": data_fingerprint(daily),
        "category": category,
        "user_id": int(user_id),
        "model_version": 1,
        "features": ["day_index", "weekday_sin", "weekday_cos", "year_sin", "year_cos"],
    }
    joblib.dump(bundle, _model_path(user_id, category))
    return {
        "mae": model_mae,
        "baseline_mae": baseline_mae,
        "r2": score,
        "n_samples": int(len(daily)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "holdout_start": pd.Timestamp(daily.iloc[split]["date"]).date().isoformat(),
        "trained_through": pd.Timestamp(daily["date"].max()).date().isoformat(),
        "beats_baseline": model_mae < baseline_mae,
        "fingerprint": bundle["fingerprint"],
    }


def load_model_bundle(user_id: int, category: str):
    path = _model_path(user_id, category)
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
    except (OSError, ValueError, EOFError):
        return None
    if bundle.get("user_id") != int(user_id) or bundle.get("category") != category:
        return None
    return bundle


def forecast(df: pd.DataFrame, user_id: int, category: str, periods: int = 7):
    bundle = load_model_bundle(user_id, category)
    daily = daily_series(df, category)
    if bundle is None or daily.empty:
        return None, "No saved model. Train a model first."
    current_fp = data_fingerprint(daily)
    if current_fp != bundle.get("fingerprint"):
        return None, "The data changed after training. Retrain the model before using this forecast."
    last_date = pd.Timestamp(daily["date"].max()).normalize()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq="D")
    X_future = make_features(future_dates, pd.Timestamp(bundle["origin"]))
    preds = np.clip(bundle["model"].predict(X_future), 0, None)
    # Empirical residual range from the chronological holdout is deliberately not called a formal interval.
    result = pd.DataFrame({"date": future_dates, "predicted_liters": preds})
    result["predicted_liters"] = result["predicted_liters"].round(2)
    return result, None


def detect_spike(df: pd.DataFrame, category: str, threshold_pct: float = 25.0) -> dict:
    daily = daily_series(df, category)
    if len(daily) < 6:
        return {"status": "insufficient", "message": "At least 6 distinct dated observations are needed for a spike check."}
    latest = float(daily.iloc[-1]["liters"])
    prior = daily.iloc[-6:-1]["liters"].to_numpy(dtype=float)
    baseline = float(np.mean(prior))
    if baseline <= 0:
        return {"status": "insufficient", "message": "The historical baseline is zero; a percentage comparison is not meaningful."}
    pct = (latest - baseline) / baseline * 100
    # Require the percentage threshold and a robust deviation to reduce noisy single-point alerts.
    median = float(np.median(prior))
    mad = float(np.median(np.abs(prior - median)))
    robust_limit = median + max(3.0 * 1.4826 * mad, 0.05 * median)
    flagged = pct >= threshold_pct and latest > robust_limit
    return {
        "status": "alert" if flagged else "normal",
        "latest_date": pd.Timestamp(daily.iloc[-1]["date"]).date().isoformat(),
        "latest_liters": latest,
        "baseline_liters": baseline,
        "change_pct": float(pct),
        "message": f"Latest recorded daily total is {pct:.1f}% above the previous five observed dates' average." if flagged else f"No spike met both configured thresholds. Latest change: {pct:+.1f}% versus the previous five observed dates.",
    }


def trend_summary(df: pd.DataFrame, category: str) -> dict:
    daily = daily_series(df, category)
    if len(daily) < 6:
        return {"status": "insufficient", "message": "Log at least 6 distinct dates to compare recent and earlier usage."}
    n = min(7, len(daily) // 2)
    earlier = daily.iloc[-2 * n:-n]["liters"].mean()
    recent = daily.iloc[-n:]["liters"].mean()
    change = ((recent - earlier) / earlier * 100) if earlier > 0 else None
    return {"status": "ok", "average_liters": float(daily["liters"].mean()), "total_liters": float(daily["liters"].sum()),
            "peak_liters": float(daily["liters"].max()), "peak_date": pd.Timestamp(daily.loc[daily["liters"].idxmax(), "date"]).date().isoformat(),
            "recent_change_pct": None if change is None else float(change), "observed_days": int(len(daily))}


def goal_progress(df: pd.DataFrame, category: str, goal_pct: float) -> dict:
    """Compare most recent 7 calendar days with the preceding 7 calendar days; only complete observed days count."""
    daily = daily_series(df, category)
    if daily.empty:
        return {"status": "insufficient", "message": "No records exist for this category."}
    latest = pd.Timestamp(daily["date"].max()).normalize()
    current_start = latest - pd.Timedelta(days=6)
    baseline_start = current_start - pd.Timedelta(days=7)
    baseline_end = current_start - pd.Timedelta(days=1)
    current = daily[(daily["date"] >= current_start) & (daily["date"] <= latest)]
    baseline = daily[(daily["date"] >= baseline_start) & (daily["date"] <= baseline_end)]
    if len(current) < 4 or len(baseline) < 4:
        return {"status": "insufficient", "message": "Need at least 4 observed days in both the current 7-day window and the preceding 7-day baseline."}
    current_avg = float(current["liters"].mean())
    baseline_avg = float(baseline["liters"].mean())
    if baseline_avg <= 0:
        return {"status": "insufficient", "message": "Baseline average is zero; goal progress cannot be calculated."}
    reduction = (baseline_avg - current_avg) / baseline_avg * 100
    target_avg = baseline_avg * (1 - goal_pct / 100)
    return {"status": "ok", "baseline_average": baseline_avg, "current_average": current_avg,
            "target_average": target_avg, "reduction_pct": reduction, "goal_pct": float(goal_pct),
            "achieved": current_avg <= target_avg, "baseline_days": len(baseline), "current_days": len(current)}


def recommendations(category: str, spike: dict, trend: dict) -> list[str]:
    tips = {
        "Household": ["Check the fuel-use record against heating demand and occupancy before changing settings.", "Schedule maintenance for oil-burning appliances with a qualified technician.", "Review insulation and draughts if heating is the main oil use.", "Compare like-for-like periods; weather can change heating demand."],
        "Vehicle": ["Check the recorded quantity and odometer/distance data for this period.", "Avoid unnecessary idling and combine short trips when practical.", "Keep tires at the vehicle manufacturer's recommended pressure.", "Compare liters per 100 km when distance data is available."],
        "Industrial": ["Verify the meter reading and production volume for the flagged period.", "Inspect equipment and transfer lines using approved safety procedures.", "Track liters per unit of output, not just total liters.", "Review operating hours and planned maintenance records."],
    }
    result = []
    if spike.get("status") == "alert":
        result.append("A statistical spike was flagged; verify the entry and investigate operational context before concluding there is a leak or fault.")
    if trend.get("status") == "ok" and trend.get("recent_change_pct") is not None and trend["recent_change_pct"] > 10:
        result.append("Recent recorded usage is higher than the preceding observed period; compare workload, weather, and logging completeness.")
    result.extend(tips.get(category, []))
    return result[:4]
