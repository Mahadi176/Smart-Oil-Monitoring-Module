from datetime import datetime, timedelta

import pandas as pd
import pytest

from analytics import daily_series, detect_spike, goal_progress, train_evaluate


def sample_data(days=40, spike=False):
    start = datetime(2025, 1, 1)
    rows = []
    for i in range(days):
        liters = 10.0
        if spike and i == days - 1:
            liters = 20.0
        rows.append({"date": start + timedelta(days=i), "category": "Vehicle", "liters": liters, "cost": liters * 2})
    return pd.DataFrame(rows)


def test_daily_series_aggregates_multiple_entries_same_day():
    df = pd.DataFrame([
        {"date": datetime(2025, 1, 1), "category": "Vehicle", "liters": 3, "cost": 6},
        {"date": datetime(2025, 1, 1), "category": "Vehicle", "liters": 4, "cost": 8},
    ])
    out = daily_series(df, "Vehicle")
    assert len(out) == 1
    assert out.iloc[0]["liters"] == 7
    assert out.iloc[0]["cost"] == 14


def test_spike_is_flagged_for_large_increase():
    result = detect_spike(sample_data(spike=True), "Vehicle")
    assert result["status"] == "alert"


def test_spike_reports_insufficient_data():
    result = detect_spike(sample_data(days=4), "Vehicle")
    assert result["status"] == "insufficient"


def test_training_requires_enough_observations():
    with pytest.raises(ValueError):
        train_evaluate(sample_data(days=8), 1, "Vehicle")


def test_training_uses_chronological_holdout_and_saves_model(tmp_path, monkeypatch):
    import analytics
    monkeypatch.setattr(analytics, "MODEL_DIR", tmp_path)
    result = train_evaluate(sample_data(days=40), 123, "Vehicle")
    assert result["n_train"] < result["n_samples"]
    assert result["n_test"] >= 3
    assert "baseline_mae" in result
    assert list(tmp_path.glob("*.joblib"))


def test_goal_requires_enough_observed_days():
    result = goal_progress(sample_data(days=5), "Vehicle", 10)
    assert result["status"] == "insufficient"
