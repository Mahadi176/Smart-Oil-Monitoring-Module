# Project summary for presentation

## Proposed title
Smart Oil Consumption Monitoring and Forecasting System Using Python and Machine Learning

## Problem
Manual oil-use records are difficult to summarize consistently. Users may miss unusual changes, struggle to estimate near-term consumption, and lack a repeatable way to compare reduction targets.

## Objectives
1. Record categorized oil consumption and cost.
2. Store records in a persistent relational database.
3. Summarize daily and category-level consumption.
4. Forecast short-term consumption from historical records.
5. Flag unusually high recorded daily totals.
6. Compare recent observed consumption against a configurable reduction target.
7. Export data and summary reports.

## Methodology
Records are stored in SQLite and aggregated into daily totals by category. A Ridge regression model uses a time index and cyclical calendar features. Evaluation reserves later observations as a chronological holdout and compares model MAE with a simple recent-mean baseline. A separate rule-based anomaly check compares the latest daily total with recent observed dates. The app shows insufficient-data states rather than implying that missing evidence means normal behavior.

## Evaluation plan
- Functional tests: account creation, login, record validation, CRUD, user isolation, CSV/PDF export.
- Forecasting: MAE, R², baseline MAE, chronological holdout, and rolling-origin tests on a real dataset.
- Anomaly alerts: use labelled anomalies if available; report precision/recall and false alarms.
- Usability: task-based tests with representative users.
- Limitations: synthetic data is for demonstration only; no physical sensors or causal fault diagnosis are included.
