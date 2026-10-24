# Smart Oil Consumption Monitoring Module 

A Python web application for recording oil consumption, reviewing category trends, forecasting future consumption, flagging unusually high daily totals, tracking a reduction target, and exporting CSV/PDF reports.

## Technology stack

- **UI:** Streamlit
- **Storage:** SQLite
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn `Ridge` regression, persisted with joblib
- **Charts:** Plotly
- **PDF:** fpdf2
- **Tests:** pytest

## Features

- Account registration and sign-in with salted PBKDF2-HMAC-SHA256 password hashes.
- Parameterized SQL queries and user-scoped record operations.
- Manual records with category, date, liters, cost, note, and source label.
- Clearly labelled synthetic demo records.
- Dashboard with category totals and daily totals on dates with observations.
- Chronological train/holdout split; MAE and R² plus comparison to a rolling-mean baseline.
- Model fingerprint checking so changed data is not silently forecast using a stale model.
- Spike checks with a separate insufficient-data state.
- Recent-versus-prior usage summaries and reduction-goal comparison.
- CSV and PDF exports.
- Unit tests for core analytics behavior.

## Quick start (Windows / VS Code)

Use Python 3.11 or 3.12 if available. In VS Code, open this project folder and run the commands below in the terminal.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

If PowerShell blocks activation, use the terminal's Command Prompt profile and run:

```bat
.venv\Scripts\activate.bat
```

If `py` is unavailable, install Python from https://www.python.org/downloads/ and select **Add Python to PATH** during installation. Then reopen VS Code.

## First use

1. Open the local URL printed by Streamlit (usually `http://localhost:8501`).
2. Create an account using a password of at least 10 characters.
3. Optionally click **Load demo data** in the sidebar. It only inserts demo data when that account has no records.
4. Explore the dashboard and train a model in the Forecasting tab.
5. Add records over time to test goal tracking and alerts.

## Data files and configuration

- SQLite database defaults to `oil_monitor.db` in the working directory.
- Set environment variable `OIL_MONITOR_DB` to change the database path.
- Saved models are stored under `models/` and are scoped by numeric user ID and category.
- Keep the database and model directory in persistent storage when deploying.
- Do not commit personal consumption records, database files, or model files to a public repository.

## Important deployment/security notes

This is a final-year-project reference implementation, not a security-audited production service. Passwords are hashed, but production deployments should use HTTPS, secrets management, rate limiting, stronger session controls, backups, and managed persistent storage. SQLite is suitable for a single-instance demonstration but may not be appropriate for high-write, multi-instance hosting. Streamlit Community Cloud's local filesystem should not be assumed durable across restarts/redeployments.

Never load joblib files from untrusted sources: deserialization can execute code. The app loads only its locally generated model files. Back up the database before migrations or manual changes.

## ML interpretation and evaluation

- The app aggregates multiple entries on a date into a daily total.
- Unlogged dates are missing observations, **not automatically zero consumption**.
- Model evaluation holds out later observations chronologically and compares MAE to a rolling-mean baseline.
- Forecasts are point estimates, not guaranteed outcomes or formal prediction intervals.
- Synthetic demo data is patterned and must not be used as evidence of real-world accuracy.
- An alert indicates an unusual recorded value under the configured rule; it does not prove a leak or equipment fault.

For a defensible report, use a documented real dataset, report data provenance, perform rolling-origin evaluation, and compare results to a baseline. For anomaly-detection metrics such as precision/recall, you need labelled normal/anomalous events.

## Current scope / limitations

The system does **not** read physical sensors, measure tank levels, connect to vehicle telemetry, or automatically diagnose the cause of high consumption. These require additional hardware/data integrations. It is a software-only monitoring and decision-support application.

## Suggested project report chapters

1. Abstract and problem statement
2. Objectives and scope
3. Related work / literature review
4. Requirements and feasibility analysis
5. System architecture and database design
6. Methodology and feature engineering
7. Forecasting algorithms and evaluation metrics
8. Implementation and screenshots
9. Testing and results
10. Limitations, ethics, security, and future work
11. Conclusion and references

## Suggested demo sequence

1. Register a test account.
2. Load demo data and explain that it is synthetic.
3. Show dashboard and category filtering.
4. Train a model and explain chronological holdout vs baseline.
5. Generate a forecast; add a record and show stale-model protection.
6. Show spike/insufficient-data states and reduction-goal requirements.
7. Export CSV and PDF.
8. Run `python -m pytest -q` and discuss limitations.
# Smart-Oil-Monitoring-Module
