"""Smart Oil Consumption Monitoring System — final-year project reference implementation."""
from __future__ import annotations

from datetime import date, datetime
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st
from fpdf import FPDF

from analytics import CATEGORIES, detect_spike, forecast, goal_progress, recommendations, train_evaluate, trend_summary
from db import authenticate, create_user, delete_usage, get_username, init_db, insert_usage, load_usage, seed_demo_data

st.set_page_config(page_title="Smart Oil Monitor", page_icon="🛢️", layout="wide")
init_db()


def pdf_report(username: str, data: pd.DataFrame, currency: str) -> bytes:
    """Create a compact PDF report. Core text is ASCII-normalized for built-in Helvetica."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Smart Oil Consumption Monitoring Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 7, f"User: {username}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 6, "Note: conclusions depend on the completeness and accuracy of manually logged data. Demo records are synthetic.")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Summary by category", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    if data.empty:
        pdf.cell(0, 7, "No usage records found.", new_x="LMARGIN", new_y="NEXT")
    else:
        for category in CATEGORIES:
            sub = data[data["category"] == category]
            liters = float(sub["liters"].sum())
            cost = float(sub["cost"].sum())
            pdf.cell(0, 7, f"{category}: {liters:.2f} L | {currency} {cost:.2f} | {len(sub)} entries", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Recent records", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=8)
        recent = data.sort_values("date", ascending=False).head(25)
        for _, row in recent.iterrows():
            d = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
            line = f"{d} | {row['category']} | {float(row['liters']):.2f} L | {currency} {float(row['cost']):.2f} | {row['source']}"
            pdf.multi_cell(0, 5, line.encode("latin-1", "replace").decode("latin-1"))
    return bytes(pdf.output())


def show_auth() -> None:
    st.title("🛢️ Smart Oil Consumption Monitor")
    st.caption("Consumption analytics, forecasting, anomaly alerts, and reporting")
    login_tab, register_tab = st.tabs(["Sign in", "Create account"])
    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            user_id = authenticate(username, password)
            if user_id is None:
                st.error("Invalid username or password.")
            else:
                st.session_state["user_id"] = user_id
                st.rerun()
    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username", key="register_username")
            new_password = st.text_input("Password (10+ characters)", type="password", key="register_password")
            confirm_password = st.text_input("Confirm password", type="password", key="register_confirm")
            create = st.form_submit_button("Create account", use_container_width=True)
        if create:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                ok, message = create_user(new_username, new_password)
                (st.success if ok else st.error)(message)
    st.info("For a demo, create your own account and click **Load demo data** in the sidebar after signing in. Demo records are labelled as synthetic.")


if "user_id" not in st.session_state:
    show_auth()
    st.stop()

user_id = int(st.session_state["user_id"])
try:
    username = get_username(user_id)
except ValueError:
    st.session_state.pop("user_id", None)
    st.rerun()

data = load_usage(user_id)

with st.sidebar:
    st.title("🛢️ Smart Oil Monitor")
    st.caption(f"Signed in as **{username}**")
    if st.button("Sign out", use_container_width=True):
        st.session_state.pop("user_id", None)
        st.rerun()
    st.divider()
    st.subheader("Log consumption")
    with st.form("usage_form", clear_on_submit=True):
        entry_date = st.date_input("Usage date", value=date.today(), max_value=date.today())
        entry_category = st.selectbox("Category", CATEGORIES)
        entry_liters = st.number_input("Liters used", min_value=0.0, max_value=1_000_000.0, value=0.0, step=0.1)
        currency = st.selectbox("Currency for cost display", ["BDT", "USD", "CAD", "EUR"], index=0)
        entry_cost = st.number_input(f"Cost ({currency})", min_value=0.0, max_value=1_000_000_000.0, value=0.0, step=10.0)
        entry_note = st.text_input("Note (optional)", max_chars=500)
        add_entry = st.form_submit_button("Add entry", use_container_width=True)
    if add_entry:
        try:
            insert_usage(user_id, entry_date, entry_category, entry_liters, entry_cost, entry_note)
            st.success("Entry saved.")
            st.rerun()
        except (ValueError, Exception) as exc:
            st.error(f"Could not save entry: {exc}")
    st.divider()
    if st.button("Load demo data", use_container_width=True, help="Adds synthetic records only if this account has no records."):
        try:
            count = seed_demo_data(user_id)
            st.success(f"Added {count} synthetic records." if count else "Demo data was not added because this account already has records.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not seed demo data: {exc}")
    st.caption("Demo rows are labelled `demo`. Do not present them as real measurements.")

# Reload after sidebar mutations / on every Streamlit rerun.
data = load_usage(user_id)
st.title("Smart Oil Consumption Monitoring System")
st.caption("Python · Streamlit · SQLite · Scikit-learn · Plotly · fpdf2")
if not data.empty and (data["source"] == "demo").any():
    st.warning("This account contains synthetic demo records. Filter or remove them before reporting real-world findings.")

tabs = st.tabs(["Dashboard", "Records", "Forecasting", "Alerts & Insights", "Goals", "PDF Report", "About"])

with tabs[0]:
    st.subheader("Consumption overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total logged", f"{data['liters'].sum():,.2f} L" if not data.empty else "0.00 L")
    col2.metric("Recorded entries", f"{len(data):,}")
    col3.metric("Total cost", f"{currency if 'currency' in locals() else 'BDT'} {(data['cost'].sum() if not data.empty else 0):,.2f}")
    col4.metric("Categories used", str(data["category"].nunique()) if not data.empty else "0")
    st.markdown("#### Total liters by category")
    category_totals = data.groupby("category", as_index=False)["liters"].sum() if not data.empty else pd.DataFrame(columns=["category", "liters"])
    if not category_totals.empty:
        fig = px.bar(category_totals, x="category", y="liters", color="category", title="Total logged consumption")
        fig.update_layout(showlegend=False, yaxis_title="Liters", xaxis_title="Category")
        st.plotly_chart(fig, use_container_width=True)
        daily_source = data.copy()
        daily_source["usage_date"] = pd.to_datetime(daily_source["date"]).dt.normalize()
        daily = daily_source.groupby(["usage_date", "category"], as_index=False)["liters"].sum()
        fig2 = px.line(daily, x="usage_date", y="liters", color="category", markers=True, title="Daily totals on recorded dates")
        fig2.update_layout(yaxis_title="Liters", xaxis_title="Date")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Only dates with records are plotted. A missing date is not assumed to mean zero consumption.")
    else:
        st.info("No data yet. Add an entry or load demo data from the sidebar.")

with tabs[1]:
    st.subheader("Usage records")
    if data.empty:
        st.info("No records to display.")
    else:
        filter_cat = st.multiselect("Filter categories", CATEGORIES, default=CATEGORIES)
        source_filter = st.multiselect("Record source", ["manual", "demo", "sensor_import"], default=["manual", "demo", "sensor_import"])
        filtered = data[data["category"].isin(filter_cat) & data["source"].isin(source_filter)].copy()
        st.dataframe(filtered.sort_values(["date", "id"], ascending=False), use_container_width=True, hide_index=True)
        with st.expander("Delete an entry"):
            st.warning("Deletion is permanent. Select only an entry you own and intend to remove.")
            options = {f"#{int(r.id)} · {pd.Timestamp(r.date).date()} · {r.category} · {r.liters:.2f} L · {r.source}": int(r.id) for r in filtered.itertuples()}
            if options:
                selected_label = st.selectbox("Entry", list(options.keys()))
                confirm_delete = st.checkbox("I understand this deletion cannot be undone.")
                if st.button("Delete selected entry", disabled=not confirm_delete):
                    if delete_usage(user_id, options[selected_label]):
                        st.success("Entry deleted.")
                        st.rerun()
                    else:
                        st.error("Entry was not found or could not be deleted.")
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", csv, file_name="oil_usage_records.csv", mime="text/csv")

with tabs[2]:
    st.subheader("Forecast future consumption")
    st.write("Forecasting requires at least 14 distinct recorded dates for the selected category. Evaluation uses a chronological holdout rather than a random split.")
    forecast_category = st.selectbox("Forecast category", CATEGORIES, key="forecast_category")
    if st.button("Train and evaluate model", type="primary"):
        try:
            metrics = train_evaluate(data, user_id, forecast_category)
            st.session_state[f"metrics_{forecast_category}"] = metrics
            st.success("Model trained and saved. It will be marked stale if the underlying data changes.")
        except ValueError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"Training failed: {exc}")
    metrics = st.session_state.get(f"metrics_{forecast_category}")
    if metrics:
        a, b, c, d = st.columns(4)
        a.metric("Holdout MAE", f"{metrics['mae']:.2f} L")
        b.metric("Baseline MAE", f"{metrics['baseline_mae']:.2f} L")
        c.metric("Holdout R²", f"{metrics['r2']:.3f}" if pd.notna(metrics["r2"]) else "N/A")
        d.metric("Distinct dates", str(metrics["n_samples"]))
        st.caption(f"Holdout dates start {metrics['holdout_start']}; training uses {metrics['n_train']} earlier observations and tests on {metrics['n_test']} later observations.")
        if metrics["beats_baseline"]:
            st.success("The model had lower MAE than the simple baseline on this holdout. This is encouraging, not proof of future accuracy.")
        else:
            st.warning("The model did not outperform the baseline on this holdout. Consider using the baseline until more/better data is available.")
    days = st.slider("Forecast horizon (days)", 3, 30, 7)
    if st.button("Generate forecast"):
        result, error = forecast(data, user_id, forecast_category, days)
        if error:
            st.warning(error)
        else:
            st.session_state[f"forecast_{forecast_category}"] = result
    result = st.session_state.get(f"forecast_{forecast_category}")
    if result is not None:
        fig = px.line(result, x="date", y="predicted_liters", markers=True, title=f"{forecast_category}: forecast")
        fig.update_layout(yaxis_title="Predicted liters", xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.caption("Point forecasts only. They are estimates, not guarantees or formal prediction intervals. Validate with real historical data.")

with tabs[3]:
    st.subheader("Alerts and plain-language insights")
    alert_category = st.selectbox("Category to inspect", CATEGORIES, key="alert_category")
    spike = detect_spike(data, alert_category)
    trend = trend_summary(data, alert_category)
    if spike["status"] == "alert":
        st.error(spike["message"])
    elif spike["status"] == "normal":
        st.success(spike["message"])
    else:
        st.info(spike["message"])
    if trend["status"] == "ok":
        if trend["recent_change_pct"] is not None:
            st.write(f"Recent mean usage changed **{trend['recent_change_pct']:+.1f}%** compared with the preceding observed period.")
        st.write(f"Average on recorded dates: **{trend['average_liters']:.2f} L** · Total: **{trend['total_liters']:.2f} L** · Peak: **{trend['peak_liters']:.2f} L** on **{trend['peak_date']}**.")
        st.caption(f"Summary based on {trend['observed_days']} distinct dates. It describes recorded dates, not necessarily every calendar day.")
    else:
        st.info(trend["message"])
    st.markdown("#### Suggested checks")
    for tip in recommendations(alert_category, spike, trend):
        st.markdown(f"- {tip}")
    st.caption("A statistical alert does not establish the cause of a change. Verify readings and context before taking operational action.")

with tabs[4]:
    st.subheader("Consumption reduction goal")
    goal_category = st.selectbox("Goal category", CATEGORIES, key="goal_category")
    goal_pct = st.slider("Target reduction (%)", 0, 50, 10, key="goal_pct")
    progress = goal_progress(data, goal_category, goal_pct)
    if progress["status"] != "ok":
        st.info(progress["message"])
    else:
        x, y, z = st.columns(3)
        x.metric("Baseline average", f"{progress['baseline_average']:.2f} L / observed day")
        y.metric("Current average", f"{progress['current_average']:.2f} L / observed day")
        z.metric("Actual change", f"{-progress['reduction_pct']:+.1f}%" if False else f"{progress['reduction_pct']:+.1f}% reduction")
        st.write(f"Target average: **{progress['target_average']:.2f} L per observed day** for a {goal_pct}% reduction.")
        if progress["achieved"]:
            st.success("Goal achieved for the current comparison window.")
        else:
            st.warning("Goal not yet achieved for the current comparison window.")
        st.caption(f"Compares observed dates in the latest 7 calendar days ({progress['current_days']} records/dates) with the preceding 7 calendar days ({progress['baseline_days']} observed dates). Missing dates are not treated as zero.")

with tabs[5]:
    st.subheader("Export a PDF report")
    st.write("The report summarizes logged totals and recent entries. It does not certify the accuracy of records or forecasts.")
    report_currency = st.selectbox("Report currency label", ["BDT", "USD", "CAD", "EUR"], index=0, key="report_currency")
    if st.button("Generate report"):
        try:
            st.session_state["pdf_report_bytes"] = pdf_report(username, data, report_currency)
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")
    if st.session_state.get("pdf_report_bytes"):
        st.download_button("Download PDF report", st.session_state["pdf_report_bytes"], file_name=f"oil_report_{username}.pdf", mime="application/pdf")

with tabs[6]:
    st.subheader("Project scope and limitations")
    st.markdown("""
    **Purpose:** record oil consumption, summarize category usage and cost, forecast short-term consumption, flag unusually high recorded daily totals, and compare recent consumption against a reduction target.

    **Implemented:** account registration and sign-in; password hashing; per-user database filtering; SQLite persistence; validated entry functions; daily aggregation; chronological ML holdout; baseline comparison; saved model fingerprints; spike detection with an insufficient-data state; reduction-goal tracking; CSV export; PDF reports; demo-data labelling.

    **Not implemented:** physical flow-meter/tank sensors, vehicle telemetry, automatic leak diagnosis, formal prediction intervals, production-grade external identity management, multi-instance database hosting, or independent real-world validation.

    **Responsible interpretation:** predictions and alerts are decision-support estimates. Do not use them as safety alarms or as proof of a leak. Verify records and use qualified professionals for equipment concerns.
    """)
    st.markdown("#### Suggested final-year evaluation")
    st.markdown("1. Collect or obtain a documented real dataset with permission.\n2. Separate real and synthetic data.\n3. Evaluate forecasts with a chronological holdout and rolling-origin validation.\n4. Compare MAE against a naive/rolling-average baseline.\n5. Measure anomaly precision/recall only if labelled anomalies exist.\n6. Test user isolation, validation, deletion, and report generation.\n7. Document dataset limitations, ethical considerations, and reproducible setup steps.")
