"""
ProjectPulse AI — SIH 2026 MVP
Single-file build: all model/utils logic is inlined below so the app has
zero local-module imports. This avoids GitHub upload / folder-structure
deployment errors (e.g. ModuleNotFoundError: No module named 'models').
Only external dependency files needed alongside this one: requirements.txt
and data/projects.csv.
"""

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "data" / "projects.csv"
DB_PATH = BASE / "data" / "projectpulse.db"

# ----------------------------------------------------------------------
# utils/helpers.py (inlined)
# ----------------------------------------------------------------------
def money(v):
    sign = "-" if v < 0 else ""
    return f"{sign}₹{abs(v):,.0f} Cr"


def apply_style():
    st.markdown(
        """
    <style>
    .block-container{padding-top:1.2rem;max-width:1450px}
    .hero{padding:1.2rem 1.4rem;border-radius:16px;background:linear-gradient(135deg,#102a43,#176b87);color:white;margin-bottom:1rem}
    .hero h1{margin:0;font-size:2rem}.hero p{margin:.35rem 0 0;opacity:.88}
    .kpi{padding:1rem;border:1px solid #dfe7ef;border-radius:14px;background:white;box-shadow:0 2px 10px rgba(0,0,0,.04)}
    .kpi .label{font-size:.82rem;color:#667085}.kpi .value{font-size:1.7rem;font-weight:750;color:#102a43}
    .alert{padding:1rem;border-radius:12px;border-left:5px solid #cbd5e1;background:#f8fafc;margin:.5rem 0}
    .alert.high{border-left-color:#dc2626}.alert.medium{border-left-color:#f59e0b}.alert.low{border-left-color:#16a34a}
    .small{font-size:.82rem;color:#667085}
    </style>
    """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# utils/data_loader.py (inlined)
# ----------------------------------------------------------------------
def load_projects():
    df = pd.read_csv(CSV_PATH)
    with sqlite3.connect(DB_PATH) as con:
        df.to_sql("projects", con, if_exists="replace", index=False)
    return df


# ----------------------------------------------------------------------
# models/feature_engineering.py (inlined)
# ----------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cost_variance"] = out["current_cost"] - out["planned_cost"]
    out["cost_variance_percent"] = np.where(
        out["planned_cost"] > 0, out["cost_variance"] / out["planned_cost"] * 100, 0
    )
    out["schedule_variance"] = out["elapsed_duration_months"] - (
        out["planned_duration_months"] * out["physical_progress"] / 100
    )
    out["schedule_variance_percent"] = np.where(
        out["planned_duration_months"] > 0,
        out["schedule_variance"] / out["planned_duration_months"] * 100,
        0,
    )
    out["progress_gap"] = out["expected_progress"] - out["physical_progress"]
    out["financial_progress_gap"] = out["financial_progress"] - out["physical_progress"]
    out["milestone_delay_ratio"] = np.where(
        out["milestones_total"] > 0, out["milestones_delayed"] / out["milestones_total"], 0
    )
    out["expenditure_ratio"] = np.where(
        out["planned_cost"] > 0, out["expenditure"] / out["planned_cost"], 0
    )
    out["delay_indicator"] = (out["progress_gap"] > 10).astype(int)
    out["issue_density"] = np.where(
        out["milestones_total"] > 0, out["issues_count"] / out["milestones_total"], 0
    )
    return out


# ----------------------------------------------------------------------
# models/risk_model.py (inlined)
# ----------------------------------------------------------------------
FEATURES = [
    "cost_variance_percent", "schedule_variance_percent", "progress_gap",
    "financial_progress", "expected_progress", "milestone_delay_ratio",
    "expenditure_ratio", "issues_count", "previous_delay", "contractor_rating",
]


def synthetic_label(row):
    score = (
        np.clip(row["cost_variance_percent"], 0, 35) * 1.6
        + np.clip(row["schedule_variance_percent"], 0, 35) * 1.2
        + np.clip(row["progress_gap"], 0, 40) * 1.4
        + row["milestone_delay_ratio"] * 35
        + min(row["issues_count"], 15) * 1.2
        + row["previous_delay"] * 7
        + max(0, 3.5 - row["contractor_rating"]) * 8
    )
    return "High" if score >= 75 else ("Medium" if score >= 38 else "Low")


def train_model(df):
    labels = df.apply(synthetic_label, axis=1)
    model = RandomForestClassifier(n_estimators=180, random_state=42, class_weight="balanced")
    model.fit(df[FEATURES], labels)
    return model


def model_predict(model, df):
    return model.predict(df[FEATURES])


def feature_importance(model):
    return dict(zip(FEATURES, model.feature_importances_))


# ----------------------------------------------------------------------
# models/prediction.py (inlined)
# ----------------------------------------------------------------------
def risk_scores(row):
    clip = lambda x: float(np.clip(x, 0, 100))
    cost = clip(max(0, row["cost_variance_percent"]) / 30 * 100)
    time = clip(max(0, row["schedule_variance_percent"]) / 30 * 100)
    progress = clip(max(0, row["progress_gap"]) / 30 * 100)
    issue = clip(
        row["milestone_delay_ratio"] * 100 * 0.65
        + min(row["issues_count"], 15) / 15 * 100 * 0.35
    )
    overall = cost * 0.35 + time * 0.35 + progress * 0.20 + issue * 0.10
    level = lambda v: "High" if v >= 70 else ("Medium" if v >= 40 else "Low")
    return {
        "cost_score": round(cost, 1),
        "time_score": round(time, 1),
        "progress_score": round(progress, 1),
        "issue_score": round(issue, 1),
        "overall_score": round(overall, 1),
        "cost_risk": level(cost),
        "time_risk": level(time),
        "overall_risk": level(overall),
    }


def top_drivers(row, scores):
    vals = {
        "Progress gap": max(0, row["progress_gap"]),
        "Cost variance": max(0, row["cost_variance_percent"]),
        "Milestone delays": row["milestone_delay_ratio"] * 100,
        "Schedule variance": max(0, row["schedule_variance_percent"]),
        "Issue count": row["issues_count"] * 3,
    }
    return [k for k, v in sorted(vals.items(), key=lambda x: x[1], reverse=True)[:4]]


# ----------------------------------------------------------------------
# utils/risk_engine.py (inlined)
# ----------------------------------------------------------------------
def generate_warnings(row, scores):
    warnings = []

    def add(sev, issue, evidence, action):
        warnings.append({"severity": sev, "issue": issue, "evidence": evidence, "action": action})

    if row["cost_variance_percent"] > 10:
        add(
            "HIGH" if row["cost_variance_percent"] > 18 else "MEDIUM",
            "Cost variance is above monitoring threshold.",
            f"Current cost is {row['cost_variance_percent']:.1f}% above plan.",
            "Review cost escalation and remaining budget.",
        )
    if row["progress_gap"] > 15:
        add(
            "HIGH" if row["progress_gap"] > 22 else "MEDIUM",
            "Physical progress is significantly below expected progress.",
            f"Progress gap is {row['progress_gap']:.1f} percentage points.",
            "Review implementation schedule and delayed activities.",
        )
    if row["milestone_delay_ratio"] > 0.30:
        add(
            "HIGH" if row["milestone_delay_ratio"] > 0.45 else "MEDIUM",
            "Multiple project milestones are delayed.",
            f"{int(row['milestones_delayed'])} of {int(row['milestones_total'])} milestones are delayed.",
            "Identify critical delayed milestones.",
        )
    if row["schedule_variance_percent"] > 15:
        add(
            "HIGH" if row["schedule_variance_percent"] > 25 else "MEDIUM",
            "Schedule deviation requires attention.",
            f"Schedule variance is {row['schedule_variance_percent']:.1f}%.",
            "Reassess remaining activities and completion timeline.",
        )
    if scores["overall_score"] >= 70:
        add(
            "HIGH",
            "High-risk project requires management attention.",
            f"Overall demonstration risk score is {scores['overall_score']:.1f}/100.",
            "Prioritize a management review of the leading risk drivers.",
        )
    if not warnings:
        add(
            "LOW",
            "No threshold breach detected.",
            "Current indicators remain within demo monitoring thresholds.",
            "Continue routine monitoring.",
        )
    return warnings


# ----------------------------------------------------------------------
# utils/recommendations.py (inlined)
# ----------------------------------------------------------------------
def recommendations(row, scores):
    rec = []
    if row["cost_variance_percent"] > 10:
        rec.append("Review cost escalation and remaining budget.")
    if row["progress_gap"] > 15:
        rec.append("Review implementation schedule and delayed activities.")
    if row["milestone_delay_ratio"] > 0.30:
        rec.append("Identify critical delayed milestones.")
    if row["issues_count"] >= 6:
        rec.append("Conduct an issue-resolution review.")
    if row["schedule_variance_percent"] > 15:
        rec.append("Reassess remaining activities and completion timeline.")
    return rec or ["Continue routine monitoring against planned milestones."]


# ----------------------------------------------------------------------
# Streamlit app
# ----------------------------------------------------------------------
st.set_page_config(page_title="ProjectPulse AI", page_icon="📊", layout="wide")
apply_style()


@st.cache_data
def get_data():
    return engineer_features(load_projects())


@st.cache_resource
def get_model():
    return train_model(get_data())


df = get_data()
model = get_model()
df["ml_risk"] = model_predict(model, df)

for k in ["cost_score", "time_score", "progress_score", "issue_score", "overall_score",
          "cost_risk", "time_risk", "overall_risk"]:
    df[k] = [risk_scores(r)[k] for _, r in df.iterrows()]
df["drivers"] = df.apply(lambda r: top_drivers(r, risk_scores(r)), axis=1)

st.sidebar.markdown("## 📊 ProjectPulse AI")
st.sidebar.caption("SIH 2026 • SIH26103")
st.sidebar.info("Demo Environment — Synthetic Project Data")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Projects", "Risk Analysis", "Early Warnings", "Analytics", "AI Assistant", "About"],
)

st.markdown(
    '<div class="hero"><h1>ProjectPulse AI</h1>'
    "<p>Integrated project monitoring • predictive risk • transparent early warnings</p></div>",
    unsafe_allow_html=True,
)
st.caption(
    "⚠️ Demonstration model trained on synthetic data. This is not an official government "
    "risk assessment and does not represent official MoSPI data."
)


def kpi(label, value):
    st.markdown(
        f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )


if page == "Overview":
    counts = df["overall_risk"].value_counts()
    cols = st.columns(6)
    vals = [
        ("Total Projects", len(df)), ("High Risk", int(counts.get("High", 0))),
        ("Medium Risk", int(counts.get("Medium", 0))), ("Low Risk", int(counts.get("Low", 0))),
        ("Average Project Health", f"{100 - df.overall_score.mean():.1f}%"),
        ("Projects With Alerts", int(sum(len(generate_warnings(r, risk_scores(r))) > 0 for _, r in df.iterrows()))),
    ]
    for col, (label, val) in zip(cols, vals):
        with col:
            kpi(label, val)

    a, b = st.columns(2)
    with a:
        st.plotly_chart(px.pie(df, names="overall_risk", title="Risk distribution", hole=.45), width="stretch")
    with b:
        tmp = df.sort_values("overall_score", ascending=False).head(12).copy()
        tmp["project"] = tmp.project_name.str.slice(0, 28)
        st.plotly_chart(
            px.bar(tmp, x="overall_score", y="project", orientation="h",
                   title="Highest demonstration risk scores", text="overall_score"),
            width="stretch",
        )
    c, d, e = st.columns(3)
    with c:
        st.plotly_chart(px.histogram(df, x="cost_variance_percent", title="Cost variance %", nbins=12), width="stretch")
    with d:
        st.plotly_chart(px.histogram(df, x="schedule_variance_percent", title="Schedule variance %", nbins=12), width="stretch")
    with e:
        st.plotly_chart(
            px.scatter(df, x="expected_progress", y="physical_progress", color="overall_risk",
                       hover_name="project_name", title="Physical vs expected progress"),
            width="stretch",
        )

elif page == "Projects":
    st.subheader("Project Portfolio")
    c1, c2, c3, c4 = st.columns(4)
    states = c1.multiselect("State", sorted(df.state.unique()))
    sectors = c2.multiselect("Sector", sorted(df.sector.unique()))
    risks = c3.multiselect("Risk level", ["High", "Medium", "Low"])
    statuses = c4.multiselect("Project status", sorted(df.project_status.unique()))
    view = df.copy()
    if states:
        view = view[view.state.isin(states)]
    if sectors:
        view = view[view.sector.isin(sectors)]
    if risks:
        view = view[view.overall_risk.isin(risks)]
    if statuses:
        view = view[view.project_status.isin(statuses)]
    st.dataframe(
        view[["project_id", "project_name", "sector", "state", "cost_risk", "time_risk",
              "overall_risk", "physical_progress", "project_status"]].rename(columns={
            "project_id": "Project ID", "project_name": "Project Name", "sector": "Sector", "state": "State",
            "cost_risk": "Cost Risk", "time_risk": "Time Risk", "overall_risk": "Overall Risk",
            "physical_progress": "Physical Progress", "project_status": "Status",
        }),
        width="stretch", hide_index=True,
    )
    options = view.project_id.tolist() or df.project_id.tolist()
    selected = st.selectbox("Open project detail", options)
    r = df[df.project_id == selected].iloc[0]
    s = risk_scores(r)
    st.markdown(f"### {r.project_name} · {r.project_id}")
    x1, x2, x3, x4 = st.columns(4)
    for col, label, val in [
        (x1, "Cost Risk", s["cost_risk"]), (x2, "Time Risk", s["time_risk"]),
        (x3, "Overall Risk", s["overall_risk"]), (x4, "Risk Score", f'{s["overall_score"]:.1f}/100'),
    ]:
        with col:
            kpi(label, val)
    a, b = st.columns(2)
    with a:
        st.markdown("#### Financial Information")
        st.write(f"Original cost: **{money(r.planned_cost)}**")
        st.write(f"Current cost: **{money(r.current_cost)}**")
        st.write(f"Variance: **{money(r.cost_variance)} ({r.cost_variance_percent:.1f}%)**")
        st.write(f"Expenditure: **{money(r.expenditure)}**")
        st.plotly_chart(px.bar(x=["Planned", "Current"], y=[r.planned_cost, r.current_cost], title="Cost comparison"), width="stretch")
    with b:
        st.markdown("#### Progress & Timeline")
        st.progress(int(r.physical_progress), text=f"Physical progress: {r.physical_progress:.1f}%")
        st.progress(int(r.expected_progress), text=f"Expected progress: {r.expected_progress:.1f}%")
        st.write(f"Planned duration: **{r.planned_duration_months:.0f} months** · Elapsed: **{r.elapsed_duration_months:.1f} months**")
        st.write(f"Schedule variance: **{r.schedule_variance_percent:.1f}%**")
        st.plotly_chart(px.bar(x=["Expected", "Actual"], y=[r.expected_progress, r.physical_progress], title="Progress comparison"), width="stretch")
    st.markdown("#### Top Risk Drivers")
    st.write(" • ".join(r.drivers))
    st.markdown("#### Early Warnings")
    for w in generate_warnings(r, s):
        st.markdown(
            f'<div class="alert {w["severity"].lower()}"><b>{w["severity"]}</b> · {w["issue"]}<br>'
            f'<span class="small">Evidence: {w["evidence"]}<br>Suggested action: {w["action"]}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("#### AI-generated Demonstration Recommendations")
    for x in recommendations(r, s):
        st.write("• " + x)
    st.caption("Recommendations are demonstration outputs, not official government instructions.")

elif page == "Risk Analysis":
    st.subheader("Risk Analysis")
    selected = st.selectbox("Select project", df.project_id)
    r = df[df.project_id == selected].iloc[0]
    s = risk_scores(r)
    st.metric("Overall Risk Score", f'{s["overall_score"]:.1f}/100', delta=s["overall_risk"])
    a, b = st.columns(2)
    with a:
        st.plotly_chart(
            px.bar(x=["Cost", "Time", "Progress", "Issue/Milestone"],
                   y=[s["cost_score"], s["time_score"], s["progress_score"], s["issue_score"]],
                   range_y=[0, 100], title="Risk contribution scores"),
            width="stretch",
        )
    with b:
        st.markdown("### Transparent weighting")
        st.write("Cost risk — **35%**")
        st.write("Time risk — **35%**")
        st.write("Progress risk — **20%**")
        st.write("Issue/Milestone risk — **10%**")
        st.info("Thresholds: 0–39 Low · 40–69 Medium · 70–100 High")
    imp = feature_importance(model)
    impdf = pd.DataFrame({"Feature": list(imp), "Importance": list(imp.values())}).sort_values("Importance", ascending=False)
    st.plotly_chart(px.bar(impdf.head(10), x="Importance", y="Feature", orientation="h", title="Random Forest feature importance"), width="stretch")
    st.caption("Feature importance is a model-level explanation, not causal inference.")

elif page == "Early Warnings":
    st.subheader("Early Warning Center")
    filt = st.multiselect("Severity", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"])
    alerts = []
    for _, r in df.iterrows():
        for w in generate_warnings(r, risk_scores(r)):
            if w["severity"] in filt:
                alerts.append((r, w))
    for r, w in sorted(alerts, key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[x[1]["severity"]]):
        st.markdown(
            f'<div class="alert {w["severity"].lower()}"><h4>{w["severity"]} · {r.project_name}</h4>'
            f'<b>{w["issue"]}</b><br>{w["evidence"]}<br>'
            f'<span class="small">Recommended action: {w["action"]}</span></div>',
            unsafe_allow_html=True,
        )
    st.caption(f"{len(alerts)} alert(s) shown. Rules are transparent demo thresholds.")

elif page == "Analytics":
    st.subheader("Portfolio Analytics")
    a, b = st.columns(2)
    sector = df.groupby("sector", as_index=False).agg(avg_risk=("overall_score", "mean"))
    state = df.groupby("state", as_index=False).agg(avg_risk=("overall_score", "mean"))
    with a:
        st.plotly_chart(px.bar(sector.sort_values("avg_risk"), x="avg_risk", y="sector", orientation="h", title="Average risk by sector"), width="stretch")
    with b:
        st.plotly_chart(px.bar(state.sort_values("avg_risk"), x="avg_risk", y="state", orientation="h", title="Average risk by state"), width="stretch")
    st.plotly_chart(
        px.scatter(df, x="cost_variance_percent", y="schedule_variance_percent", size="overall_score",
                   color="overall_risk", hover_name="project_name", title="Cost vs schedule risk landscape"),
        width="stretch",
    )
    st.markdown("### Largest Progress Gaps")
    st.dataframe(df.nlargest(10, "progress_gap")[["project_name", "sector", "state", "progress_gap", "overall_score"]], width="stretch", hide_index=True)

elif page == "AI Assistant":
    st.subheader("Project Monitoring Assistant")
    st.caption("Local deterministic assistant — works without an API key.")
    q = st.text_input("Ask a question", placeholder="Which projects are high risk?")
    ql = q.lower().strip()
    if q:
        if "high risk" in ql:
            ans = df[df.overall_risk == "High"].sort_values("overall_score", ascending=False)
            st.write(f"**{len(ans)} high-risk projects:**")
            st.dataframe(ans[["project_id", "project_name", "sector", "overall_score"]], width="stretch", hide_index=True)
        elif ("highest average cost" in ql) or ("sector" in ql and "cost variance" in ql):
            g = df.groupby("sector", as_index=False).cost_variance_percent.mean().sort_values("cost_variance_percent", ascending=False)
            st.write(f"Highest average cost variance sector in this demo dataset: **{g.iloc[0].sector} ({g.iloc[0].cost_variance_percent:.1f}%)**")
            st.dataframe(g, width="stretch", hide_index=True)
        elif "below 60" in ql:
            st.dataframe(df[df.physical_progress < 60][["project_id", "project_name", "physical_progress", "overall_risk"]], width="stretch", hide_index=True)
        elif "milestone" in ql and ("delay" in ql or "delayed" in ql):
            st.dataframe(
                df[df.milestone_delay_ratio > .30].sort_values("milestone_delay_ratio", ascending=False)
                [["project_id", "project_name", "milestones_delayed", "milestones_total", "milestone_delay_ratio"]],
                width="stretch", hide_index=True,
            )
        else:
            term = ql.replace("project", "").strip()
            match = df[df.project_name.str.lower().str.contains(term, na=False, regex=False)] if term else df.iloc[0:0]
            if len(match):
                r = match.iloc[0]
                st.write(f"**{r.project_name}** — {r.overall_risk} risk ({r.overall_score:.1f}/100).")
                st.write("Main drivers: " + ", ".join(r.drivers))
                for x in recommendations(r, risk_scores(r)):
                    st.write("• " + x)
            else:
                st.info(
                    "Try: \u201cWhich projects are high risk?\u201d, \u201cWhich sector has the highest "
                    "average cost variance?\u201d, \u201cShow projects with progress below 60%\u201d, or "
                    "\u201cWhich projects have major milestone delays?\u201d"
                )

elif page == "About":
    st.subheader("About ProjectPulse AI")
    st.markdown(
        """
    **SIH 2026 · SIH26103 · Smart Automation**

    ProjectPulse AI is a hackathon MVP for demonstrating how a web-based integrated project-monitoring platform can turn structured infrastructure project indicators into risk scores, early warnings, analytics and explainable recommendations.

    **Architecture:** CSV → Pandas feature engineering → Random Forest demonstration classifier + transparent risk engine → Streamlit/Plotly dashboard → SQLite persistence.

    **Data:** 48 reproducible synthetic infrastructure projects. No official or confidential government data is used.

    **Limitations:** Synthetic labels and indicators are not calibrated to government workflows; risk scores are demonstrations; no causal inference; no live departmental systems or external APIs.

    **Future scope:** role-based access, real project data connectors, document/NLP ingestion, GIS, time-series forecasting, model monitoring, human-in-the-loop approvals, audit logs and secure deployment.
    """
    )
    st.download_button("Download synthetic dataset", df.to_csv(index=False).encode(), "projects_demo.csv", "text/csv")

st.sidebar.markdown("---")
st.sidebar.caption("ProjectPulse AI • Synthetic demo • SIH 2026")
