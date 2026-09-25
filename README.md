# ProjectPulse AI — SIH 2026 MVP

**SIH26103 — Use case on web-based integrated project-monitoring platform**  
**Organization:** Ministry of Statistics and Programme Implementation (MoSPI)  
**Theme:** Smart Automation

## Important
This prototype uses synthetic data for demonstration. It does not represent official MoSPI data or official government risk assessments.

## Run
```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

No login or API key is required. The app opens at http://localhost:8501.

## Architecture
CSV → Pandas feature engineering → Random Forest demo classifier + transparent risk engine → Streamlit/Plotly dashboard → SQLite.

## Risk methodology
Cost 35% + Time 35% + Progress 20% + Issue/Milestone 10%.
0–39 Low, 40–69 Medium, 70–100 High.

## Pages
Overview, Projects, Risk Analysis, Early Warnings, Analytics, AI Assistant, About.

## Demo flow
1. Overview: portfolio KPIs and risk distribution.
2. Projects: filter High Risk and open PP-001 Highway Project Alpha.
3. Explain cost variance, progress gap and delayed milestones.
4. Early Warnings: show evidence + suggested action.
5. Risk Analysis: transparent weighting + Random Forest feature importance.
6. Analytics: sector/state comparisons.
7. AI Assistant: ask “Which projects are high risk?”
8. Close with future integration to authorized real project data.

## 2-minute pitch
ProjectPulse AI demonstrates how structured project-monitoring indicators can become early, explainable management signals. The prototype derives cost variance, schedule variance, progress gaps and milestone delay ratios from synthetic infrastructure project data. A transparent weighted risk engine and a Random Forest demonstration model classify projects as low, medium or high risk. The dashboard then provides evidence-based early warnings, demonstration recommendations, portfolio analytics and a deterministic local assistant. The system is intentionally lightweight, runs locally with Streamlit and requires no API key. The data is synthetic and the scores are not official government assessments. The architecture can later connect to authorized institutional data sources, GIS, documents, time-series forecasting and audit workflows.

## 5-minute demo
- 0:00–0:40 Overview and synthetic-data disclaimer
- 0:40–1:20 KPIs and risk charts
- 1:20–2:20 PP-001 project detail
- 2:20–3:00 Early Warning Center
- 3:00–3:45 Risk Analysis and feature importance
- 3:45–4:20 Analytics
- 4:20–5:00 AI Assistant

## Screenshots for PPT
Overview, Project Portfolio, PP-001 Detail, Early Warnings, Risk Analysis, Analytics, AI Assistant, About.

## Fixes applied (MVP hardening pass)
- Replaced the deprecated/removed `use_container_width=True` argument (dropped from `st.plotly_chart`/`st.vega_lite_chart` in 2026 Streamlit releases) with `width="stretch"` across every chart and dataframe call in `app.py`. Without this fix the app crashed with `TypeError` on the very first page load against current Streamlit versions.
- Hardened the AI Assistant's free-text project search: it now uses a literal (non-regex) match and no longer matches every row when the query is empty after stripping the word "project".
- Fixed the `money()` currency formatter so negative values render as `-₹500 Cr` instead of `₹-500 Cr`.
- Verified all modules compile and the full pipeline (data load → feature engineering → model training → risk scoring → warnings → recommendations) runs cleanly end to end, and simulated every page of the dashboard plus every AI Assistant query branch to confirm they render without errors.

## Limitations
Synthetic data/labels, demo thresholds, no live government integration, no causal inference, deterministic assistant.

## Future scope
Authorized data connectors, role-based access, GIS, document/NLP ingestion, time-series forecasting, model monitoring, audit logs, human approval workflows and secure deployment.
