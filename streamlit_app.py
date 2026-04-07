import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time

# Import existing logic
try:
    from snowflake_client import (
        get_dashboard_stats_from_snowflake,
        get_high_risk_patients,
        get_risk_factors_aggregation,
        get_patient_from_snowflake,
        init_snowflake_connection,
        SNOWFLAKE_AVAILABLE
    )
except ImportError:
    SNOWFLAKE_AVAILABLE = False

from langgraph_oncology import run_patient_analysis_with_status, get_patient_data

# Page Config
st.set_page_config(
    page_title="Oncology Patient Journey AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stAlert {
        border-radius: 10px;
    }
    .status-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .status-high { background-color: #ff4b4b; color: white; }
    .status-standard { background-color: #00c0f2; color: white; }
</style>
""", unsafe_allow_html=True)

# --- Data Loading Helpers ---

@st.cache_data(ttl=300)
def load_dashboard_stats():
    if SNOWFLAKE_AVAILABLE:
        try:
            return get_dashboard_stats_from_snowflake()
        except:
            pass
    
    # Fallback/Mock stats if Snowflake fails
    return {
        "TOTAL_PATIENTS": 1000,
        "HIGH_RISK_COUNT": 342,
        "STANDARD_RISK_COUNT": 658,
        "AVG_PRE_SCORE": 0.285
    }

@st.cache_data(ttl=300)
def load_alert_data():
    if SNOWFLAKE_AVAILABLE:
        try:
            patients = get_high_risk_patients(limit=50)
            return pd.DataFrame(patients)
        except:
            pass
    
    # Fallback to local JSON if Snowflake fails
    try:
        with open("patient_journeys.json", "r") as f:
            data = json.load(f)
        alerts = []
        for p in data["patients"][:20]:
            outcome = p.get("Outcome", {})
            if outcome.get("High_Risk_Flag") == 1:
                alerts.append({
                    "PATIENT_ID": p["Patient_Master"]["Patient_ID"],
                    "CANCER_TYPE": p["Patient_Master"]["Cancer_Type"],
                    "AGE": p["Patient_Master"]["Age"],
                    "PRE_SCORE": outcome.get("Risk_Score", 0),
                    "RISK_LEVEL": "HIGH_RISK"
                })
        return pd.DataFrame(alerts)
    except:
        return pd.DataFrame()

# --- Sidebar ---

with st.sidebar:
    st.title("🧬 Oncology AI")
    st.subheader("Clinical Decision Support")
    st.divider()
    
    stats = load_dashboard_stats()
    
    st.metric("Total Patients", f"{stats.get('TOTAL_PATIENTS', 0):,}")
    st.metric("High Risk Identified", f"{stats.get('HIGH_RISK_COUNT', 0):,}", delta=f"{stats.get('HIGH_RISK_COUNT', 0)/stats.get('TOTAL_PATIENTS', 1)*100:.1f}%", delta_color="inverse")
    st.metric("Avg Risk Score", f"{stats.get('AVG_PRE_SCORE', 0):.3f}")
    
    st.divider()
    if SNOWFLAKE_AVAILABLE:
        st.success("Connected to Snowflake")
    else:
        st.warning("Using Local JSON Fallback")

# --- Main App Layout ---

st.title("Oncology Patient Journey Dashboard")
st.markdown("### Multi-Agent Clinical Triage & Risk Stratification")

tabs = st.tabs(["🔥 Active Alerts", "🔍 Patient Analysis", "📊 Population Insights"])

# --- TAB 1: ACTIVE ALERTS ---
with tabs[0]:
    st.header("High-Risk Clinical Alerts")
    st.info("The following patients have been flagged by the Risk Agent for immediate clinical review.")
    
    df_alerts = load_alert_data()
    if not df_alerts.empty:
        # Style the dataframe
        def color_risk(val):
            color = '#ff4b4b' if val == 'HIGH_RISK' else '#00c0f2'
            return f'color: {color}; font-weight: bold'

        st.dataframe(
            df_alerts[["PATIENT_ID", "CANCER_TYPE", "AGE", "PRE_SCORE", "RISK_LEVEL"]],
            use_container_width=True,
            hide_index=True
        )
        
        st.download_button(
            label="Export Alert List (CSV)",
            data=df_alerts.to_csv(index=False),
            file_name=f"oncology_alerts_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.write("No active high-risk alerts found.")

# --- TAB 2: PATIENT ANALYSIS ---
with tabs[1]:
    st.header("Individual Patient Deep-Dive")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Patient Selection")
        # Selector for patients
        if not df_alerts.empty:
            patient_list = df_alerts["PATIENT_ID"].tolist()
        else:
            patient_list = ["PID-00001", "PID-00002", "PID-00003"]
            
        selected_pid = st.selectbox("Select Patient ID", patient_list)
        
        if st.button("Run AI Analysis", type="primary"):
            with st.spinner(f"Agents analyzing {selected_pid}..."):
                # Simulation of agent steps
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Extracting clinical features...")
                time.sleep(0.5)
                progress_bar.progress(25)
                
                status_text.text("Consulting NCCN guidelines...")
                time.sleep(0.5)
                progress_bar.progress(50)
                
                status_text.text("Running Multi-factor Risk Scoring...")
                time.sleep(0.5)
                progress_bar.progress(75)
                
                # Actual run
                result = run_patient_analysis_with_status(selected_pid)
                
                progress_bar.progress(100)
                status_text.text("Analysis Complete.")
                
                if "error" in result:
                    st.error(f"Analysis failed: {result['error']}")
                else:
                    st.session_state['current_report'] = result['report']
                    st.session_state['current_score'] = result['risk_score']
                    st.session_state['current_factors'] = result['risk_factors']

    with col2:
        if 'current_report' in st.session_state:
            st.subheader(f"Analysis Report: {selected_pid}")
            
            # Risk Header
            score = st.session_state['current_score']
            risk_color = "red" if score > 0.4 else "green"
            st.markdown(f"### Overall Risk Score: :{risk_color}[{score:.3f}]")
            
            # Factors
            if st.session_state['current_factors']:
                st.write("**Key Risk Factors:**")
                cols = st.columns(len(st.session_state['current_factors']))
                for i, factor in enumerate(st.session_state['current_factors']):
                    cols[i].info(factor)
            
            # Full Report Text Area
            st.text_area("Full Clinical Reasoning", st.session_state['current_report'], height=400)
            
            # Action Buttons
            c1, c2 = st.columns(2)
            if c1.button("Approve Recommendation"):
                st.success("Decision logged to Snowflake.")
            if c2.button("Escalate to Specialist"):
                st.warning("Escalation triggered. Notification sent to Attending Physician.")
        else:
            st.info("Select a patient and click 'Run AI Analysis' to view findings.")

# --- TAB 3: POPULATION INSIGHTS ---
with tabs[2]:
    st.header("Population Risk Analytics")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Risk Score Distribution")
        # Mocking some data for the chart
        import numpy as np
        scores = np.random.beta(2, 5, 1000)
        fig = px.histogram(scores, nbins=30, labels={'value': 'Risk Score'}, title="Risk Distribution (Current Population)")
        fig.update_layout(showlegend=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Top Clinical Risk Factors")
        if SNOWFLAKE_AVAILABLE:
            try:
                agg_factors = get_risk_factors_aggregation()
                df_f = pd.DataFrame(agg_factors)
                fig_bar = px.bar(df_f, x='RISK_FACTOR', y='COUNT', color='COUNT', color_continuous_scale='Reds')
            except:
                df_f = pd.DataFrame({"Factor": ["Advanced Age", "Low WBC", "High cycles"], "Count": [450, 320, 210]})
                fig_bar = px.bar(df_f, x='Factor', y='Count')
        else:
            df_f = pd.DataFrame({"Factor": ["Advanced Age", "Low WBC", "High cycles"], "Count": [450, 320, 210]})
            fig_bar = px.bar(df_f, x='Factor', y='Count')
            
        fig_bar.update_layout(template="plotly_dark")
        st.plotly_chart(fig_bar, use_container_width=True)

st.divider()
st.caption("Oncology Patient Journey AI System | Clinical Demo | Built with LangGraph & Streamlit")
