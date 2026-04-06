import json
import os
import uuid
from datetime import datetime
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

try:
    from snowflake_client import (
        get_patient_from_snowflake,
        init_snowflake_connection,
        close_snowflake_connection
    )
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    get_patient_from_snowflake = None
    init_snowflake_connection = None
    close_snowflake_connection = None

# Unconditionally load fallback data
with open("patient_journeys.json", "r") as f:
    PATIENT_DATA = json.load(f)

class AgentState(TypedDict):
    thread_id: str
    patient_id: str
    patient_data: dict
    clinical_summary: str
    nba_recommendation: str
    safety_check: dict
    audit_passed: bool
    retry_count: int
    final_report: str
    risk_score: float
    risk_factors: list
    critic_notes: str
    pending_human_review: bool

def search_medical_guidelines(query: str) -> str:
    """Search medical guidelines (NCCN) for treatment recommendations."""
    guidelines_db = {
        "nsclc_first_line": "For NSCLC: Pembrolizumab monotherapy for PD-L1 >=50%, or platinum-based chemo for PD-L1 <50%",
        "nsclc_second_line": "For NSCLC progression: Osimertinib if EGFR mutation, or Docetaxel + Ramucirumab",
        "breast_her2_positive": "For HER2+ Breast: Trastuzumab + Pertuzumab + Docetaxel first line",
        "breast_triple_negative": "For TNBC: Pembrolizumab + chemo for PD-L1 positive",
        "colorectal": "For Colorectal: FOLFOX or FOLFIRI based on KRAS status",
        "radiation_safety": "Radiation contraindicated if WBC <2.0, Platelets <50K, or severe anemia Hb <8",
        "chemo_safety": "Chemotherapy caution if: WBC <3.0, Hb <10, Creatinine >1.5x baseline, EF <50%",
        "tumor_marker_monitoring": "Tumor markers should trend monthly; >30% rise warrants imaging",
        "supportive_care": "All patients on treatment should receive antiemetic prophylaxis and growth factor support if neutrophils <1000"
    }
    
    query_lower = query.lower()
    results = []
    for key, guideline in guidelines_db.items():
        if any(word in key or word in guideline.lower() for word in query_lower.split()):
            results.append(guideline)
    
    return "\n".join(results) if results else "No specific guidelines found. Consult oncology specialist."

def extract_clinical_features(state: AgentState) -> AgentState:
    """Node 1: Extract clinical features using LLM to summarize patient status."""
    patient = state["patient_data"]
    patient_master = patient.get("Patient_Master", {})
    
    # Normalize snowflake vs json casing
    age = patient.get("AGE", patient.get("Age", patient_master.get("Age", 0)))
    gender = patient.get("GENDER", patient.get("Gender", patient_master.get("Gender", "Unknown")))
    cancer_type = patient.get("CANCER_TYPE", patient.get("Cancer_Type", patient_master.get("Cancer_Type", "Unknown")))
    
    # Prioritize Snowflake metrics if present, otherwise use Outcome JSON
    wbc_avg = patient.get("WBC_AVG", 0)
    tumor_markers = patient.get("TUMOR_MARKERS_AVG", 0)
    wbc_slope = patient.get("WBC_SLOPE_TOTAL", 0)
    chemo_cycles = patient.get("CHEMO_CYCLE_COUNT", 0)
    pre_score = patient.get("PRE_SCORE", 0)
    risk_level = patient.get("RISK_LEVEL", "UNKNOWN")
    
    # Fallback to JSON logic if Snowflake is empty
    high_risk_flag = patient.get("HIGH_RISK_FLAG", 0)
    if not high_risk_flag and "Outcome" in patient:
        high_risk_flag = patient["Outcome"].get("High_Risk_Flag", 0)
    
    patient_id = patient.get("PATIENT_ID", patient.get("Patient_ID", patient_master.get("Patient_ID", "")))
    
    summary = f"""{cancer_type} patient, Age {age}, {gender}. 
    Latest labs show WBC Avg {wbc_avg:.2f} K/uL, Tumor Markers Avg {tumor_markers:.2f}. 
    WBC Slope (cumulative) = {wbc_slope:.2f}. Treatment trajectory: {chemo_cycles} chemo cycles completed. 
    Pre-Score (from Snowflake): {pre_score:.3f}. Risk Level: {risk_level}. 
    Risk Status: {'HIGH RISK' if high_risk_flag == 1 or risk_level == 'HIGH_RISK' else 'Standard'}."""
    
    state["clinical_summary"] = summary
    return state

def recommend_nba(state: AgentState) -> AgentState:
    """Node 2: Recommend Next Best Action based on NCCN-inspired logic."""
    patient = state["patient_data"]
    patient_master = patient.get("Patient_Master", {})
    labs = patient.get("Lab_Results", [])
    
    cancer_type = patient.get("CANCER_TYPE", patient.get("Cancer_Type", patient_master.get("Cancer_Type", "Unknown")))
    
    # Use state metrics calculated in previous risk scoring node
    wbc_avg = patient.get("WBC_AVG", 0)
    if not wbc_avg and labs:
        wbc_avg = sum(l.get("White_Blood_Cell_Count", 0) for l in labs) / len(labs)
    wbc_avg = wbc_avg or 5.0
        
    pre_score = state.get("risk_score", 0) or patient.get("PRE_SCORE", 0)
    chemo_cycles = patient.get("CHEMO_CYCLE_COUNT", 0) or len(patient.get("Treatment_Logs", {}).get("Chemo_Cycles", []))
    
    guidelines_context = search_medical_guidelines(cancer_type)
    
    if pre_score > 0.4 or wbc_avg < 3.5:
        recommendation = f"""1. RECOMMENDED ACTION: Defer systemic chemotherapy. Consider supportive care.

2. RATIONALE: Patient has elevated risk score ({pre_score:.3f}) or low WBC ({wbc_avg:.2f}). 
   NCCN guidelines recommend holding treatment until hematologic recovery.

3. TIMELINE: Re-evaluate in 1-2 weeks after labs are repeated.

4. MONITORING: Daily CBC, transfusion support as needed, growth factor consideration."""
    else:
        recommendation = f"""1. RECOMMENDED ACTION: Continue {cancer_type} treatment protocol with next chemo cycle.

2. RATIONALE: Patient risk score ({pre_score:.3f}) is within acceptable range. {guidelines_context}

3. TIMELINE: Initiate within 1 week if ANC >1500 and platelets >100K.

4. MONITORING: Weekly CBC, monthly tumor markers, imaging every 2-3 cycles."""
    
    state["nba_recommendation"] = recommendation
    return state

def risk_scoring_node(state: AgentState) -> AgentState:
    """Node 2.5: Multi-Factor Risk Scoring - Uses pre-calculated values (Snowflake) or calculates locally."""
    patient = state["patient_data"]
    patient_master = patient.get("Patient_Master", {})
    labs = patient.get("Lab_Results", [])
    
    # 1. Normalize Core Metrics
    pre_score = patient.get("PRE_SCORE", 0.0)
    age = patient.get("AGE", patient.get("Age", patient_master.get("Age", 0)))
    
    # Calculate WBC Avg if not in Snowflake
    wbc_avg = patient.get("WBC_AVG", 0)
    if not wbc_avg and labs:
        wbc_avg = sum(l.get("White_Blood_Cell_Count", 0) for l in labs) / len(labs)
    
    # Calculate Tumor Markers Avg
    tumor_markers = patient.get("TUMOR_MARKERS_AVG", 0)
    if not tumor_markers and labs:
        tumor_markers = sum(l.get("Tumor_Markers", 0) for l in labs) / len(labs)
        
    chemo_cycles = patient.get("CHEMO_CYCLE_COUNT", 0)
    if not chemo_cycles and "Treatment_Logs" in patient:
        chemo_cycles = len(patient["Treatment_Logs"].get("Chemo_Cycles", []))
        
    wbc_slope = patient.get("WBC_SLOPE_TOTAL", 0)
    if not wbc_slope and len(labs) >= 2:
        # Simple local slope calculation
        wbc_slope = labs[-1].get("White_Blood_Cell_Count", 0) - labs[0].get("White_Blood_Cell_Count", 0)
    
    # 2. Risk Calculation (Simulate Snowflake Model if missing)
    risk_factors = []
    if pre_score == 0:
        # Heuristic if ML score is missing
        if age > 70: risk_factors.append("Advanced Age (>70)")
        if tumor_markers > 20: risk_factors.append(f"High Tumor Load ({tumor_markers:.1f})")
        if wbc_avg < 4.0: risk_factors.append(f"Low Baseline WBC ({wbc_avg:.1f})")
        if chemo_cycles > 4: risk_factors.append(f"High Chemo Cycle Count ({chemo_cycles})")
        if wbc_slope < -2.0: risk_factors.append("Declining WBC Trend")
        
        # Simple probabilistic score
        pre_score = min(0.1 + (len(risk_factors) * 0.15), 0.95)
    
    derived_factors_str = patient.get("RISK_FACTORS_DERIVED", "")
    if derived_factors_str:
        for f in derived_factors_str.split(","):
            f = f.strip()
            if f and f not in risk_factors:
                risk_factors.append(f)
                
    state["risk_score"] = pre_score
    state["risk_factors"] = risk_factors
    
    return state

def critic_node(state: AgentState) -> AgentState:
    """Node 2.7: Reflexion/Critic Loop - Analyze 'Gray Area' patients."""
    patient = state["patient_data"]
    labs = patient.get("Lab_Results", [])
    
    wbc_slope = state["patient_data"].get("WBC_SLOPE_TOTAL", 0)
    if not wbc_slope and len(labs) >= 2:
        wbc_slope = labs[-1].get("White_Blood_Cell_Count", 0) - labs[0].get("White_Blood_Cell_Count", 0)
        
    chemo_cycles = state["patient_data"].get("CHEMO_CYCLE_COUNT", 0)
    if not chemo_cycles and "Treatment_Logs" in patient:
        chemo_cycles = len(patient["Treatment_Logs"].get("Chemo_Cycles", []))
    
    critic_notes = []
    override_to_fail = False
    
    if wbc_slope < 0:
        if chemo_cycles > 3:
            critic_notes.append(f"CRITICAL: WBC declining (slope={wbc_slope:.2f}) with {chemo_cycles} prior chemo cycles - cumulative bone marrow toxicity risk")
            override_to_fail = True
        elif wbc_slope < -1.0:
            critic_notes.append(f"WARNING: Rapid WBC drop ({wbc_slope:.2f}) - possible infection or marrow suppression")
            override_to_fail = True
    
    if chemo_cycles > 6:
        critic_notes.append(f"WARNING: High cumulative chemo exposure ({chemo_cycles} cycles) - consider treatment holiday")
        override_to_fail = True
    
    if not critic_notes:
        critic_notes.append("GRAY AREA REVIEW: No critical flags detected. Patient may proceed with standard monitoring.")
    
    state["critic_notes"] = " | ".join(critic_notes)
    
    if override_to_fail:
        state["safety_check"] = state.get("safety_check", {})
        state["safety_check"]["critic_override"] = True
        state["safety_check"]["critic_notes"] = critic_notes
        state["audit_passed"] = False
        state["pending_human_review"] = True
    else:
        state["pending_human_review"] = False
    
    return state

def safety_audit(state: AgentState) -> AgentState:
    """Node 3: Audit if NBA is safe given patient's lab results."""
    patient = state["patient_data"]
    labs = patient.get("Lab_Results", [])
    
    wbc_avg = state.get("patient_data", {}).get("WBC_AVG")
    if not wbc_avg and labs:
        wbc_avg = sum(l.get("White_Blood_Cell_Count", 0) for l in labs) / len(labs)
    wbc_avg = wbc_avg or 5.0
    
    pre_score = state.get("risk_score", 0)
    risk_level = patient.get("RISK_LEVEL", "STANDARD")
    if pre_score > 0.6: risk_level = "HIGH_RISK"
    
    safety_checks = []
    is_safe = True
    
    if wbc_avg < 3.0:
        safety_checks.append(f"FAIL: WBC Avg ({wbc_avg:.2f}) < 3.0 - Chemotherapy contraindicated")
        is_safe = False
    else:
        safety_checks.append(f"PASS: WBC Avg ({wbc_avg:.2f}) within safe range")
    
    if pre_score > 0.4 or risk_level == "HIGH_RISK":
        safety_checks.append(f"WARNING: Risk Score ({pre_score:.3f}) > 0.4 or HIGH_RISK classification")
        is_safe = False
    
    audit_result = {
        "timestamp": datetime.now().isoformat(),
        "checks": safety_checks,
        "overall_safe": is_safe,
        "wbc_avg": float(wbc_avg),
        "pre_score": float(pre_score)
    }
    
    state["safety_check"] = audit_result
    state["audit_passed"] = is_safe
    
    if is_safe:
        state["retry_count"] = 0
    else:
        state["retry_count"] = state.get("retry_count", 0) + 1
    
    return state

def should_retry(state: AgentState) -> str:
    """Routing logic: if safety audit failed and retries < 3, go back to recommend_nba."""
    if not state["audit_passed"] and state.get("retry_count", 0) < 3:
        return "retry"
    return "proceed"

def route_from_risk_scoring(state: AgentState) -> str:
    """Route based on risk_score: Gray area (0.2-0.4) goes to critic_node."""
    risk_score = state.get("risk_score", 0.0)
    if 0.2 <= risk_score < 0.4:
        return "critic"
    return "safety_audit"

def generate_final_report(state: AgentState) -> AgentState:
    """Generate final report after safety audit passes."""
    patient = state["patient_data"]
    
    patient_id = patient.get("PATIENT_ID", "Unknown")
    age = patient.get("AGE", 0)
    gender = patient.get("GENDER", "Unknown")
    cancer_type = patient.get("CANCER_TYPE", "Unknown")
    
    risk_score = state.get("risk_score", 0.0)
    risk_factors = state.get("risk_factors", [])
    critic_notes = state.get("critic_notes", "")
    
    audit_status = "PASSED"
    if not state["audit_passed"]:
        audit_status = "FAILED"
    elif critic_notes and "CRITICAL" in critic_notes:
        audit_status = "FAILED"
    
    report = f"""=== ONCOLOGY PATIENT JOURNEY REPORT ===
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Data Source: Snowflake GOLD_PATIENT_RISK

PATIENT: {patient_id}
Demographics: {age}yo {gender}, {cancer_type}

RISK SCORE (PRE_SCORE): {risk_score}
RISK LEVEL: {patient.get('RISK_LEVEL', 'UNKNOWN')}
RISK FACTORS: {', '.join(risk_factors) if risk_factors else 'None'}
CRITIC NOTES: {critic_notes if critic_notes else 'None'}

CLINICAL SUMMARY:
{state['clinical_summary']}

NEXT BEST ACTION RECOMMENDATION:
{state['nba_recommendation']}

SAFETY AUDIT: {audit_status}
{json.dumps(state['safety_check'], indent=2)}

GROUND TRUTH: {'HIGH RISK' if patient.get('HIGH_RISK_FLAG', 0) == 1 else 'STANDARD'}

===========================================
"""
    
    state["final_report"] = report
    return state

# Global cache for the checkpointer and graph
_graph_instance = None
_checkpointer_instance = None

def get_graph_instance() -> StateGraph:
    """Return a singleton instance of the compiled LangGraph workflow with its MemorySaver."""
    global _graph_instance, _checkpointer_instance
    if _graph_instance is None:
        _checkpointer_instance = MemorySaver()
        workflow = StateGraph(AgentState)
        
        workflow.add_node("extract_clinical_features", extract_clinical_features)
        workflow.add_node("recommend_nba", recommend_nba)
        workflow.add_node("risk_scoring_node", risk_scoring_node)
        workflow.add_node("critic_node", critic_node)
        workflow.add_node("safety_audit", safety_audit)
        workflow.add_node("generate_final_report", generate_final_report)
        
        workflow.set_entry_point("extract_clinical_features")
        
        workflow.add_edge("extract_clinical_features", "recommend_nba")
        workflow.add_edge("recommend_nba", "risk_scoring_node")
        
        workflow.add_conditional_edges(
            "risk_scoring_node",
            route_from_risk_scoring,
            {
                "critic": "critic_node",
                "safety_audit": "safety_audit"
            }
        )
        
        workflow.add_edge("critic_node", "safety_audit")
        
        workflow.add_conditional_edges(
            "safety_audit",
            should_retry,
            {
                "retry": "recommend_nba",
                "proceed": "generate_final_report"
            }
        )
        
        workflow.add_edge("generate_final_report", END)
        _graph_instance = workflow.compile(checkpointer=_checkpointer_instance)
    
    return _graph_instance

def get_patient_data(patient_id: str) -> Optional[dict]:
    """Get patient data from Snowflake or fallback to local JSON."""
    if SNOWFLAKE_AVAILABLE and get_patient_from_snowflake:
        try:
            patient = get_patient_from_snowflake(patient_id)
            if patient:
                return patient
        except Exception as e:
            print(f"Snowflake query failed: {e}, falling back to local data")
    
    if 'PATIENT_DATA' in globals():
        for p in PATIENT_DATA["patients"]:
            if p["Patient_Master"]["Patient_ID"] == patient_id:
                return p
    return None

def run_patient_analysis(patient_id: str, patient_data: dict = None) -> str:
    """Run the full workflow for a single patient."""
    if patient_data is None:
        patient_data = get_patient_data(patient_id)
    
    if not patient_data:
        return f"Patient {patient_id} not found"
    
    thread_id = str(uuid.uuid4())
    
    initial_state: AgentState = {
        "thread_id": thread_id,
        "patient_id": patient_id,
        "patient_data": patient_data,
        "clinical_summary": "",
        "nba_recommendation": "",
        "safety_check": {},
        "audit_passed": False,
        "retry_count": 0,
        "final_report": "",
        "risk_score": 0.0,
        "risk_factors": [],
        "critic_notes": "",
        "pending_human_review": False
    }
    
    graph = get_graph_instance()
    result = graph.invoke(initial_state, config={"configurable": {"thread_id": thread_id}})
    
    return result["final_report"]

def run_patient_analysis_with_status(patient_id: str) -> dict:
    """Run the full workflow and return status including if human review is needed."""
    patient = get_patient_data(patient_id)
    
    if not patient:
        return {"error": f"Patient {patient_id} not found"}
    
    thread_id = str(uuid.uuid4())
    
    initial_state: AgentState = {
        "thread_id": thread_id,
        "patient_id": patient_id,
        "patient_data": patient,
        "clinical_summary": "",
        "nba_recommendation": "",
        "safety_check": {},
        "audit_passed": False,
        "retry_count": 0,
        "final_report": "",
        "risk_score": 0.0,
        "risk_factors": [],
        "critic_notes": "",
        "pending_human_review": False
    }
    
    graph = get_graph_instance()
    
    try:
        result = graph.invoke(initial_state, config={"configurable": {"thread_id": thread_id}})
        
        # Check if we should interrupt / wait for human review
        if result.get("pending_human_review", False):
            return {
                "status": "pending",
                "thread_id": thread_id,
                "report": result.get("final_report", "Pending Review..."),
                "risk_score": result.get("risk_score", 0.0),
                "risk_factors": result.get("risk_factors", []),
                "pending_review": True
            }
            
        return {
            "status": "completed",
            "thread_id": thread_id,
            "report": result.get("final_report", ""),
            "risk_score": result.get("risk_score", 0.0),
            "risk_factors": result.get("risk_factors", []),
            "pending_review": False
        }
    except Exception as e:
        return {"error": str(e)}

def resume_after_approval(thread_id: str) -> dict:
    """Resume the graph after human approval."""
    graph = get_graph_instance()
    try:
        # Pass a command to the graph's checkpoint
        result = graph.invoke(Command(resume={"approved": True}), config={"configurable": {"thread_id": thread_id}})
        return {
            "status": "completed", 
            "report": result.get("final_report", ""),
            "risk_score": result.get("risk_score", 0.0),
            "risk_factors": result.get("risk_factors", [])
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if SNOWFLAKE_AVAILABLE:
        init_snowflake_connection()
    
    sample_patient_id = "PID-00001"
    print(f"Running analysis for {sample_patient_id}...\n")
    report = run_patient_analysis(sample_patient_id)
    print(report)
