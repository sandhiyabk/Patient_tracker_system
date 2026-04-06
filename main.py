import json
import re
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, List, Dict
from collections import Counter
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langgraph.types import Command

try:
    from snowflake_client import (
        get_patient_from_snowflake,
        get_dashboard_stats_from_snowflake,
        get_risk_factors_aggregation,
        get_high_risk_patients,
        init_snowflake_connection,
        close_snowflake_connection,
        snowflake_client
    )
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    snowflake_client = None
    get_patient_from_snowflake = None
    get_dashboard_stats_from_snowflake = None
    get_risk_factors_aggregation = None
    get_high_risk_patients = None
    init_snowflake_connection = None
    close_snowflake_connection = None

from langgraph_oncology import run_patient_analysis

patient_db: dict = {}

def load_patient_data():
    global patient_db
    if SNOWFLAKE_AVAILABLE:
        try:
            patients = get_high_risk_patients(limit=1000)
            for p in patients:
                patient_db[p["PATIENT_ID"]] = p
            print(f"Loaded {len(patient_db)} patients from Snowflake")
            return
        except Exception as e:
            print(f"Snowflake load failed: {e}")
    
    with open("patient_journeys.json", "r") as f:
        data = json.load(f)
    for p in data["patients"]:
        pid = p["Patient_Master"]["Patient_ID"]
        patient_db[pid] = p
    print(f"Loaded {len(patient_db)} patients from JSON")

@asynccontextmanager
async def lifespan(app: FastAPI):
    if SNOWFLAKE_AVAILABLE:
        init_snowflake_connection()
    load_patient_data()
    yield
    if SNOWFLAKE_AVAILABLE:
        close_snowflake_connection()

app = FastAPI(title="Oncology Patient Journey API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class AnalysisResponse(BaseModel):
    patient_id: str
    report: str
    risk_score: float
    risk_factors: List[str]
    generated_at: datetime

class RiskFactorCount(BaseModel):
    factor: str
    count: int

class DashboardStats(BaseModel):
    total_patients: int
    auto_approved: int
    auto_rejected: int
    pending_review: int
    high_risk_caught: int
    top_risk_factors: List[RiskFactorCount]
    baseline_high_risk: int

class HighRiskAlert(BaseModel):
    Patient_ID: str
    Cancer_Type: str
    Age: int
    Latest_WBC: float
    Latest_Hemoglobin: float
    Latest_Tumor_Markers: float

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/analyze/{patient_id}", response_model=AnalysisResponse)
async def analyze_patient(patient_id: str):
    patient_id_upper = patient_id.upper()
    if patient_id_upper not in patient_db:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    try:
        patient_data = patient_db[patient_id_upper]
        report = run_patient_analysis(patient_id, patient_data)
        
        risk_score_match = re.search(r"RISK SCORE.*:\s*([\d.]+)", report)
        risk_score = float(risk_score_match.group(1)) if risk_score_match else 0.0
        
        risk_factors_match = re.search(r"RISK FACTORS:\s*(.+?)(?=\n|$)", report)
        risk_factors_str = risk_factors_match.group(1) if risk_factors_match else ""
        risk_factors = [f.strip() for f in risk_factors_str.split(",")] if risk_factors_str and risk_factors_str != "None" else []
        
        return AnalysisResponse(
            patient_id=patient_id,
            report=report,
            risk_score=risk_score,
            risk_factors=risk_factors,
            generated_at=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    if SNOWFLAKE_AVAILABLE:
        try:
            stats = get_dashboard_stats_from_snowflake()
            factors = get_risk_factors_aggregation()
            
            total = stats.get("TOTAL_PATIENTS", 0) or 0
            high_risk = stats.get("HIGH_RISK_COUNT", 0) or 0
            standard = stats.get("STANDARD_RISK_COUNT", 0) or 0
            
            top_factors = [RiskFactorCount(factor=f["RISK_FACTOR"], count=f["COUNT"]) for f in factors[:3]]
            
            return DashboardStats(
                total_patients=total,
                auto_approved=standard,
                auto_rejected=0,
                pending_review=0,
                high_risk_caught=high_risk,
                top_risk_factors=top_factors,
                baseline_high_risk=high_risk
            )
        except Exception as e:
            print(f"Snowflake stats failed: {e}")
    
    all_factors = []
    high_risk = 0
    for p in patient_db.values():
        if SNOWFLAKE_AVAILABLE:
            if p.get("RISK_LEVEL") == "HIGH_RISK":
                high_risk += 1
            factors_str = p.get("RISK_FACTORS_DERIVED", "")
            if factors_str:
                all_factors.extend([f.strip() for f in factors_str.split(",")])
        else:
            if p.get("Outcome", {}).get("High_Risk_Flag") == 1:
                high_risk += 1
            all_factors.extend(p.get("Outcome", {}).get("Risk_Factors", []))
    
    factor_counts = Counter(all_factors)
    top_factors = [RiskFactorCount(factor=f, count=c) for f, c in factor_counts.most_common(3)]
    
    return DashboardStats(
        total_patients=len(patient_db),
        auto_approved=len(patient_db) - high_risk,
        auto_rejected=0,
        pending_review=0,
        high_risk_caught=high_risk,
        top_risk_factors=top_factors,
        baseline_high_risk=high_risk
    )

@app.get("/alerts/high-risk", response_model=List[HighRiskAlert])
async def get_high_risk_alerts():
    if SNOWFLAKE_AVAILABLE:
        try:
            patients = get_high_risk_patients(limit=100)
            return [
                HighRiskAlert(
                    Patient_ID=p["PATIENT_ID"],
                    Cancer_Type=p.get("CANCER_TYPE", "Unknown"),
                    Age=p.get("AGE", 0),
                    Latest_WBC=p.get("WBC_AVG", 0),
                    Latest_Hemoglobin=0,
                    Latest_Tumor_Markers=p.get("TUMOR_MARKERS_AVG", 0)
                )
                for p in patients
            ]
        except Exception as e:
            print(f"Snowflake alerts failed: {e}")
    
    alerts = []
    for p in patient_db.values():
        labs = p.get("Lab_Results", [])
        if labs:
            latest = labs[-1]
            if latest.get("White_Blood_Cell_Count", 100) < 3.0:
                alerts.append(HighRiskAlert(
                    Patient_ID=p["Patient_Master"]["Patient_ID"],
                    Cancer_Type=p["Patient_Master"]["Cancer_Type"],
                    Age=p["Patient_Master"]["Age"],
                    Latest_WBC=latest.get("White_Blood_Cell_Count", 0),
                    Latest_Hemoglobin=latest.get("Hemoglobin", 0),
                    Latest_Tumor_Markers=latest.get("Tumor_Markers", 0)
                ))
    return alerts

@app.get("/patients/{patient_id}")
async def get_patient(patient_id: str):
    patient_id_upper = patient_id.upper()
    if patient_id_upper not in patient_db:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return patient_db[patient_id_upper]
