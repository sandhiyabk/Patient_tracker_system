"""
Risk Agent - Specialized agent for oncology patient risk assessment
Uses multi-factor analysis with clinical guidelines
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional, TypedDict
from dataclasses import dataclass, asdict


class RiskLevel:
    """Risk level constants"""
    STANDARD = "STANDARD"
    MODERATE = "MODERATE_RISK"
    HIGH = "HIGH_RISK"
    CRITICAL = "CRITICAL"


class RiskAgent:
    """Specialized agent for assessing patient risk in oncology context"""
    
    def __init__(self):
        self.risk_thresholds = {
            "wbc_min": 3.0,
            "wbc_warning": 4.0,
            "hemoglobin_min": 10.0,
            "hemoglobin_warning": 11.0,
            "platelet_min": 50.0,
            "creatinine_max": 1.5,
            "tumor_marker_threshold": 35.0,
            "age_risk_threshold": 70
        }
        
        self.risk_weights = {
            "advanced_age": 0.15,
            "low_wbc": 0.25,
            "anemia": 0.20,
            "thrombocytopenia": 0.15,
            "renal_dysfunction": 0.10,
            "high_tumor_markers": 0.15,
            "multiple_chemo": 0.10,
            "comorbidities": 0.10
        }
    
    def assess_risk(self, patient_data: Dict) -> Dict:
        """
        Comprehensive risk assessment for oncology patient.
        
        Args:
            patient_data: Patient record with labs, demographics, treatment history
            
        Returns:
            Risk assessment result with score, level, and contributing factors
        """
        assessment_id = str(uuid.uuid4())
        
        demographics = self._extract_demographics(patient_data)
        labs = self._extract_labs(patient_data)
        treatment = self._extract_treatment_history(patient_data)
        oncology = self._extract_oncology_data(patient_data)
        
        risk_factors = []
        risk_score = 0.0
        
        age_score, age_factor = self._assess_age(demographics)
        risk_score += age_score
        if age_factor:
            risk_factors.append(age_factor)
        
        wbc_score, wbc_factor = self._assess_wbc(labs)
        risk_score += wbc_score
        if wbc_factor:
            risk_factors.append(wbc_factor)
        
        hgb_score, hgb_factor = self._assess_hemoglobin(labs)
        risk_score += hgb_score
        if hgb_factor:
            risk_factors.append(hgb_factor)
        
        platelet_score, platelet_factor = self._assess_platelets(labs)
        risk_score += platelet_score
        if platelet_factor:
            risk_factors.append(platelet_factor)
        
        tumor_score, tumor_factor = self._assess_tumor_markers(labs, oncology)
        risk_score += tumor_score
        if tumor_factor:
            risk_factors.append(tumor_factor)
        
        chemo_score, chemo_factor = self._assess_chemo_burden(treatment)
        risk_score += chemo_score
        if chemo_factor:
            risk_factors.append(chemo_factor)
        
        renal_score, renal_factor = self._assess_renal(labs)
        risk_score += renal_score
        if renal_factor:
            risk_factors.append(renal_factor)
        
        risk_score = min(risk_score, 1.0)
        
        risk_level = self._determine_risk_level(risk_score, risk_factors)
        
        recommendations = self._generate_recommendations(risk_level, risk_factors)
        
        return {
            "assessment_id": assessment_id,
            "timestamp": datetime.now().isoformat(),
            "patient_id": demographics.get("patient_id", "Unknown"),
            "demographics": demographics,
            "labs_summary": labs,
            "risk_score": round(risk_score, 3),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommendations": recommendations,
            "oncology_stage": oncology.get("stage", "Unknown"),
            "performance_status": oncology.get("performance_status", "ECOG 1"),
            "requires_review": risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        }
    
    def _extract_demographics(self, patient: Dict) -> Dict:
        """Extract demographic information"""
        if "demography" in patient:
            demo = patient["demography"].copy()
            demo["patient_id"] = patient.get("patient_id", "Unknown")
            return demo
        
        if "Patient_Master" in patient:
            return patient["Patient_Master"]
        
        return {
            "age": patient.get("AGE", patient.get("Age", 60)),
            "gender": patient.get("GENDER", patient.get("Gender", "Unknown")),
            "patient_id": patient.get("PATIENT_ID", patient.get("Patient_ID", "Unknown"))
        }
    
    def _extract_labs(self, patient: Dict) -> Dict:
        """Extract latest lab values"""
        labs_list = patient.get("labs", patient.get("Lab_Results", []))
        
        if not labs_list:
            return {
                "wbc": 5.0,
                "hemoglobin": 12.0,
                "platelet": 200.0,
                "creatinine": 1.0,
                "albumin": 3.5
            }
        
        latest = labs_list[-1]
        
        wbc = latest.get("wbc", latest.get("White_Blood_Cell_Count", 5.0))
        hgb = latest.get("hgb", latest.get("Hemoglobin", 12.0))
        platelet = latest.get("platelet", latest.get("Platelets", 200.0))
        creatinine = latest.get("creatinine", 1.0)
        albumin = latest.get("albumin", 3.5)
        
        tumor_markers = {}
        for key in ["tumor_marker_ca125", "tumor_marker_cea", "tumor_marker_afp", "Tumor_Markers"]:
            if key in latest:
                tumor_markers[key] = latest[key]
        
        return {
            "wbc": float(wbc),
            "hemoglobin": float(hgb),
            "platelet": float(platelet),
            "creatinine": float(creatinine),
            "albumin": float(albumin),
            "tumor_markers": tumor_markers,
            "lab_date": latest.get("charttime", latest.get("Date", "Unknown"))
        }
    
    def _extract_treatment_history(self, patient: Dict) -> Dict:
        """Extract treatment history"""
        if "medications" in patient:
            return {
                "chemo_cycles": len([m for m in patient["medications"] if m.get("class") != "supportive"]),
                "medications": patient["medications"],
                "last_chemo_date": self._get_last_chemo_date(patient["medications"])
            }
        
        treatment_logs = patient.get("Treatment_Logs", {})
        chemo_cycles = treatment_logs.get("Chemo_Cycles", [])
        
        return {
            "chemo_cycles": len(chemo_cycles),
            "medications": [],
            "last_chemo_date": chemo_cycles[-1].get("Date") if chemo_cycles else None
        }
    
    def _get_last_chemo_date(self, medications: List[Dict]) -> Optional[str]:
        """Get date of most recent chemotherapy"""
        chemo = [m for m in medications if m.get("class") not in ["supportive", None]]
        if not chemo:
            return None
        return max((m.get("starttime", "") for m in chemo), default=None)
    
    def _extract_oncology_data(self, patient: Dict) -> Dict:
        """Extract oncology-specific data"""
        if "oncology_specific" in patient:
            return patient["oncology_specific"]
        
        if "Patient_Master" in patient:
            return {
                "cancer_type": patient["Patient_Master"].get("Cancer_Type", "Unknown"),
                "stage": "Unknown",
                "grade": "Unknown"
            }
        
        return {
            "cancer_type": patient.get("CANCER_TYPE", patient.get("Cancer_Type", "Unknown")),
            "stage": patient.get("STAGE", "Unknown")
        }
    
    def _assess_age(self, demographics: Dict) -> tuple:
        """Assess age-related risk"""
        age = demographics.get("age", 60)
        
        if age > 80:
            return 0.25, "Very Advanced Age (>80)"
        elif age > 70:
            return 0.15, "Advanced Age (>70)"
        elif age > 65:
            return 0.05, "Elderly (65-70)"
        
        return 0.0, None
    
    def _assess_wbc(self, labs: Dict) -> tuple:
        """Assess white blood cell count risk"""
        wbc = labs.get("wbc", 5.0)
        
        if wbc < 2.0:
            return 0.40, "Severe Neutropenia (WBC <2.0)"
        elif wbc < 3.0:
            return 0.30, "Neutropenia (WBC <3.0)"
        elif wbc < 4.0:
            return 0.15, "Low WBC (3.0-4.0)"
        
        return 0.0, None
    
    def _assess_hemoglobin(self, labs: Dict) -> tuple:
        """Assess hemoglobin/anemia risk"""
        hgb = labs.get("hemoglobin", 12.0)
        
        if hgb < 8.0:
            return 0.35, "Severe Anemia (Hgb <8.0)"
        elif hgb < 10.0:
            return 0.25, "Anemia (Hgb <10.0)"
        elif hgb < 11.0:
            return 0.10, "Mild Anemia (Hgb <11.0)"
        
        return 0.0, None
    
    def _assess_platelets(self, labs: Dict) -> tuple:
        """Assess platelet count risk"""
        platelet = labs.get("platelet", 200.0)
        
        if platelet < 25:
            return 0.30, "Severe Thrombocytopenia (Plt <25K)"
        elif platelet < 50:
            return 0.20, "Thrombocytopenia (Plt <50K)"
        elif platelet < 100:
            return 0.10, "Mild Thrombocytopenia (Plt <100K)"
        
        return 0.0, None
    
    def _assess_tumor_markers(self, labs: Dict, oncology: Dict) -> tuple:
        """Assess tumor marker levels"""
        tumor_markers = labs.get("tumor_markers", {})
        
        if not tumor_markers:
            return 0.0, None
        
        max_marker = max(tumor_markers.values()) if tumor_markers else 0
        
        if max_marker > 100:
            return 0.25, f"Markedly Elevated Tumor Markers ({max_marker:.1f})"
        elif max_marker > 50:
            return 0.15, f"Elevated Tumor Markers ({max_marker:.1f})"
        elif max_marker > 35:
            return 0.08, f"Mildly Elevated Tumor Markers ({max_marker:.1f})"
        
        return 0.0, None
    
    def _assess_chemo_burden(self, treatment: Dict) -> tuple:
        """Assess chemotherapy burden"""
        cycles = treatment.get("chemo_cycles", 0)
        
        if cycles > 8:
            return 0.20, f"High Cumulative Chemo Exposure ({cycles} cycles)"
        elif cycles > 5:
            return 0.12, f"Moderate Chemo Burden ({cycles} cycles)"
        elif cycles > 3:
            return 0.05, f"Multiple Chemo Cycles ({cycles})"
        
        return 0.0, None
    
    def _assess_renal(self, labs: Dict) -> tuple:
        """Assess renal function"""
        creatinine = labs.get("creatinine", 1.0)
        
        if creatinine > 2.0:
            return 0.20, f"Severe Renal Dysfunction (Cr {creatinine:.1f})"
        elif creatinine > 1.5:
            return 0.12, f"Renal Impairment (Cr {creatinine:.1f})"
        elif creatinine > 1.3:
            return 0.05, f"Mild Renal Impairment (Cr {creatinine:.1f})"
        
        return 0.0, None
    
    def _determine_risk_level(self, score: float, factors: List[str]) -> str:
        """Determine risk level from score and factors"""
        if score >= 0.60 or len(factors) >= 4:
            return RiskLevel.CRITICAL
        elif score >= 0.40 or len(factors) >= 3:
            return RiskLevel.HIGH
        elif score >= 0.25 or len(factors) >= 2:
            return RiskLevel.MODERATE
        
        return RiskLevel.STANDARD
    
    def _generate_recommendations(self, risk_level: str, factors: List[str]) -> List[Dict]:
        """Generate recommendations based on risk level"""
        recommendations = []
        
        if risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            recommendations.append({
                "priority": "HIGH",
                "action": "DEFER_TREATMENT",
                "reason": "High-risk patient - requires careful evaluation before proceeding",
                "timeline": "1-2 weeks"
            })
            recommendations.append({
                "priority": "HIGH",
                "action": "DAILY_MONITORING",
                "reason": "Close hematologic monitoring required",
                "timeline": "Daily CBC until stable"
            })
            recommendations.append({
                "priority": "MEDIUM",
                "action": "CONSULT_SPECIALIST",
                "reason": "Consider oncology specialist review",
                "timeline": "Within 48 hours"
            })
        
        elif risk_level == RiskLevel.MODERATE:
            recommendations.append({
                "priority": "MEDIUM",
                "action": "ENHANCED_MONITORING",
                "reason": "Moderate risk - increased vigilance warranted",
                "timeline": "Twice weekly labs"
            })
            recommendations.append({
                "priority": "LOW",
                "action": "REVIEW_CONTRAINDICATIONS",
                "reason": "Ensure no drug interactions or contraindications",
                "timeline": "Before next treatment"
            })
        
        else:
            recommendations.append({
                "priority": "LOW",
                "action": "STANDARD_CARE",
                "reason": "Patient within normal treatment parameters",
                "timeline": "Standard monitoring"
            })
        
        return recommendations
    
    def compare_with_baseline(self, patient_data: Dict) -> Dict:
        """Compare agentic risk assessment with baseline rule-based"""
        agent_result = self.assess_risk(patient_data)
        
        labs = self._extract_labs(patient_data)
        
        baseline_score = 0.0
        if labs["wbc"] < 3.0:
            baseline_score = 0.7
        elif labs["hemoglobin"] < 10.0:
            baseline_score = 0.5
        else:
            baseline_score = 0.2
        
        baseline_level = "HIGH_RISK" if baseline_score >= 0.5 else "STANDARD"
        
        return {
            "agent_score": agent_result["risk_score"],
            "agent_level": agent_result["risk_level"],
            "baseline_score": baseline_score,
            "baseline_level": baseline_level,
            "agreement": agent_result["risk_level"] == baseline_level,
            "agent_added_sensitivity": len(agent_result["risk_factors"]) > 0
        }


@dataclass
class RiskAssessmentResult:
    """Structured result for risk assessment"""
    patient_id: str
    risk_score: float
    risk_level: str
    risk_factors: List[str]
    requires_review: bool
    recommendations: List[Dict]
    assessment_id: str
    timestamp: str
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RiskAssessmentResult':
        return cls(
            patient_id=data.get("patient_id", "Unknown"),
            risk_score=data.get("risk_score", 0.0),
            risk_level=data.get("risk_level", "STANDARD"),
            risk_factors=data.get("risk_factors", []),
            requires_review=data.get("requires_review", False),
            recommendations=data.get("recommendations", []),
            assessment_id=data.get("assessment_id", ""),
            timestamp=data.get("timestamp", "")
        )


if __name__ == "__main__":
    agent = RiskAgent()
    
    test_patient = {
        "patient_id": "TEST-001",
        "demography": {"age": 72, "gender": "M"},
        "labs": [{
            "wbc": 2.8,
            "hgb": 9.5,
            "platelet": 180.0,
            "creatinine": 1.2,
            "albumin": 3.2,
            "tumor_marker_ca125": 45.0
        }],
        "medications": [
            {"drug": "Carboplatin", "class": "platinum"},
            {"drug": "Paclitaxel", "class": "taxane"}
        ],
        "oncology_specific": {
            "cancer_type": "NSCLC",
            "stage": "III",
            "performance_status": "ECOG 1"
        }
    }
    
    result = agent.assess_risk(test_patient)
    print(f"Risk Assessment for {result['patient_id']}")
    print(f"  Score: {result['risk_score']}")
    print(f"  Level: {result['risk_level']}")
    print(f"  Factors: {result['risk_factors']}")
    print(f"  Requires Review: {result['requires_review']}")
