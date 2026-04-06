"""
Treatment Agent - Specialized agent for oncology treatment recommendations
Uses NCCN guidelines and clinical protocols
"""
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


class TreatmentAgent:
    """Specialized agent for generating oncology treatment recommendations"""
    
    NCCN_GUIDELINES = {
        "NSCLC": {
            "first_line": {
                "PDL1_high": ["Pembrolizumab"],
                "PDL1_low": ["Platinum doublet chemo", "Pembrolizumab + chemo"],
                "EGFR_mutation": ["Osimertinib", "Erlotinib"],
                "ALK_positive": ["Alectinib", "Lorlatinib"]
            },
            "second_line": ["Docetaxel", "Ramucirumab + Docetaxel", "Nivolumab"],
            "contraindications": {
                "WBC < 3.0": "Hold chemotherapy until ANC > 1500",
                "Platelets < 75": "Delay treatment, consider transfusion",
                "Creatinine > 2.0": "Dose adjust or hold nephrotoxic agents"
            }
        },
        "Breast Cancer": {
            "first_line": {
                "HER2_positive": ["Trastuzumab", "Pertuzumab", "Taxane"],
                "HER2_negative": ["AC-T", "TC", "Endocrine therapy if HR+"],
                "TNBC": ["Chemo + Pembrolizumab"]
            },
            "contraindications": {
                "EF < 50%": "Hold trastuzumab, cardio workup",
                "WBC < 3.0": "Growth factor support or dose reduction"
            }
        },
        "Colorectal": {
            "first_line": ["FOLFOX", "FOLFIRI", "CAPOX"],
            "targeted": ["Bevacizumab", "Cetuximab", "Panitumumab"],
            "contraindications": {
                "KRAS_mutated": "Avoid anti-EGFR agents",
                "WBC < 3.0": "Delay treatment"
            }
        },
        "Prostate": {
            "first_line": ["ADT", "ADT + Docetaxel", "ADT + Abiraterone"],
            "castration_resistant": ["Enzalutamide", "Abiraterone", "Radium-223"],
            "contraindications": {
                "WBC < 3.0": "Hold chemotherapy, consider ADT alone"
            }
        },
        "Pancreatic": {
            "first_line": ["FOLFIRINOX", "Gemcitabine + Nab-paclitaxel"],
            "adjuvant": ["Gemcitabine", "S-1", "Adjuvant CAPOX"],
            "contraindications": {
                "CA19-9 > 1000": "Consider imaging, may indicate progression",
                "WBC < 3.0": "Dose reduce or hold"
            }
        },
        "Ovarian": {
            "first_line": ["Carboplatin + Paclitaxel", "Carboplatin + Gemcitabine"],
            "BRCA_mutation": ["Olaparib", "Niraparib", "Rucaparib"],
            "contraindications": {
                "WBC < 3.0": "Growth factor support recommended"
            }
        },
        "Melanoma": {
            "first_line": ["Immunotherapy (PD-1)", "BRAF/MEK inhibitors if mutated"],
            "contraindications": {
                "Autoimmune disease": "Avoid immunotherapy",
                "High steroid dose": "Wean steroids before immunotherapy"
            }
        }
    }
    
    DRUG_PROTOCOLS = {
        "Carboplatin": {"AUC_target": 5-6, "nephrotoxicity": True, "myelosuppression": "moderate"},
        "Cisplatin": {"AUC_target": None, "nephrotoxicity": True, "myelosuppression": "high"},
        "Paclitaxel": {"infusion_hours": 3, "premedication": "Steroid + antihistamine", "neurotoxicity": True},
        "Docetaxel": {"infusion_hours": 1, "premedication": "Steroid", "myelosuppression": "moderate"},
        "Pembrolizumab": {"cycle_days": 21, "monitoring": "immune-related AEs"},
        "Trastuzumab": {"cardiac_monitoring": True, "infusion_reaction": True},
        "Gemcitabine": {"weekly_dosing": True, "radiation_sensitizer": True},
        "FOLFIRI": {"schedule": "biweekly", "toxicities": ["diarrhea", "neutropenia"]},
        "FOLFOX": {"schedule": "biweekly", "toxicities": ["neuropathy", "neutropenia"]}
    }
    
    def __init__(self):
        self.default_cycle_length = 21
    
    def generate_recommendation(self, patient_data: Dict, risk_assessment: Dict) -> Dict:
        """
        Generate treatment recommendation based on patient data and risk assessment.
        
        Args:
            patient_data: Patient clinical data
            risk_assessment: Risk assessment from RiskAgent
            
        Returns:
            Treatment recommendation with protocol and rationale
        """
        recommendation_id = str(uuid.uuid4())
        
        oncology = self._extract_oncology_data(patient_data)
        labs = self._extract_labs(patient_data)
        treatment_history = self._extract_treatment_history(patient_data)
        
        cancer_type = oncology.get("cancer_type", "Unknown")
        stage = oncology.get("stage", "Unknown")
        treatment_intent = oncology.get("treatment_intent", "Palliative")
        performance_status = oncology.get("performance_status", "ECOG 1")
        
        risk_level = risk_assessment.get("risk_level", "STANDARD")
        risk_score = risk_assessment.get("risk_score", 0.0)
        
        is_safe = self._check_treatment_safety(labs, risk_level)
        
        if not is_safe["safe"]:
            return self._generate_deferred_recommendation(
                is_safe, cancer_type, stage, labs, recommendation_id
            )
        
        recommended_regimen = self._select_regimen(
            cancer_type, stage, treatment_intent, treatment_history, performance_status
        )
        
        dosing = self._calculate_dosing(
            recommended_regimen["regimen"],
            labs,
            patient_data
        )
        
        supportive_care = self._generate_supportive_care(
            recommended_regimen["regimen"],
            labs,
            risk_level
        )
        
        monitoring = self._generate_monitoring_plan(
            recommended_regimen["regimen"],
            risk_level,
            labs
        )
        
        return {
            "recommendation_id": recommendation_id,
            "timestamp": datetime.now().isoformat(),
            "patient_id": patient_data.get("patient_id", "Unknown"),
            
            "cancer_type": cancer_type,
            "stage": stage,
            "treatment_intent": treatment_intent,
            "performance_status": performance_status,
            
            "risk_assessment": {
                "level": risk_level,
                "score": risk_score
            },
            
            "treatment_recommendation": {
                "regimen": recommended_regimen["regimen"],
                "alternatives": recommended_regimen.get("alternatives", []),
                "rationale": recommended_regimen.get("rationale", ""),
                "intent": treatment_intent
            },
            
            "dosing": dosing,
            "supportive_care": supportive_care,
            "monitoring": monitoring,
            
            "approval_status": "PENDING_REVIEW" if risk_level in ["HIGH_RISK", "MODERATE_RISK"] else "AUTO_APPROVED",
            "next_action": self._determine_next_action(risk_level, is_safe)
        }
    
    def _extract_oncology_data(self, patient: Dict) -> Dict:
        """Extract oncology-specific data"""
        if "oncology_specific" in patient:
            return patient["oncology_specific"]
        
        if "Patient_Master" in patient:
            return {
                "cancer_type": patient["Patient_Master"].get("Cancer_Type", "Unknown")
            }
        
        return {
            "cancer_type": patient.get("CANCER_TYPE", patient.get("Cancer_Type", "Unknown")),
            "stage": patient.get("STAGE", "Unknown")
        }
    
    def _extract_labs(self, patient: Dict) -> Dict:
        """Extract lab values for treatment decision"""
        labs_list = patient.get("labs", patient.get("Lab_Results", []))
        
        if not labs_list:
            return {"wbc": 5.0, "hgb": 12.0, "platelet": 200, "creatinine": 1.0}
        
        latest = labs_list[-1]
        
        return {
            "wbc": float(latest.get("wbc", latest.get("White_Blood_Cell_Count", 5.0))),
            "hgb": float(latest.get("hgb", latest.get("Hemoglobin", 12.0))),
            "platelet": float(latest.get("platelet", latest.get("Platelets", 200))),
            "creatinine": float(latest.get("creatinine", 1.0))
        }
    
    def _extract_treatment_history(self, patient: Dict) -> Dict:
        """Extract prior treatment history"""
        if "medications" in patient:
            chemo = [m for m in patient["medications"] if m.get("class") != "supportive"]
            return {
                "prior_regimens": [m["drug"] for m in chemo],
                "chemo_count": len(chemo)
            }
        
        treatment = patient.get("Treatment_Logs", {})
        cycles = treatment.get("Chemo_Cycles", [])
        
        return {
            "prior_regimens": [c.get("Drug", "Unknown") for c in cycles],
            "chemo_count": len(cycles)
        }
    
    def _check_treatment_safety(self, labs: Dict, risk_level: str) -> Dict:
        """Check if treatment is safe given current labs"""
        warnings = []
        hold_required = False
        
        if labs["wbc"] < 3.0:
            warnings.append(f"WBC ({labs['wbc']}) below safe threshold for chemotherapy")
            hold_required = True
        
        if labs["hgb"] < 8.0:
            warnings.append(f"Hemoglobin ({labs['hgb']}) critically low - transfusion may be needed")
            hold_required = True
        
        if labs["platelet"] < 50:
            warnings.append(f"Platelets ({labs['platelet']}) below threshold")
            hold_required = True
        
        if labs["creatinine"] > 2.0:
            warnings.append(f"Creatinine elevated ({labs['creatinine']}) - requires dose adjustment")
            hold_required = True
        
        return {
            "safe": not hold_required,
            "warnings": warnings,
            "can_proceed_with_support": len(warnings) <= 2 and labs["wbc"] >= 2.5
        }
    
    def _select_regimen(
        self,
        cancer_type: str,
        stage: str,
        treatment_intent: str,
        treatment_history: Dict,
        performance_status: str
    ) -> Dict:
        """Select appropriate treatment regimen based on guidelines"""
        
        guidelines = self.NCCN_GUIDELINES.get(cancer_type, {})
        
        if not guidelines:
            return {
                "regimen": "Standard of care per institutional guidelines",
                "alternatives": [],
                "rationale": "Cancer type-specific guidelines not available"
            }
        
        if treatment_history["chemo_count"] == 0:
            first_line = guidelines.get("first_line", {})
            
            if isinstance(first_line, list):
                regimen = first_line[0]
            else:
                regimen = first_line.get("PDL1_high", ["Platinum doublet"])[0] if isinstance(first_line.get("PDL1_high"), list) else "Platinum doublet"
            
            return {
                "regimen": regimen,
                "alternatives": first_line.get("PDL1_low", []) if isinstance(first_line, dict) else [],
                "rationale": f"First-line therapy for {cancer_type} stage {stage}"
            }
        else:
            second_line = guidelines.get("second_line", [])
            if isinstance(second_line, list) and second_line:
                regimen = second_line[0]
            else:
                regimen = "Alternative regimen per guidelines"
            
            return {
                "regimen": regimen,
                "alternatives": second_line[1:] if isinstance(second_line, list) else [],
                "rationale": f"Second-line therapy for {cancer_type} after {treatment_history['chemo_count']} prior cycles"
            }
    
    def _calculate_dosing(self, regimen: str, labs: Dict, patient: Dict) -> Dict:
        """Calculate appropriate drug doses"""
        dosing = {
            "regimen": regimen,
            "cycle_length_days": self.default_cycle_length,
            "drugs": []
        }
        
        if "Carboplatin" in regimen:
            dosing["drugs"].append({
                "drug": "Carboplatin",
                "dose": "AUC 5-6",
                "calculation": "Calvert formula based on GFR"
            })
        
        if "Paclitaxel" in regimen:
            dosing["drugs"].append({
                "drug": "Paclitaxel",
                "dose": "175 mg/m²",
                "infusion_time": "3 hours"
            })
        
        if "Pembro" in regimen or "Pembrolizumab" in regimen:
            dosing["drugs"].append({
                "drug": "Pembrolizumab",
                "dose": "200 mg flat dose",
                "frequency": "Every 3 weeks"
            })
        
        if labs["wbc"] < 4.0 or labs["platelet"] < 150:
            dosing["dose_reduction"] = "10-20% dose reduction may be considered"
        
        if labs["creatinine"] > 1.5:
            dosing["renal_adjustment"] = "Consider dose reduction for nephrotoxic agents"
        
        return dosing
    
    def _generate_supportive_care(self, regimen: str, labs: Dict, risk_level: str) -> List[Dict]:
        """Generate supportive care recommendations"""
        supportive = []
        
        if labs["wbc"] < 3.5:
            supportive.append({
                "type": "Growth Factor",
                "agent": "Pegfilgrastim",
                "timing": "24-72 hours post-chemo",
                "indication": "WBC below threshold"
            })
        
        if labs["hgb"] < 10.0:
            supportive.append({
                "type": "Transfusion",
                "agent": "PRBC",
                "threshold": "Hgb < 8.0 or symptomatic",
                "indication": "Anemia"
            })
        
        if "Paclitaxel" in regimen or "Docetaxel" in regimen:
            supportive.append({
                "type": "Premedication",
                "agents": ["Dexamethasone 8-12mg BID x3 days", "Diphenhydramine 50mg", "Famotidine 20mg"],
                "timing": "Before taxane infusion"
            })
        
        supportive.extend([
            {"type": "Antiemetic", "regimen": "NK1 antagonist + 5-HT3 + Dexamethasone"},
            {"type": "Hydration", "agents": ["1-2L IV fluids"], "indication": "For platinum agents"}
        ])
        
        return supportive
    
    def _generate_monitoring_plan(self, regimen: str, risk_level: str, labs: Dict) -> List[Dict]:
        """Generate monitoring plan recommendations"""
        monitoring = []
        
        if risk_level in ["HIGH_RISK", "CRITICAL"]:
            monitoring.append({
                "parameter": "CBC with differential",
                "frequency": "Daily",
                "duration": "Until recovery or stable"
            })
        else:
            monitoring.append({
                "parameter": "CBC with differential",
                "frequency": "Weekly",
                "duration": "During active treatment"
            })
        
        monitoring.extend([
            {"parameter": "Comprehensive metabolic panel", "frequency": "Every 2-3 cycles"},
            {"parameter": "Tumor markers", "frequency": "Every 2 cycles"},
            {"parameter": "Imaging (CT)", "frequency": "Every 2-3 cycles"}
        ])
        
        if "Trastuzumab" in regimen:
            monitoring.append({
                "parameter": "LVEF (Echocardiogram)",
                "frequency": "Every 3 months",
                "indication": "Cardiac monitoring for HER2-targeted therapy"
            })
        
        return monitoring
    
    def _generate_deferred_recommendation(
        self,
        safety: Dict,
        cancer_type: str,
        stage: str,
        labs: Dict,
        recommendation_id: str
    ) -> Dict:
        """Generate recommendation when treatment must be deferred"""
        return {
            "recommendation_id": recommendation_id,
            "timestamp": datetime.now().isoformat(),
            "treatment_recommendation": {
                "regimen": "DEFERRED",
                "rationale": "Treatment safety criteria not met",
                "hold_reason": safety["warnings"]
            },
            "deferred_plan": {
                "action": "Address safety parameters",
                "timeline": "Re-evaluate in 1-2 weeks",
                "target_wbc": "≥ 3.0 K/uL",
                "target_hgb": "≥ 10.0 g/dL",
                "target_platelet": "≥ 75 K/uL"
            },
            "interim_care": [
                "Transfusion support as needed",
                "Growth factor support if appropriate",
                "Close monitoring of hematologic parameters",
                "Infection precautions"
            ],
            "approval_status": "HOLD",
            "next_action": "Re-assess when labs recover"
        }
    
    def _determine_next_action(self, risk_level: str, safety: Dict) -> str:
        """Determine the next action based on risk and safety"""
        if not safety["safe"]:
            return "HOLD_TREATMENT"
        elif risk_level == "CRITICAL":
            return "ONCOLOGY_CONSULT"
        elif risk_level == "HIGH_RISK":
            return "PHYSICIAN_REVIEW"
        else:
            return "PROCEED_TO_TREATMENT"


@dataclass
class TreatmentRecommendation:
    """Structured treatment recommendation"""
    recommendation_id: str
    regimen: str
    dosing: Dict
    supportive_care: List[Dict]
    monitoring: List[Dict]
    approval_status: str
    timestamp: str


if __name__ == "__main__":
    agent = TreatmentAgent()
    
    test_patient = {
        "patient_id": "TEST-001",
        "oncology_specific": {
            "cancer_type": "NSCLC",
            "stage": "IV",
            "treatment_intent": "Palliative",
            "performance_status": "ECOG 1"
        },
        "labs": [{
            "wbc": 5.5,
            "hgb": 11.5,
            "platelet": 220,
            "creatinine": 1.0
        }],
        "medications": []
    }
    
    test_risk = {
        "risk_level": "STANDARD",
        "risk_score": 0.2
    }
    
    result = agent.generate_recommendation(test_patient, test_risk)
    print(f"Treatment Recommendation for {result['patient_id']}")
    print(f"  Regimen: {result['treatment_recommendation']['regimen']}")
    print(f"  Status: {result['approval_status']}")
