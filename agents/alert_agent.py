"""
Alert Agent - Specialized agent for clinical alerts and early warning
Monitors patient status and generates actionable alerts
"""
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class AlertCategory(Enum):
    """Alert category types"""
    LAB_ABNORMALITY = "LAB_ABNORMALITY"
    DRUG_INTERACTION = "DRUG_INTERACTION"
    CONTRAINDICATION = "CONTRAINDICATION"
    VITAL_SIGN = "VITAL_SIGN"
    TREATMENT_DELAY = "TREATMENT_DELAY"
    EARLY_DETECTION = "EARLY_DETECTION"
    SAFETY = "SAFETY"
    COMMUNICATION = "COMMUNICATION"


class AlertAgent:
    """Specialized agent for generating clinical alerts and early warnings"""
    
    CRITICAL_THRESHOLDS = {
        "wbc": {"critical_low": 2.0, "low": 3.0, "high": 30.0},
        "hemoglobin": {"critical_low": 7.0, "low": 10.0, "high": 20.0},
        "platelet": {"critical_low": 20.0, "low": 50.0, "high": 1000.0},
        "creatinine": {"critical_high": 3.0, "high": 1.5},
        "potassium": {"critical_low": 2.5, "low": 3.0, "high": 6.0},
        "sodium": {"critical_low": 120, "low": 130, "high": 155},
        "temperature": {"critical_high": 39.5, "high": 38.5},
        "heart_rate": {"critical_high": 150, "high": 110, "low": 40},
        "sbp": {"critical_low": 80, "low": 90, "high": 200},
        "spo2": {"critical_low": 88, "low": 92}
    }
    
    DRUG_INTERACTIONS = {
        ("Cisplatin", "Aminoglycoside"): {
            "severity": "CRITICAL",
            "description": "Increased nephrotoxicity and ototoxicity risk"
        },
        ("Trastuzumab", "Anthracycline"): {
            "severity": "HIGH",
            "description": "Increased cardiotoxicity risk - sequential preferred"
        },
        ("5-FU", "Methotrexate"): {
            "severity": "MEDIUM",
            "description": "Increased toxicity with high-dose methotrexate"
        },
        ("Warfarin", "Many chemo agents"): {
            "severity": "HIGH",
            "description": "Multiple agents affect warfarin metabolism"
        },
        ("NSAIDs", "Cisplatin"): {
            "severity": "HIGH",
            "description": "NSAIDs may reduce cisplatin renal clearance"
        },
        ("ACE Inhibitors", "Lithium"): {
            "severity": "MEDIUM",
            "description": "ACE inhibitors may increase lithium levels"
        },
        ("Metformin", "IV Contrast"): {
            "severity": "HIGH",
            "description": "Hold metformin before contrast studies"
        }
    }
    
    CONTRAINDICATIONS = {
        "Paclitaxel": {
            "absolute": ["Hypersensitivity to Cremophor"],
            "relative": ["Baseline neuropathy G2+", "EF < 50%"]
        },
        "Pembrolizumab": {
            "absolute": ["Active autoimmune disease", "Organ transplant recipient"],
            "relative": ["Chronic steroid use >10mg pred", "Active infection"]
        },
        "Trastuzumab": {
            "absolute": ["Baseline EF < 50%"],
            "relative": ["History of CHF", "Concomitant anthracycline"]
        },
        "Cisplatin": {
            "absolute": ["Pre-existing renal failure", "Hearing loss"],
            "relative": ["CrCl < 60 mL/min", "Pre-existing neuropathy"]
        },
        "Anthracyclines": {
            "absolute": ["EF < 50%", "Recent MI"],
            "relative": ["Cumulative lifetime dose limits", "Liver dysfunction"]
        }
    }
    
    EARLY_DETECTION_PATTERNS = {
        "sepsis": {
            "indicators": ["fever > 38.3", "wbc < 4 or > 12", "heart_rate > 90", "rr > 20", "spo2 < 94"],
            "min_indicators": 3
        },
        "tumor_lysis": {
            "indicators": ["uric_acid > 8", "potassium > 6", "phosphate > 4.5", "creatinine_increase > 0.3"],
            "min_indicators": 2
        },
        "neutropenic_fever": {
            "indicators": ["wbc < 1", "temperature > 38.3", "anc < 500"],
            "min_indicators": 2
        },
        "cardiac_toxicity": {
            "indicators": ["ef_decline > 10%", "ef < 50%", "bnp_elevation"],
            "min_indicators": 1
        }
    }
    
    def __init__(self):
        self.alert_history: List[Dict] = []
    
    def generate_alerts(self, patient_data: Dict, risk_assessment: Dict, treatment_recommendation: Dict) -> Dict:
        """
        Generate comprehensive alerts for a patient.
        
        Args:
            patient_data: Patient clinical data
            risk_assessment: Risk assessment from RiskAgent
            treatment_recommendation: Treatment recommendation from TreatmentAgent
            
        Returns:
            Alert summary with all generated alerts
        """
        alert_id = str(uuid.uuid4())
        
        all_alerts = []
        
        lab_alerts = self._check_lab_alerts(patient_data)
        all_alerts.extend(lab_alerts)
        
        vital_alerts = self._check_vital_alerts(patient_data)
        all_alerts.extend(vital_alerts)
        
        drug_alerts = self._check_drug_interactions(patient_data, treatment_recommendation)
        all_alerts.extend(drug_alerts)
        
        contraindication_alerts = self._check_contraindications(patient_data, treatment_recommendation)
        all_alerts.extend(contraindication_alerts)
        
        early_detection_alerts = self._check_early_detection(patient_data)
        all_alerts.extend(early_detection_alerts)
        
        risk_alerts = self._check_risk_alerts(risk_assessment)
        all_alerts.extend(risk_alerts)
        
        all_alerts.sort(key=lambda x: (
            AlertSeverity[x["severity"]].value,
            AlertCategory[x["category"]].value
        ))
        
        self._record_alerts(all_alerts, patient_data.get("patient_id", "Unknown"))
        
        critical_count = sum(1 for a in all_alerts if a["severity"] == "CRITICAL")
        high_count = sum(1 for a in all_alerts if a["severity"] == "HIGH")
        
        return {
            "alert_id": alert_id,
            "timestamp": datetime.now().isoformat(),
            "patient_id": patient_data.get("patient_id", "Unknown"),
            
            "summary": {
                "total_alerts": len(all_alerts),
                "critical_count": critical_count,
                "high_count": high_count,
                "requires_immediate_action": critical_count > 0 or high_count > 2,
                "overall_risk": self._calculate_overall_risk(all_alerts, risk_assessment)
            },
            
            "alerts": all_alerts,
            
            "priorities": {
                "immediate": [a for a in all_alerts if a["severity"] in ["CRITICAL"]],
                "today": [a for a in all_alerts if a["severity"] == "HIGH"],
                "this_week": [a for a in all_alerts if a["severity"] in ["MEDIUM", "LOW"]]
            },
            
            "action_items": self._generate_action_items(all_alerts),
            "escalation": self._determine_escalation(all_alerts)
        }
    
    def _check_lab_alerts(self, patient: Dict) -> List[Dict]:
        """Check for abnormal lab values"""
        alerts = []
        labs_list = patient.get("labs", patient.get("Lab_Results", []))
        
        if not labs_list:
            return alerts
        
        latest = labs_list[-1]
        
        thresholds = self.CRITICAL_THRESHOLDS
        
        wbc = latest.get("wbc", latest.get("White_Blood_Cell_Count"))
        if wbc:
            if wbc < thresholds["wbc"]["critical_low"]:
                alerts.append(self._create_alert(
                    "CRITICAL", AlertCategory.LAB_ABNORMALITY,
                    f"SEVERE NEUTROPENIA: WBC {wbc}",
                    f"WBC critically low at {wbc} K/uL - high infection risk",
                    "Notify physician immediately. Consider growth factor."
                ))
            elif wbc < thresholds["wbc"]["low"]:
                alerts.append(self._create_alert(
                    "HIGH", AlertCategory.LAB_ABNORMALITY,
                    f"Neutropenia: WBC {wbc}",
                    f"WBC below chemotherapy threshold",
                    "Hold chemotherapy until ANC > 1500"
                ))
        
        hgb = latest.get("hgb", latest.get("Hemoglobin"))
        if hgb:
            if hgb < thresholds["hemoglobin"]["critical_low"]:
                alerts.append(self._create_alert(
                    "CRITICAL", AlertCategory.LAB_ABNORMALITY,
                    f"SEVERE ANEMIA: Hgb {hgb}",
                    f"Hemoglobin critically low at {hgb} g/dL",
                    "Transfusion likely needed. Notify physician."
                ))
            elif hgb < thresholds["hemoglobin"]["low"]:
                alerts.append(self._create_alert(
                    "MEDIUM", AlertCategory.LAB_ABNORMALITY,
                    f"Anemia: Hgb {hgb}",
                    f"Hemoglobin below threshold",
                    "Monitor, consider transfusion if symptomatic"
                ))
        
        platelet = latest.get("platelet", latest.get("Platelets"))
        if platelet:
            if platelet < thresholds["platelet"]["critical_low"]:
                alerts.append(self._create_alert(
                    "CRITICAL", AlertCategory.LAB_ABNORMALITY,
                    f"SEVERE THROMBOCYTOPENIA: Platelets {platelet}",
                    f"Platelet count critically low",
                    "Bleeding precautions. Transfusion likely needed."
                ))
            elif platelet < thresholds["platelet"]["low"]:
                alerts.append(self._create_alert(
                    "HIGH", AlertCategory.LAB_ABNORMALITY,
                    f"Thrombocytopenia: Platelets {platelet}",
                    f"Platelet count below threshold",
                    "Hold anticoagulation. Monitor for bleeding."
                ))
        
        creatinine = latest.get("creatinine")
        if creatinine and creatinine > thresholds["creatinine"]["critical_high"]:
            alerts.append(self._create_alert(
                "CRITICAL", AlertCategory.LAB_ABNORMALITY,
                f"ACUTE KIDNEY INJURY: Creatinine {creatinine}",
                f"Creatinine significantly elevated",
                "Hold nephrotoxic drugs. Evaluate for intervention."
            ))
        
        return alerts
    
    def _check_vital_alerts(self, patient: Dict) -> List[Dict]:
        """Check for abnormal vital signs"""
        alerts = []
        vitals_list = patient.get("vitals", [])
        
        if not vitals_list:
            return alerts
        
        latest = vitals_list[-1]
        thresholds = self.CRITICAL_THRESHOLDS
        
        temp = latest.get("temperature")
        if temp and temp > thresholds["temperature"]["critical_high"]:
            alerts.append(self._create_alert(
                "HIGH", AlertCategory.VITAL_SIGN,
                f"High Fever: {temp}°C",
                "Temperature exceeding threshold",
                "Rule out infection. Consider neutropenic fever protocol."
            ))
        
        hr = latest.get("heart_rate")
        if hr:
            if hr > thresholds["heart_rate"]["critical_high"]:
                alerts.append(self._create_alert(
                    "MEDIUM", AlertCategory.VITAL_SIGN,
                    f"Tachycardia: HR {hr}",
                    "Heart rate elevated",
                    "Monitor. Evaluate for dehydration, infection, or bleeding."
                ))
            elif hr < thresholds["heart_rate"]["low"]:
                alerts.append(self._create_alert(
                    "MEDIUM", AlertCategory.VITAL_SIGN,
                    f"Bradycardia: HR {hr}",
                    "Heart rate low",
                    "Evaluate cause. Check medications."
                ))
        
        sbp = latest.get("sbp")
        if sbp and sbp < thresholds["sbp"]["critical_low"]:
            alerts.append(self._create_alert(
                "CRITICAL", AlertCategory.VITAL_SIGN,
                f"Hypotension: SBP {sbp}",
                "Blood pressure critically low",
                "Activate rapid response. Evaluate for shock."
            ))
        
        spo2 = latest.get("spo2")
        if spo2 and spo2 < thresholds["spo2"]["critical_low"]:
            alerts.append(self._create_alert(
                "HIGH", AlertCategory.VITAL_SIGN,
                f"Hypoxemia: SpO2 {spo2}%",
                "Oxygen saturation low",
                "Supplemental oxygen. Evaluate pulmonary status."
            ))
        
        return alerts
    
    def _check_drug_interactions(self, patient: Dict, treatment: Dict) -> List[Dict]:
        """Check for drug-drug interactions"""
        alerts = []
        
        meds = patient.get("medications", [])
        if not meds:
            return alerts
        
        drug_names = set(m.get("drug", "").lower() for m in meds)
        
        for (drug1, drug2), interaction in self.DRUG_INTERACTIONS.items():
            if drug1.lower() in drug_names and drug2.lower() in drug_names:
                alerts.append(self._create_alert(
                    interaction["severity"],
                    AlertCategory.DRUG_INTERACTION,
                    f"Drug Interaction: {drug1} + {drug2}",
                    interaction["description"],
                    "Review therapy. Consider alternative agents."
                ))
        
        return alerts
    
    def _check_contraindications(self, patient: Dict, treatment: Dict) -> List[Dict]:
        """Check for treatment contraindications"""
        alerts = []
        
        regimen = treatment.get("treatment_recommendation", {}).get("regimen", "")
        
        if not regimen or regimen == "DEFERRED":
            return alerts
        
        labs = patient.get("labs", patient.get("Lab_Results", []))
        if labs:
            latest = labs[-1]
            wbc = latest.get("wbc", 5.0)
            hgb = latest.get("hgb", 12.0)
            platelet = latest.get("platelet", 200)
            creatinine = latest.get("creatinine", 1.0)
        else:
            wbc, hgb, platelet, creatinine = 5.0, 12.0, 200, 1.0
        
        for drug, contraindications in self.CONTRAINDICATIONS.items():
            if drug.lower() in regimen.lower():
                if "absolute" in contraindications:
                    for contra in contraindications["absolute"]:
                        if "WBC" in contra and wbc < 3.0:
                            alerts.append(self._create_alert(
                                "CRITICAL", AlertCategory.CONTRAINDICATION,
                                f"CONTRAINDICATED: {drug}",
                                f"Absolute contraindication: {contra}",
                                "DO NOT ADMINISTER. Select alternative therapy."
                            ))
                        if "EF" in contra:
                            alerts.append(self._create_alert(
                                "CRITICAL", AlertCategory.CONTRAINDICATION,
                                f"CONTRAINDICATED: {drug}",
                                f"Absolute contraindication: {contra}",
                                "Cardiac evaluation required before treatment."
                            ))
                
                if "relative" in contraindications:
                    for contra in contraindications["relative"]:
                        alerts.append(self._create_alert(
                            "MEDIUM", AlertCategory.CONTRAINDICATION,
                            f"Use Caution: {drug}",
                            f"Relative contraindication: {contra}",
                            "Monitor closely. May require dose adjustment."
                        ))
        
        return alerts
    
    def _check_early_detection(self, patient: Dict) -> List[Dict]:
        """Check for early detection patterns"""
        alerts = []
        labs_list = patient.get("labs", patient.get("Lab_Results", []))
        vitals_list = patient.get("vitals", [])
        
        if len(labs_list) < 2:
            return alerts
        
        latest_lab = labs_list[-1]
        prev_lab = labs_list[-2] if len(labs_list) > 1 else latest_lab
        
        trends = {
            "wbc_drop": latest_lab.get("wbc", 5) < prev_lab.get("wbc", 5) * 0.7,
            "hgb_drop": latest_lab.get("hgb", 12) < prev_lab.get("hgb", 12) * 0.85,
            "creatinine_rise": latest_lab.get("creatinine", 1) > prev_lab.get("creatinine", 1) * 1.5,
            "tumor_marker_rise": self._check_tumor_marker_trend(labs_list)
        }
        
        if trends["wbc_drop"] and trends["hgb_drop"]:
            alerts.append(self._create_alert(
                "HIGH", AlertCategory.EARLY_DETECTION,
                "Possible Bone Marrow Suppression",
                "Rapid decline in multiple cell lines",
                "Monitor CBC daily. Consider growth factor support."
            ))
        
        if trends["creatinine_rise"]:
            alerts.append(self._create_alert(
                "MEDIUM", AlertCategory.EARLY_DETECTION,
                "Possible Tumor Lysis Syndrome",
                "Rising creatinine suggests tumor lysis",
                "Check uric acid, LDH, phosphate. Aggressive hydration."
            ))
        
        if trends["tumor_marker_rise"]:
            alerts.append(self._create_alert(
                "MEDIUM", AlertCategory.EARLY_DETECTION,
                "Tumor Marker Elevation",
                "Rapid increase in tumor markers",
                "Consider imaging to evaluate disease progression."
            ))
        
        return alerts
    
    def _check_tumor_marker_trend(self, labs_list: List[Dict]) -> bool:
        """Check if tumor markers are rising rapidly"""
        if len(labs_list) < 3:
            return False
        
        tumor_keys = ["tumor_marker_ca125", "tumor_marker_cea", "tumor_marker_afp"]
        
        for key in tumor_keys:
            values = [l.get(key, 0) for l in labs_list[-3:] if l.get(key, 0) > 0]
            if len(values) >= 3 and values[-1] > values[0] * 1.3:
                return True
        
        return False
    
    def _check_risk_alerts(self, risk_assessment: Dict) -> List[Dict]:
        """Generate alerts based on risk assessment"""
        alerts = []
        
        risk_level = risk_assessment.get("risk_level", "STANDARD")
        risk_factors = risk_assessment.get("risk_factors", [])
        
        if risk_level in ["CRITICAL", "HIGH"]:
            alerts.append(self._create_alert(
                "HIGH" if risk_level == "HIGH" else "CRITICAL",
                AlertCategory.SAFETY,
                f"High Risk Patient: {risk_level}",
                f"Risk factors: {', '.join(risk_factors)}",
                "Enhanced monitoring required. Physician notification recommended."
            ))
        
        if risk_assessment.get("requires_review", False):
            alerts.append(self._create_alert(
                "MEDIUM", AlertCategory.SAFETY,
                "Case Review Required",
                "Patient flagged for clinical review",
                "Submit for multidisciplinary team review."
            ))
        
        return alerts
    
    def _create_alert(
        self,
        severity: str,
        category: AlertCategory,
        title: str,
        description: str,
        recommendation: str
    ) -> Dict:
        """Create a structured alert"""
        return {
            "alert_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "category": category.value,
            "title": title,
            "description": description,
            "recommendation": recommendation,
            "acknowledged": False
        }
    
    def _calculate_overall_risk(self, alerts: List[Dict], risk_assessment: Dict) -> str:
        """Calculate overall risk level from alerts"""
        critical_count = sum(1 for a in alerts if a["severity"] == "CRITICAL")
        high_count = sum(1 for a in alerts if a["severity"] == "HIGH")
        
        if critical_count > 0:
            return "CRITICAL"
        elif high_count > 2:
            return "HIGH"
        elif high_count > 0 or risk_assessment.get("risk_level") in ["HIGH_RISK", "CRITICAL"]:
            return "MODERATE"
        else:
            return "LOW"
    
    def _generate_action_items(self, alerts: List[Dict]) -> List[Dict]:
        """Generate prioritized action items from alerts"""
        actions = []
        
        for alert in alerts:
            if alert["severity"] in ["CRITICAL", "HIGH"]:
                actions.append({
                    "priority": alert["severity"],
                    "action": alert["recommendation"],
                    "reason": alert["title"],
                    "due": "IMMEDIATE" if alert["severity"] == "CRITICAL" else "Within 4 hours"
                })
        
        return actions
    
    def _determine_escalation(self, alerts: List[Dict]) -> Dict:
        """Determine escalation path based on alerts"""
        critical_count = sum(1 for a in alerts if a["severity"] == "CRITICAL")
        
        if critical_count > 0:
            return {
                "level": "IMMEDIATE",
                "contact": "Attending Physician / Rapid Response Team",
                "path": "Direct phone call + documentation"
            }
        
        high_count = sum(1 for a in alerts if a["severity"] == "HIGH")
        if high_count > 0:
            return {
                "level": "URGENT",
                "contact": "Primary Oncology Team",
                "path": "Page + verbal communication within 4 hours"
            }
        
        return {
            "level": "ROUTINE",
            "contact": "Nursing team",
            "path": "Document in chart for next rounds"
        }
    
    def _record_alerts(self, alerts: List[Dict], patient_id: str):
        """Record alerts to history"""
        for alert in alerts:
            self.alert_history.append({
                **alert,
                "patient_id": patient_id
            })
    
    def get_alert_summary(self, patient_id: Optional[str] = None) -> Dict:
        """Get summary of alert history"""
        if patient_id:
            patient_alerts = [a for a in self.alert_history if a.get("patient_id") == patient_id]
        else:
            patient_alerts = self.alert_history
        
        return {
            "total_alerts": len(patient_alerts),
            "by_severity": {
                "CRITICAL": sum(1 for a in patient_alerts if a["severity"] == "CRITICAL"),
                "HIGH": sum(1 for a in patient_alerts if a["severity"] == "HIGH"),
                "MEDIUM": sum(1 for a in patient_alerts if a["severity"] == "MEDIUM"),
                "LOW": sum(1 for a in patient_alerts if a["severity"] == "LOW")
            },
            "by_category": {}
        }


if __name__ == "__main__":
    agent = AlertAgent()
    
    test_patient = {
        "patient_id": "TEST-001",
        "labs": [{
            "wbc": 2.5,
            "hgb": 9.0,
            "platelet": 180,
            "creatinine": 1.2,
            "tumor_marker_ca125": 55
        }],
        "vitals": [{
            "temperature": 38.8,
            "heart_rate": 95,
            "sbp": 110,
            "spo2": 96
        }],
        "medications": [
            {"drug": "Carboplatin", "class": "platinum"},
            {"drug": "Ondansetron", "class": "supportive"}
        ]
    }
    
    test_risk = {
        "risk_level": "HIGH_RISK",
        "risk_score": 0.55,
        "risk_factors": ["Neutropenia", "Anemia"]
    }
    
    test_treatment = {
        "treatment_recommendation": {
            "regimen": "Carboplatin + Paclitaxel"
        }
    }
    
    result = agent.generate_alerts(test_patient, test_risk, test_treatment)
    print(f"Alert Summary for {result['patient_id']}")
    print(f"  Total Alerts: {result['summary']['total_alerts']}")
    print(f"  Critical: {result['summary']['critical_count']}")
    print(f"  Requires Immediate Action: {result['summary']['requires_immediate_action']}")
    print(f"  Overall Risk: {result['summary']['overall_risk']}")
