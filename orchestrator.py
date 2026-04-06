"""
Oncology AI Orchestrator - Unified Patient Journey Analysis
Combines Risk, Treatment, and Alert agents into a single workflow
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from agents.risk_agent import RiskAgent, RiskLevel
from agents.treatment_agent import TreatmentAgent
from agents.alert_agent import AlertAgent, AlertSeverity
from safety.safety_guardrails import SafetyGuardrails, SafetyLevel


@dataclass
class PatientAnalysisResult:
    """Complete analysis result for a patient"""
    analysis_id: str
    timestamp: str
    patient_id: str
    
    risk_assessment: Dict
    treatment_recommendation: Dict
    alert_summary: Dict
    safety_result: Dict
    
    overall_status: str
    recommendations: List[Dict]
    
    @classmethod
    def from_results(
        cls,
        patient_id: str,
        risk_result: Dict,
        treatment_result: Dict,
        alert_result: Dict,
        safety_result
    ) -> 'PatientAnalysisResult':
        """Create result from agent outputs"""
        all_recommendations = []
        
        for rec in risk_result.get("recommendations", []):
            all_recommendations.append({
                "source": "Risk Agent",
                "priority": rec.get("priority", "LOW"),
                "action": rec.get("action", ""),
                "reason": rec.get("reason", "")
            })
        
        treatment_recs = treatment_result.get("treatment_recommendation", {})
        if treatment_recs.get("regimen") != "DEFERRED":
            all_recommendations.append({
                "source": "Treatment Agent",
                "priority": "HIGH",
                "action": f"Proceed with {treatment_recs.get('regimen', 'treatment')}",
                "reason": treatment_recs.get("rationale", "")
            })
        
        if alert_result.get("summary", {}).get("requires_immediate_action"):
            all_recommendations.insert(0, {
                "source": "Alert Agent",
                "priority": "CRITICAL",
                "action": "IMMEDIATE physician notification required",
                "reason": f"{alert_result['summary']['critical_count']} critical alerts"
            })
        
        overall_status = "APPROVED"
        if safety_result.blocked:
            overall_status = "HOLD - Safety Review Required"
        elif alert_result.get("summary", {}).get("critical_count", 0) > 0:
            overall_status = "REVIEW - Critical Alerts Present"
        elif risk_result.get("risk_level") in ["HIGH_RISK", "CRITICAL"]:
            overall_status = "PENDING - High Risk Patient"
        
        return cls(
            analysis_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            patient_id=patient_id,
            risk_assessment=risk_result,
            treatment_recommendation=treatment_result,
            alert_summary=alert_result,
            safety_result=asdict(safety_result) if hasattr(safety_result, '__dataclass_fields__') else safety_result,
            overall_status=overall_status,
            recommendations=all_recommendations
        )


class OncologyOrchestrator:
    """
    Orchestrates the multi-agent oncology AI system.
    Coordinates Risk Agent, Treatment Agent, and Alert Agent.
    """
    
    def __init__(self, strict_mode: bool = True):
        self.risk_agent = RiskAgent()
        self.treatment_agent = TreatmentAgent()
        self.alert_agent = AlertAgent()
        self.safety_guardrails = SafetyGuardrails(strict_mode=strict_mode)
        
        self.analysis_history: List[PatientAnalysisResult] = []
    
    def analyze_patient(self, patient_data: Dict) -> PatientAnalysisResult:
        """
        Run complete patient analysis through all agents.
        
        Args:
            patient_data: Complete patient record
            
        Returns:
            PatientAnalysisResult with all agent outputs
        """
        patient_id = patient_data.get("patient_id", "Unknown")
        
        risk_result = self.risk_agent.assess_risk(patient_data)
        
        treatment_result = self.treatment_agent.generate_recommendation(
            patient_data,
            risk_result
        )
        
        safety_result = self.safety_guardrails.check_all(
            str(treatment_result),
            patient_data
        )
        
        alert_result = self.alert_agent.generate_alerts(
            patient_data,
            risk_result,
            treatment_result
        )
        
        result = PatientAnalysisResult.from_results(
            patient_id=patient_id,
            risk_result=risk_result,
            treatment_result=treatment_result,
            alert_result=alert_result,
            safety_result=safety_result
        )
        
        self.analysis_history.append(result)
        
        return result
    
    def batch_analyze(self, patients: List[Dict]) -> List[PatientAnalysisResult]:
        """Analyze multiple patients"""
        results = []
        for patient in patients:
            try:
                result = self.analyze_patient(patient)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing patient {patient.get('patient_id', 'Unknown')}: {e}")
                results.append(None)
        return results
    
    def get_statistics(self) -> Dict:
        """Get analysis statistics"""
        if not self.analysis_history:
            return {"total_analyzed": 0}
        
        status_counts = {}
        for result in self.analysis_history:
            status = result.overall_status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        high_risk_count = sum(
            1 for r in self.analysis_history
            if r.risk_assessment.get("risk_level") in ["HIGH_RISK", "CRITICAL"]
        )
        
        return {
            "total_analyzed": len(self.analysis_history),
            "by_status": status_counts,
            "high_risk_count": high_risk_count,
            "approval_rate": status_counts.get("APPROVED", 0) / len(self.analysis_history) * 100
        }


def demo_analysis():
    """Demonstrate the orchestrator with sample patients"""
    print("=" * 80)
    print("ONCOLOGY AI ORCHESTRATOR - DEMO ANALYSIS")
    print("=" * 80)
    
    orchestrator = OncologyOrchestrator()
    
    test_patients = [
        {
            "patient_id": "DEMO-001",
            "demography": {"age": 72, "gender": "M"},
            "labs": [{
                "wbc": 2.5,
                "hgb": 9.5,
                "platelet": 180,
                "creatinine": 1.2,
                "tumor_marker_ca125": 45
            }],
            "vitals": [{
                "temperature": 38.2,
                "heart_rate": 88,
                "sbp": 125
            }],
            "medications": [
                {"drug": "Carboplatin", "class": "platinum"},
                {"drug": "Pembrolizumab", "class": "immunotherapy"}
            ],
            "oncology_specific": {
                "cancer_type": "NSCLC",
                "stage": "IV",
                "treatment_intent": "Palliative",
                "performance_status": "ECOG 1"
            }
        },
        {
            "patient_id": "DEMO-002",
            "demography": {"age": 55, "gender": "F"},
            "labs": [{
                "wbc": 6.5,
                "hgb": 12.5,
                "platelet": 250,
                "creatinine": 0.9
            }],
            "vitals": [{
                "temperature": 37.0,
                "heart_rate": 72,
                "sbp": 118
            }],
            "medications": [
                {"drug": "Trastuzumab", "class": "targeted"}
            ],
            "oncology_specific": {
                "cancer_type": "Breast Cancer",
                "stage": "II",
                "treatment_intent": "Curative",
                "performance_status": "ECOG 0"
            }
        }
    ]
    
    for patient in test_patients:
        print(f"\n{'='*60}")
        print(f"Analyzing: {patient['patient_id']}")
        print(f"{'='*60}")
        
        result = orchestrator.analyze_patient(patient)
        
        print(f"\n[Risk Assessment]")
        print(f"  Level: {result.risk_assessment.get('risk_level', 'Unknown')}")
        print(f"  Score: {result.risk_assessment.get('risk_score', 0):.3f}")
        print(f"  Factors: {result.risk_assessment.get('risk_factors', [])}")
        
        print(f"\n[Treatment Recommendation]")
        regimen = result.treatment_recommendation.get('treatment_recommendation', {})
        print(f"  Regimen: {regimen.get('regimen', 'N/A')}")
        print(f"  Status: {result.treatment_recommendation.get('approval_status', 'Unknown')}")
        
        print(f"\n[Alerts]")
        alert_summary = result.alert_summary.get('summary', {})
        print(f"  Total Alerts: {alert_summary.get('total_alerts', 0)}")
        print(f"  Critical: {alert_summary.get('critical_count', 0)}")
        print(f"  High: {alert_summary.get('high_count', 0)}")
        
        print(f"\n[Safety]")
        safety = result.safety_result
        print(f"  Blocked: {safety.get('blocked', False)}")
        print(f"  Can Proceed: {safety.get('can_proceed', False)}")
        
        print(f"\n[OVERALL STATUS: {result.overall_status}]")
    
    print(f"\n{'='*80}")
    print("ANALYSIS STATISTICS")
    print(f"{'='*80}")
    stats = orchestrator.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return orchestrator


if __name__ == "__main__":
    demo_analysis()
