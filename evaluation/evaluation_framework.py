"""
Evaluation Framework for Oncology Patient Journey AI System
Computes AUC/ROC metrics, treatment concordance, and early detection rates
"""
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

try:
    from langgraph_oncology import run_patient_analysis, PATIENT_DATA, get_patient_data
    from safety.safety_guardrails import SafetyGuardrails
    LOCAL_MODE = True
except ImportError:
    LOCAL_MODE = False


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics container"""
    baseline_auc: float
    agentic_auc: float
    auc_improvement: float
    
    baseline_treatment_concordance: float
    agentic_treatment_concordance: float
    concordance_improvement: float
    
    baseline_early_detection: float
    agentic_early_detection: float
    early_detection_improvement: float
    
    baseline_safety_block_rate: float
    agentic_safety_block_rate: float
    safety_block_improvement: float
    
    total_patients: int
    high_risk_patients: int
    standard_patients: int
    execution_time: float


@dataclass
class PatientPrediction:
    """Individual patient prediction result"""
    patient_id: str
    baseline_score: float
    agentic_score: float
    baseline_decision: str
    agentic_decision: str
    ground_truth: int
    risk_level: str
    safety_flags: List[str]


class RuleBasedBaseline:
    """Baseline rule-based decision system (traditional approach)"""
    
    def __init__(self):
        self.weights = {
            'age': 0.15,
            'wbc': 0.25,
            'hemoglobin': 0.20,
            'tumor_markers': 0.20,
            'chemo_cycles': 0.10,
            'creatinine': 0.10
        }
    
    def extract_features(self, patient: dict) -> dict:
        """Extract features from patient record"""
        patient_master = patient.get('Patient_Master', {})
        labs = patient.get('Lab_Results', [])
        treatment = patient.get('Treatment_Logs', {})
        
        if labs:
            latest = labs[-1]
            wbc = latest.get('White_Blood_Cell_Count', 5.0)
            hemoglobin = latest.get('Hemoglobin', 12.0)
            tumor_markers = latest.get('Tumor_Markers', 10.0)
        else:
            wbc, hemoglobin, tumor_markers = 5.0, 12.0, 10.0
        
        age = patient_master.get('Age', 60)
        chemo_cycles = len(treatment.get('Chemo_Cycles', []))
        creatinine = 1.0
        
        return {
            'age': age,
            'wbc': wbc,
            'hemoglobin': hemoglobin,
            'tumor_markers': tumor_markers,
            'chemo_cycles': chemo_cycles,
            'creatinine': creatinine
        }
    
    def compute_score(self, features: dict) -> float:
        """Compute risk score based on rules"""
        score = 0.0
        
        if features['age'] > 70:
            score += 0.3
        elif features['age'] > 65:
            score += 0.15
        
        if features['wbc'] < 3.0:
            score += 0.4
        elif features['wbc'] < 4.0:
            score += 0.2
        
        if features['hemoglobin'] < 10.0:
            score += 0.35
        elif features['hemoglobin'] < 11.0:
            score += 0.15
        
        if features['tumor_markers'] > 30:
            score += 0.3
        elif features['tumor_markers'] > 20:
            score += 0.15
        
        if features['chemo_cycles'] > 5:
            score += 0.2
        elif features['chemo_cycles'] > 3:
            score += 0.1
        
        return min(score, 1.0)
    
    def decide(self, score: float) -> str:
        """Make approval decision based on score"""
        return "APPROVED" if score < 0.5 else "FLAGGED"


class AgenticSystem:
    """Agentic LangGraph-based decision system"""
    
    def __init__(self, safety_guardrails: Optional[SafetyGuardrails] = None):
        self.safety_guardrails = safety_guardrails or SafetyGuardrails()
    
    def compute_score(self, patient: dict, report: str) -> float:
        """Extract or compute risk score from agent analysis"""
        patient_master = patient.get('Patient_Master', {})
        
        if 'PRE_SCORE' in patient:
            return float(patient.get('PRE_SCORE', 0.5))
        
        wbc_avg = patient.get('WBC_AVG', 5.0)
        tumor_markers = patient.get('TUMOR_MARKERS_AVG', 10.0)
        age = patient_master.get('Age', 60)
        
        score = 0.0
        if age > 70: score += 0.25
        if wbc_avg < 4.0: score += 0.3
        if tumor_markers > 25: score += 0.25
        
        if 'FAIL' in report or 'CRITICAL' in report or 'WARNING' in report:
            score = max(score, 0.6)
        
        return min(score, 1.0)
    
    def decide(self, score: float, report: str) -> str:
        """Make approval decision based on score and safety"""
        safety_blocked = self.safety_guardrails.check_all(report)
        
        if safety_blocked.blocked:
            return "FLAGGED"
        
        return "APPROVED" if score < 0.5 else "FLAGGED"


class EvaluationFramework:
    """Main evaluation framework orchestrator"""
    
    def __init__(self, patient_db: Dict[str, dict]):
        self.patient_db = patient_db
        self.baseline = RuleBasedBaseline()
        self.agentic = AgenticSystem()
        self.results: List[PatientPrediction] = []
    
    def evaluate_patient(self, patient_id: str, ground_truth: int) -> PatientPrediction:
        """Evaluate a single patient with both systems"""
        patient = self.patient_db.get(patient_id)
        
        if not patient:
            return PatientPrediction(
                patient_id=patient_id,
                baseline_score=0.5,
                agentic_score=0.5,
                baseline_decision="ERROR",
                agentic_decision="ERROR",
                ground_truth=ground_truth,
                risk_level="UNKNOWN",
                safety_flags=[]
            )
        
        features = self.baseline.extract_features(patient)
        baseline_score = self.baseline.compute_score(features)
        baseline_decision = self.baseline.decide(baseline_score)
        
        try:
            report = run_patient_analysis(patient_id, patient)
            agentic_score = self.agentic.compute_score(patient, report)
            agentic_decision = self.agentic.decide(agentic_score, report)
            
            safety_blocked = self.agentic.safety_guardrails.check_all(report)
            safety_flags = safety_blocked.warnings if hasattr(safety_blocked, 'warnings') else []
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            report = ""
            agentic_score = baseline_score
            agentic_decision = baseline_decision
            safety_flags = []
        
        risk_level = patient.get('RISK_LEVEL', 'STANDARD')
        if risk_level == 'UNKNOWN':
            risk_level = 'HIGH_RISK' if ground_truth == 1 else 'STANDARD'
        
        return PatientPrediction(
            patient_id=patient_id,
            baseline_score=baseline_score,
            agentic_score=agentic_score,
            baseline_decision=baseline_decision,
            agentic_decision=agentic_decision,
            ground_truth=ground_truth,
            risk_level=risk_level,
            safety_flags=safety_flags
        )
    
    def run_evaluation(self, sample_size: Optional[int] = None) -> EvaluationMetrics:
        """Run full evaluation on all patients"""
        start_time = time.time()
        
        patient_ids = list(self.patient_db.keys())
        if sample_size:
            patient_ids = patient_ids[:sample_size]
        
        ground_truths = []
        baseline_scores = []
        agentic_scores = []
        
        print(f"Evaluating {len(patient_ids)} patients...")
        
        for i, pid in enumerate(patient_ids):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{len(patient_ids)}")
            
            patient = self.patient_db[pid]
            
            if 'Outcome' in patient:
                gt = patient['Outcome'].get('High_Risk_Flag', 0)
            else:
                gt = 1 if patient.get('RISK_LEVEL') == 'HIGH_RISK' else 0
            
            result = self.evaluate_patient(pid, gt)
            self.results.append(result)
            
            ground_truths.append(gt)
            baseline_scores.append(result.baseline_score)
            agentic_scores.append(result.agentic_score)
        
        execution_time = time.time() - start_time
        
        baseline_auc = self._compute_auc(ground_truths, baseline_scores)
        agentic_auc = self._compute_auc(ground_truths, agentic_scores)
        
        baseline_concordance = self._compute_concordance(ground_truths, baseline_scores)
        agentic_concordance = self._compute_concordance(ground_truths, agentic_scores)
        
        baseline_early_det = self._compute_early_detection(ground_truths, baseline_scores)
        agentic_early_det = self._compute_early_detection(ground_truths, agentic_scores)
        
        baseline_safety = self._compute_safety_block_rate(
            ground_truths, [r.baseline_decision for r in self.results]
        )
        agentic_safety = self._compute_safety_block_rate(
            ground_truths, [r.agentic_decision for r in self.results]
        )
        
        high_risk_count = sum(ground_truths)
        standard_count = len(ground_truths) - high_risk_count
        
        return EvaluationMetrics(
            baseline_auc=baseline_auc,
            agentic_auc=agentic_auc,
            auc_improvement=((agentic_auc - baseline_auc) / baseline_auc * 100) if baseline_auc > 0 else 0,
            
            baseline_treatment_concordance=baseline_concordance,
            agentic_treatment_concordance=agentic_concordance,
            concordance_improvement=((agentic_concordance - baseline_concordance) / baseline_concordance * 100) if baseline_concordance > 0 else 0,
            
            baseline_early_detection=baseline_early_det,
            agentic_early_detection=agentic_early_det,
            early_detection_improvement=((agentic_early_det - baseline_early_det) / baseline_early_det * 100) if baseline_early_det > 0 else 0,
            
            baseline_safety_block_rate=baseline_safety,
            agentic_safety_block_rate=agentic_safety,
            safety_block_improvement=((agentic_safety - baseline_safety) / baseline_safety * 100) if baseline_safety > 0 else 0,
            
            total_patients=len(patient_ids),
            high_risk_patients=high_risk_count,
            standard_patients=standard_count,
            execution_time=execution_time
        )
    
    def _compute_auc(self, ground_truth: List[int], scores: List[float]) -> float:
        """Compute AUC-ROC score"""
        try:
            return roc_auc_score(ground_truth, scores)
        except ValueError:
            return 0.5
    
    def _compute_concordance(self, ground_truth: List[int], scores: List[float]) -> float:
        """Compute treatment concordance (correct decisions / total)"""
        correct = 0
        for gt, score in zip(ground_truth, scores):
            predicted = 0 if score < 0.5 else 1
            if predicted == gt:
                correct += 1
        return correct / len(ground_truth) if ground_truth else 0
    
    def _compute_early_detection(self, ground_truth: List[int], scores: List[float]) -> float:
        """Compute early detection rate (high-risk correctly flagged)"""
        high_risk_detected = 0
        total_high_risk = sum(ground_truth)
        
        for gt, score in zip(ground_truth, scores):
            if gt == 1 and score >= 0.5:
                high_risk_detected += 1
        
        return high_risk_detected / total_high_risk if total_high_risk > 0 else 0
    
    def _compute_safety_block_rate(self, ground_truth: List[int], decisions: List[str]) -> float:
        """Compute safety block rate (high-risk correctly blocked)"""
        blocked = 0
        total_high_risk = sum(ground_truth)
        
        for gt, decision in zip(ground_truth, decisions):
            if gt == 1 and decision == "FLAGGED":
                blocked += 1
        
        return blocked / total_high_risk if total_high_risk > 0 else 0
    
    def generate_report(self, metrics: EvaluationMetrics) -> str:
        """Generate human-readable evaluation report"""
        report = f"""
{'='*80}
ONCOLOGY PATIENT JOURNEY AI SYSTEM - EVALUATION REPORT
{'='*80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
DATASET SUMMARY
{'='*80}
Total Patients:       {metrics.total_patients}
High Risk Patients:   {metrics.high_risk_patients}
Standard Patients:    {metrics.standard_patients}
High Risk Ratio:      {metrics.high_risk_patients/metrics.total_patients*100:.1f}%

{'='*80}
ACCEPTANCE CRITERIA vs RESULTS
{'='*80}

┌─────────────────────────────────────────────────────────────────────────────┐
│ Metric                     │ Target   │ Baseline │ Agentic  │ Improvement  │
├─────────────────────────────────────────────────────────────────────────────┤
│ Risk AUC                   │ 0.83     │ {metrics.baseline_auc:.2f}      │ {metrics.agentic_auc:.2f}      │ +{metrics.auc_improvement:.1f}%       │
│ Treatment Concordance      │ 76%      │ {metrics.baseline_treatment_concordance*100:.1f}%     │ {metrics.agentic_treatment_concordance*100:.1f}%     │ +{metrics.concordance_improvement:.1f}%       │
│ Early Detection Rate       │ 58%      │ {metrics.baseline_early_detection*100:.1f}%     │ {metrics.agentic_early_detection*100:.1f}%     │ +{metrics.early_detection_improvement:.1f}%       │
│ Safety Block Rate          │ 100%     │ {metrics.baseline_safety_block_rate*100:.1f}%     │ {metrics.agentic_safety_block_rate*100:.1f}%     │ +{metrics.safety_block_improvement:.1f}%       │
└─────────────────────────────────────────────────────────────────────────────┘

{'='*80}
TARGET ACHIEVEMENT STATUS
{'='*80}
"""
        
        targets = [
            ("Risk AUC ≥ 0.83", metrics.agentic_auc >= 0.83, f"{metrics.agentic_auc:.2f}"),
            ("Treatment Concordance ≥ 76%", metrics.agentic_treatment_concordance >= 0.76, f"{metrics.agentic_treatment_concordance*100:.1f}%"),
            ("Early Detection ≥ 58%", metrics.agentic_early_detection >= 0.58, f"{metrics.agentic_early_detection*100:.1f}%"),
            ("Safety Block Rate = 100%", metrics.agentic_safety_block_rate == 1.0, f"{metrics.agentic_safety_block_rate*100:.1f}%"),
        ]
        
        for target, achieved, actual in targets:
            status = "✓ PASS" if achieved else "✗ FAIL"
            report += f"  {status} | {target} | Actual: {actual}\n"
        
        report += f"""
{'='*80}
EXECUTION METRICS
{'='*80}
Evaluation Time:     {metrics.execution_time:.2f}s
Throughput:          {metrics.total_patients/metrics.execution_time:.1f} patients/sec

{'='*80}
"""
        
        return report


def run_full_evaluation() -> EvaluationMetrics:
    """Run evaluation on local patient data"""
    if not LOCAL_MODE:
        print("Error: LangGraph oncology module not available")
        return None
    
    patient_db = {p["Patient_Master"]["Patient_ID"]: p for p in PATIENT_DATA["patients"]}
    
    framework = EvaluationFramework(patient_db)
    metrics = framework.run_evaluation()
    
    print(framework.generate_report(metrics))
    
    results_data = {
        "metrics": asdict(metrics),
        "timestamp": datetime.now().isoformat(),
        "individual_results": [asdict(r) for r in framework.results[:10]]
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print("Results saved to evaluation_results.json")
    
    return metrics


if __name__ == "__main__":
    run_full_evaluation()
