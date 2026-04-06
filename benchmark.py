import json
import time
from typing import Dict, List, Tuple
from langgraph_oncology import run_patient_analysis, PATIENT_DATA

patient_db = {p["Patient_Master"]["Patient_ID"]: p for p in PATIENT_DATA["patients"]}

def rule_based_check(patient_id: str) -> Tuple[str, Dict]:
    """Baseline: Simple rule-based check on latest lab values."""
    patient = patient_db.get(patient_id)
    if not patient:
        return "NOT_FOUND", {}
    
    lab_results = patient.get("Lab_Results", [])
    if not lab_results:
        return "NO_DATA", {}
    
    latest = lab_results[-1]
    wbc = latest.get("White_Blood_Cell_Count", 0)
    hb = latest.get("Hemoglobin", 0)
    
    wbc_safe = wbc >= 3.0
    hb_safe = hb >= 10.0
    
    if wbc_safe and hb_safe:
        decision = "APPROVED"
    else:
        decision = "FLAGGED"
    
    return decision, {"WBC": wbc, "Hb": hb, "WBC_Safe": wbc_safe, "Hb_Safe": hb_safe}

def classify_agentic_decision(report: str) -> str:
    """Parse agentic decision from the report."""
    if "SAFETY AUDIT: FAILED" in report or "FAIL:" in report or "CRITICAL" in report:
        return "FLAGGED"
    return "APPROVED"

def calculate_metrics(predictions: List[str], ground_truth: List[int]) -> Dict:
    """Calculate precision, recall, and safety miss rate."""
    tp = sum(1 for p, gt in zip(predictions, ground_truth) if p == "FLAGGED" and gt == 1)
    fp = sum(1 for p, gt in zip(predictions, ground_truth) if p == "FLAGGED" and gt == 0)
    tn = sum(1 for p, gt in zip(predictions, ground_truth) if p == "APPROVED" and gt == 0)
    fn = sum(1 for p, gt in zip(predictions, ground_truth) if p == "APPROVED" and gt == 1)
    
    total_high_risk = sum(ground_truth)
    total_approved = sum(1 for p in predictions if p == "APPROVED")
    high_risk_approved = fn # False negative = High risk but approved
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    safety_miss_rate = high_risk_approved / total_high_risk if total_high_risk > 0 else 0
    
    return {
        "Precision": precision,
        "Recall": recall,
        "Total_Patients": len(predictions),
        "Total_High_Risk": total_high_risk,
        "Total_Approved": total_approved,
        "High_Risk_Approved": high_risk_approved,
        "Safety_Miss_Rate": safety_miss_rate,
        "True_Positives": tp,
        "False_Positives": fp,
        "True_Negatives": tn,
        "False_Negatives": fn
    }

def run_benchmark():
    print("=" * 70)
    print("BENCHMARK: Rule-Based vs Agentic (LangGraph) Oncology Decision System")
    print("=" * 70)
    
    patient_ids = list(patient_db.keys())
    
    baseline_predictions = []
    agentic_predictions = []
    ground_truth = []
    
    print(f"\nRunning baseline model on {len(patient_ids)} patients...")
    start = time.time()
    for pid in patient_ids:
        decision, _ = rule_based_check(pid)
        baseline_predictions.append(decision)
        ground_truth.append(patient_db[pid]["Outcome"]["High_Risk_Flag"])
    baseline_time = time.time() - start
    print(f"Baseline completed in {baseline_time:.2f}s")
    
    print(f"\nRunning agentic model on {len(patient_ids)} patients...")
    start = time.time()
    for i, pid in enumerate(patient_ids):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(patient_ids)}")
        try:
            report = run_patient_analysis(pid)
            decision = classify_agentic_decision(report)
        except Exception as e:
            decision = "ERROR"
        agentic_predictions.append(decision)
    agentic_time = time.time() - start
    print(f"Agentic completed in {agentic_time:.2f}s")
    
    baseline_metrics = calculate_metrics(baseline_predictions, ground_truth)
    agentic_metrics = calculate_metrics(agentic_predictions, ground_truth)
    
    baseline_miss_rate = baseline_metrics["Safety_Miss_Rate"] * 100
    agentic_miss_rate = agentic_metrics["Safety_Miss_Rate"] * 100
    reduction = baseline_miss_rate - agentic_miss_rate
    reduction_pct = (reduction / baseline_miss_rate * 100) if baseline_miss_rate > 0 else 0
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Metric':<30} {'Baseline':<15} {'Agentic':<15}")
    print("-" * 60)
    print(f"{'Total Patients':<30} {baseline_metrics['Total_Patients']:<15} {agentic_metrics['Total_Patients']:<15}")
    print(f"{'High Risk (Ground Truth)':<30} {baseline_metrics['Total_High_Risk']:<15} {agentic_metrics['Total_High_Risk']:<15}")
    print(f"{'Total Approved':<30} {baseline_metrics['Total_Approved']:<15} {agentic_metrics['Total_Approved']:<15}")
    print(f"{'High Risk Approved':<30} {baseline_metrics['High_Risk_Approved']:<15} {agentic_metrics['High_Risk_Approved']:<15}")
    
    print(f"\n{'Precision':<30} {baseline_metrics['Precision']*100:.1f}%{'':<9} {agentic_metrics['Precision']*100:.1f}%")
    print(f"{'Recall':<30} {baseline_metrics['Recall']*100:.1f}%{'':<9} {agentic_metrics['Recall']*100:.1f}%")
    
    print("\n" + "-" * 60)
    print(f"{'SAFETY MISS RATE':<30} {baseline_miss_rate:.1f}%{'':<9} {agentic_miss_rate:.1f}%")
    print("-" * 60)
    print(f"{'REDUCTION IN HIGH RISK APPROVALS:':<30} {reduction_pct:.1f}%")
    print("=" * 70)
    
    if reduction_pct >= 20:
        print(f"\n[*] TARGET ACHIEVED: {reduction_pct:.1f}% improvement (target: ~20%)")
    else:
        print(f"\n[!] TARGET NOT MET: {reduction_pct:.1f}% improvement (target: ~20%)")
    
    results = {
        "baseline": baseline_metrics,
        "agentic": agentic_metrics,
        "reduction_percentage": reduction_pct
    }
    
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to benchmark_results.json")
    
    return results

if __name__ == "__main__":
    run_benchmark()
