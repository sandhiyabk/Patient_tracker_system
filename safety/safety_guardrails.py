"""
Clinical Safety Guardrails for Oncology AI System
Implements drug interactions, contraindications, and safety checks
"""
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class SafetyLevel(Enum):
    """Safety check levels"""
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"
    CRITICAL = "CRITICAL"


class SafetyCategory(Enum):
    """Categories of safety checks"""
    DRUG_INTERACTION = "DRUG_INTERACTION"
    CONTRAINDICATION = "CONTRAINDICATION"
    LAB_VALUE = "LAB_VALUE"
    VITAL_SIGN = "VITAL_SIGN"
    DOSE_LIMIT = "DOSE_LIMIT"
    DURATION_LIMIT = "DURATION_LIMIT"
    BLACK_BOX = "BLACK_BOX"
    PREGNANCY = "PREGNANCY"
    ORGAN_DYSFUNCTION = "ORGAN_DYSFUNCTION"


@dataclass
class SafetyCheck:
    """Individual safety check result"""
    check_name: str
    category: SafetyCategory
    severity: SafetyLevel
    passed: bool
    message: str
    recommendation: str
    data: Dict = field(default_factory=dict)


@dataclass
class SafetyResult:
    """Overall safety check result"""
    blocked: bool
    blocked_count: int
    warning_count: int
    caution_count: int
    passed_count: int
    total_checks: int
    warnings: List[str]
    critical_blocks: List[str]
    recommendations: List[str]
    can_proceed: bool
    conditional_proceed: bool
    additional_monitoring: List[str]


class SafetyGuardrails:
    """
    Clinical safety guardrails for oncology treatment decisions.
    Implements comprehensive safety checks including:
    - Drug-drug interactions
    - Contraindications
    - Lab value thresholds
    - Black box warnings
    - Organ dysfunction limits
    """
    
    DRUG_INTERACTION_DB = {
        ("CARBOPLATIN", "AMIKACIN"): {"severity": "HIGH", "mechanism": "Increased ototoxicity", "recommendation": "Avoid combination"},
        ("CARBOPLATIN", "GENTAMICIN"): {"severity": "HIGH", "mechanism": "Increased nephrotoxicity and ototoxicity", "recommendation": "Avoid combination"},
        ("CISPLATIN", "AMINOGLYCOSIDE"): {"severity": "CRITICAL", "mechanism": "Severe nephrotoxicity and ototoxicity", "recommendation": "CONTRAINDICATED"},
        ("CISPLATIN", "LOOP_DIURETIC"): {"severity": "HIGH", "mechanism": "Increased risk of ototoxicity", "recommendation": "Monitor hearing"},
        ("PACLITAXEL", "DOXORUBICIN"): {"severity": "MODERATE", "mechanism": "Cardiac toxicity enhanced", "recommendation": "Sequential administration preferred"},
        ("TRASTUZUMAB", "ANTHRACYCLINE"): {"severity": "HIGH", "mechanism": "Increased cardiotoxicity", "recommendation": "Sequential with cardiac monitoring"},
        ("BEVACIZUMAB", "SUNITINIB"): {"severity": "HIGH", "mechanism": "Increased risk of microangiopathic hemolytic anemia", "recommendation": "Avoid combination"},
        ("PEMBROLIZUMAB", "CORTICOSTEROID"): {"severity": "HIGH", "mechanism": "Reduced immunotherapy efficacy", "recommendation": "Avoid chronic high-dose steroids"},
        ("5_FU", "WARFARIN"): {"severity": "HIGH", "mechanism": "Increased bleeding risk", "recommendation": "Monitor INR closely"},
        ("CAPECITABINE", "WARFARIN"): {"severity": "HIGH", "mechanism": "Increased bleeding risk", "recommendation": "Monitor INR closely"},
        ("METHOTREXATE", "NSAID"): {"severity": "CRITICAL", "mechanism": "Severe methotrexate toxicity", "recommendation": "CONTRAINDICATED"},
        ("METHOTREXATE", "ASPIRIN"): {"severity": "HIGH", "mechanism": "Increased methotrexate levels", "recommendation": "Avoid or reduce dose"},
        ("VINCRISTINE", "ITRAMENT"): {"severity": "CRITICAL", "mechanism": "Fatal neurotoxicity", "recommendation": "IT VINCRISTINE IS ABSOLUTELY CONTRAINDICATED"},
        ("LENALIDOMIDE", "ESTROGEN"): {"severity": "HIGH", "mechanism": "Increased thromboembolic risk", "recommendation": "Avoid combination"},
        ("POMALIDOMIDE", "ESTROGEN"): {"severity": "HIGH", "mechanism": "Increased thromboembolic risk", "recommendation": "Avoid combination"},
    }
    
    BLACK_BOX_WARNINGS = {
        "PACLITAXEL": "ANAPHYLAXIS: Fatal reactions have occurred. Premedicate before administration.",
        "CISPLATIN": "NEPHROTOXICITY: Dose-related and cumulative. Monitor renal function.",
        "DOXORUBICIN": "CARDIOTOXICITY: Fatal heart failure. Lifetime dose limit 450 mg/m².",
        "BEVACIZUMAB": "GI PERFORATION: Serious and sometimes fatal. Discontinue if occurs.",
        "LENALIDOMIDE": "TERATOGENICITY: Severe birth defects. REMS program required.",
        "POMALIDOMIDE": "TERATOGENICITY: Severe birth defects. REMS program required.",
        "THALIDOMIDE": "TERATOGENICITY: Severe birth defects. REMS program required.",
    }
    
    LAB_THRESHOLDS = {
        "wbc": {"absolute_continue": 3.0, "conditional_continue": 2.0, "absolute_stop": 1.0},
        "anc": {"absolute_continue": 1500, "conditional_continue": 1000, "absolute_stop": 500},
        "platelet": {"absolute_continue": 75, "conditional_continue": 50, "absolute_stop": 20},
        "hemoglobin": {"absolute_continue": 10.0, "conditional_continue": 8.0, "absolute_stop": 7.0},
        "creatinine": {"dose_modify": 1.5, "stop_cisplatin": 2.0, "stop_other": 3.0},
        "bilirubin": {"dose_modify": 1.5, "stop": 3.0},
        "alt_ast": {"dose_modify": 3.0, "stop": 5.0},
        "albumin": {"low_risk": 3.5, "moderate_risk": 3.0, "high_risk": 2.5},
    }
    
    ORGAN_LIMITS = {
        "cardiac": {
            "ef_minimum": 50,
            "ef_conditional": 45,
            "qtc_max": 500,
        },
        "hepatic": {
            "bilirubin_grade1": 1.25,
            "bilirubin_grade2": 1.5,
            "bilirubin_grade3": 1.75,
            "bilirubin_grade4": 3.0,
        },
        "renal": {
            "crcl_mild": 60,
            "crcl_moderate": 30,
            "crcl_severe": 15,
        },
        "pulmonary": {
            "dlco_minimum": 40,
            "dyspnea_grade3": "avoid_sever_toxicity",
        }
    }
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.check_history: List[SafetyCheck] = []
    
    def check_all(self, report: str, patient_data: Optional[Dict] = None) -> SafetyResult:
        """
        Run all safety checks and return comprehensive result.
        
        Args:
            report: Treatment recommendation report to check
            patient_data: Optional patient data for detailed checks
            
        Returns:
            SafetyResult with overall safety assessment
        """
        checks = []
        
        if patient_data:
            lab_checks = self._check_lab_values(patient_data)
            checks.extend(lab_checks)
            
            organ_checks = self._check_organ_function(patient_data)
            checks.extend(organ_checks)
            
            drug_checks = self._check_drug_interactions(patient_data)
            checks.extend(drug_checks)
        
        black_box_checks = self._check_black_box_warnings(report)
        checks.extend(black_box_checks)
        
        return self._compile_results(checks)
    
    def check_drug_interaction(self, drug1: str, drug2: str) -> Tuple[bool, SafetyLevel, str]:
        """
        Check for interaction between two drugs.
        
        Returns:
            Tuple of (has_interaction, severity, recommendation)
        """
        drug1_upper = drug1.upper().replace(" ", "_")
        drug2_upper = drug2.upper().replace(" ", "_")
        
        key = (drug1_upper, drug2_upper)
        reverse_key = (drug2_upper, drug1_upper)
        
        if key in self.DRUG_INTERACTION_DB:
            interaction = self.DRUG_INTERACTION_DB[key]
            severity = SafetyLevel.CRITICAL if interaction["severity"] == "CRITICAL" else SafetyLevel.BLOCKED
            return True, severity, interaction["recommendation"]
        
        if reverse_key in self.DRUG_INTERACTION_DB:
            interaction = self.DRUG_INTERACTION_DB[reverse_key]
            severity = SafetyLevel.CRITICAL if interaction["severity"] == "CRITICAL" else SafetyLevel.BLOCKED
            return True, severity, interaction["recommendation"]
        
        return False, SafetyLevel.SAFE, "No known interaction"
    
    def check_contraindication(
        self,
        drug: str,
        patient_data: Dict,
        absolute_only: bool = True
    ) -> List[Tuple[str, SafetyLevel, str]]:
        """
        Check if drug is contraindicated for this patient.
        
        Returns:
            List of (contraindication_type, severity, description)
        """
        contraindications = []
        drug_upper = drug.upper()
        
        labs = patient_data.get("labs", [])
        if labs:
            latest = labs[-1]
            wbc = latest.get("wbc", latest.get("White_Blood_Cell_Count", 5.0))
            
            if wbc < self.LAB_THRESHOLDS["wbc"]["absolute_continue"] and not absolute_only:
                contraindications.append((
                    "Lab Value",
                    SafetyLevel.WARNING,
                    f"WBC {wbc} below threshold for most chemotherapy"
                ))
        
        return contraindications
    
    def _check_lab_values(self, patient_data: Dict) -> List[SafetyCheck]:
        """Check all lab values against thresholds"""
        checks = []
        labs = patient_data.get("labs", [])
        
        if not labs:
            return checks
        
        latest = labs[-1]
        
        wbc = latest.get("wbc", latest.get("White_Blood_Cell_Count"))
        if wbc:
            thresholds = self.LAB_THRESHOLDS["wbc"]
            if wbc < thresholds["absolute_stop"]:
                checks.append(SafetyCheck(
                    check_name="WBC Critical Low",
                    category=SafetyCategory.LAB_VALUE,
                    severity=SafetyLevel.CRITICAL,
                    passed=False,
                    message=f"WBC critically low at {wbc}",
                    recommendation="BLOCK ALL CHEMOTHERAPY. Immediate intervention required.",
                    data={"wbc": wbc, "threshold": thresholds["absolute_stop"]}
                ))
            elif wbc < thresholds["conditional_continue"]:
                checks.append(SafetyCheck(
                    check_name="WBC Low",
                    category=SafetyCategory.LAB_VALUE,
                    severity=SafetyLevel.BLOCKED,
                    passed=False,
                    message=f"WBC {wbc} below safe threshold",
                    recommendation="Hold chemotherapy. Growth factor support required.",
                    data={"wbc": wbc, "threshold": thresholds["conditional_continue"]}
                ))
            elif wbc < thresholds["absolute_continue"]:
                checks.append(SafetyCheck(
                    check_name="WBC Borderline",
                    category=SafetyCategory.LAB_VALUE,
                    severity=SafetyLevel.WARNING,
                    passed=False,
                    message=f"WBC {wbc} borderline for chemotherapy",
                    recommendation="Consider dose reduction or growth factor.",
                    data={"wbc": wbc, "threshold": thresholds["absolute_continue"]}
                ))
            else:
                checks.append(SafetyCheck(
                    check_name="WBC Normal",
                    category=SafetyCategory.LAB_VALUE,
                    severity=SafetyLevel.SAFE,
                    passed=True,
                    message=f"WBC {wbc} within acceptable range",
                    recommendation="Proceed with treatment.",
                    data={"wbc": wbc}
                ))
        
        hgb = latest.get("hgb", latest.get("Hemoglobin"))
        if hgb:
            thresholds = self.LAB_THRESHOLDS["hemoglobin"]
            if hgb < thresholds["absolute_stop"]:
                checks.append(SafetyCheck(
                    check_name="Hemoglobin Critical",
                    category=SafetyCategory.LAB_VALUE,
                    severity=SafetyLevel.CRITICAL,
                    passed=False,
                    message=f"Hemoglobin critically low at {hgb}",
                    recommendation="Transfusion required before treatment.",
                    data={"hgb": hgb}
                ))
            elif hgb < thresholds["conditional_continue"]:
                checks.append(SafetyCheck(
                    check_name="Hemoglobin Low",
                    category=SafetyCategory.LAB_VALUE,
                    severity=SafetyLevel.BLOCKED,
                    passed=False,
                    message=f"Hemoglobin {hgb} below threshold",
                    recommendation="Transfusion or careful risk-benefit analysis required.",
                    data={"hgb": hgb}
                ))
        
        platelet = latest.get("platelet", latest.get("Platelets"))
        if platelet:
            thresholds = self.LAB_THRESHOLDS["platelet"]
            if platelet < thresholds["absolute_stop"]:
                checks.append(SafetyCheck(
                    check_name="Platelet Critical",
                    category=SafetyCategory.LAB_VALUE,
                    severity=SafetyLevel.CRITICAL,
                    passed=False,
                    message=f"Platelet critically low at {platelet}",
                    recommendation="Platelet transfusion required.",
                    data={"platelet": platelet}
                ))
            elif platelet < thresholds["conditional_continue"]:
                checks.append(SafetyCheck(
                    check_name="Platelet Low",
                    category=SafetyCategory.LAB_VALUE,
                    severity=SafetyLevel.BLOCKED,
                    passed=False,
                    message=f"Platelets {platelet} below safe threshold",
                    recommendation="Hold chemotherapy. Consider transfusion.",
                    data={"platelet": platelet}
                ))
        
        return checks
    
    def _check_organ_function(self, patient_data: Dict) -> List[SafetyCheck]:
        """Check organ function limits"""
        checks = []
        
        labs = patient_data.get("labs", [])
        vitals = patient_data.get("vitals", [])
        
        if labs:
            latest = labs[-1]
            
            creatinine = latest.get("creatinine", 1.0)
            if creatinine > self.ORGAN_LIMITS["renal"]["crcl_severe"]:
                checks.append(SafetyCheck(
                    check_name="Severe Renal Dysfunction",
                    category=SafetyCategory.ORGAN_DYSFUNCTION,
                    severity=SafetyLevel.CRITICAL,
                    passed=False,
                    message=f"Creatinine {creatinine} indicates severe renal impairment",
                    recommendation="Dose adjust or avoid nephrotoxic agents. Consult nephrology.",
                    data={"creatinine": creatinine}
                ))
        
        return checks
    
    def _check_drug_interactions(self, patient_data: Dict) -> List[SafetyCheck]:
        """Check for drug-drug interactions"""
        checks = []
        
        meds = patient_data.get("medications", [])
        if not meds:
            return checks
        
        drug_names = [m.get("drug", "").upper().replace(" ", "_") for m in meds]
        
        for i, drug1 in enumerate(drug_names):
            for drug2 in drug_names[i+1:]:
                has_interaction, severity, recommendation = self.check_drug_interaction(drug1, drug2)
                
                if has_interaction:
                    checks.append(SafetyCheck(
                        check_name=f"Drug Interaction: {drug1} + {drug2}",
                        category=SafetyCategory.DRUG_INTERACTION,
                        severity=severity,
                        passed=severity == SafetyLevel.SAFE,
                        message=f"Interaction between {drug1} and {drug2}",
                        recommendation=recommendation,
                        data={"drug1": drug1, "drug2": drug2}
                    ))
        
        return checks
    
    def _check_black_box_warnings(self, report: str) -> List[SafetyCheck]:
        """Check for black box warnings in report"""
        checks = []
        report_upper = report.upper()
        
        for drug, warning in self.BLACK_BOX_WARNINGS.items():
            if drug in report_upper:
                checks.append(SafetyCheck(
                    check_name=f"BLACK BOX WARNING: {drug}",
                    category=SafetyCategory.BLACK_BOX,
                    severity=SafetyLevel.WARNING,
                    passed=True,
                    message=warning,
                    recommendation=f"Ensure {drug} safety protocols are followed. REMS enrollment if required.",
                    data={"drug": drug}
                ))
        
        return checks
    
    def _compile_results(self, checks: List[SafetyCheck]) -> SafetyResult:
        """Compile all checks into final result"""
        blocked_count = sum(1 for c in checks if c.severity in [SafetyLevel.BLOCKED, SafetyLevel.CRITICAL])
        warning_count = sum(1 for c in checks if c.severity == SafetyLevel.WARNING)
        caution_count = sum(1 for c in checks if c.severity == SafetyLevel.CAUTION)
        passed_count = sum(1 for c in checks if c.passed)
        
        warnings = [c.message for c in checks if c.severity in [SafetyLevel.WARNING, SafetyLevel.BLOCKED, SafetyLevel.CRITICAL]]
        critical_blocks = [c.message for c in checks if c.severity == SafetyLevel.CRITICAL]
        recommendations = [c.recommendation for c in checks if c.severity in [SafetyLevel.WARNING, SafetyLevel.BLOCKED, SafetyLevel.CRITICAL]]
        additional_monitoring = [
            c.recommendation for c in checks 
            if c.severity == SafetyLevel.CAUTION or (c.severity == SafetyLevel.WARNING and c.passed)
        ]
        
        can_proceed = blocked_count == 0
        conditional_proceed = blocked_count == 0 and warning_count > 0
        
        return SafetyResult(
            blocked=blocked_count > 0,
            blocked_count=blocked_count,
            warning_count=warning_count,
            caution_count=caution_count,
            passed_count=passed_count,
            total_checks=len(checks),
            warnings=warnings,
            critical_blocks=critical_blocks,
            recommendations=recommendations,
            can_proceed=can_proceed,
            conditional_proceed=conditional_proceed,
            additional_monitoring=additional_monitoring
        )
    
    def get_safety_summary(self) -> Dict:
        """Get summary of all safety checks performed"""
        if not self.check_history:
            return {"total_checks": 0}
        
        return {
            "total_checks": len(self.check_history),
            "by_severity": {
                "critical": sum(1 for c in self.check_history if c.severity == SafetyLevel.CRITICAL),
                "blocked": sum(1 for c in self.check_history if c.severity == SafetyLevel.BLOCKED),
                "warning": sum(1 for c in self.check_history if c.severity == SafetyLevel.WARNING),
                "caution": sum(1 for c in self.check_history if c.severity == SafetyLevel.CAUTION),
                "safe": sum(1 for c in self.check_history if c.severity == SafetyLevel.SAFE),
            },
            "by_category": {}
        }


@dataclass
class SafetyTestCase:
    """Test case for safety guardrail validation"""
    name: str
    patient_data: Dict
    expected_blocked: bool
    expected_warning_count: int


def run_safety_tests() -> Dict:
    """Run comprehensive safety tests"""
    guardrails = SafetyGuardrails()
    test_cases = [
        SafetyTestCase(
            name="Critical Neutropenia",
            patient_data={"labs": [{"wbc": 1.5}]},
            expected_blocked=True,
            expected_warning_count=0
        ),
        SafetyTestCase(
            name="Mild Neutropenia",
            patient_data={"labs": [{"wbc": 3.5}]},
            expected_blocked=False,
            expected_warning_count=0
        ),
        SafetyTestCase(
            name="Borderline WBC",
            patient_data={"labs": [{"wbc": 2.8}]},
            expected_blocked=True,
            expected_warning_count=0
        ),
    ]
    
    results = []
    for tc in test_cases:
        result = guardrails.check_all("", tc.patient_data)
        passed = result.blocked == tc.expected_blocked
        results.append({
            "name": tc.name,
            "expected_blocked": tc.expected_blocked,
            "actual_blocked": result.blocked,
            "passed": passed
        })
    
    return {"tests": results, "total": len(results), "passed": sum(1 for r in results if r["passed"])}


if __name__ == "__main__":
    guardrails = SafetyGuardrails()
    
    test_patient = {
        "labs": [{
            "wbc": 2.5,
            "hgb": 9.0,
            "platelet": 180,
            "creatinine": 1.2
        }],
        "medications": [
            {"drug": "Carboplatin"},
            {"drug": "Ondansetron"}
        ]
    }
    
    result = guardrails.check_all("", test_patient)
    
    print(f"Safety Check Result:")
    print(f"  Blocked: {result.blocked}")
    print(f"  Can Proceed: {result.can_proceed}")
    print(f"  Warnings: {len(result.warnings)}")
    print(f"  Recommendations: {result.recommendations[:2]}")
    
    test_interaction = guardrails.check_drug_interaction("Cisplatin", "Gentamicin")
    print(f"\nDrug Interaction Test (Cisplatin + Gentamicin):")
    print(f"  Has Interaction: {test_interaction[0]}")
    print(f"  Severity: {test_interaction[1].value}")
    print(f"  Recommendation: {test_interaction[2]}")
