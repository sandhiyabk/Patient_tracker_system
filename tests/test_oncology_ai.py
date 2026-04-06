"""
Comprehensive Test Suite for Oncology AI System
Tests agents, safety guardrails, and evaluation framework
"""
import unittest
import json
import sys
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, '.')

from agents.risk_agent import RiskAgent, RiskLevel
from agents.treatment_agent import TreatmentAgent
from agents.alert_agent import AlertAgent, AlertSeverity, AlertCategory
from safety.safety_guardrails import SafetyGuardrails, SafetyLevel, SafetyCategory
from data.mimic_data_generator import MIMICDataGenerator


class TestRiskAgent(unittest.TestCase):
    """Test suite for Risk Agent"""
    
    def setUp(self):
        self.agent = RiskAgent()
    
    def test_assess_risk_standard(self):
        """Test risk assessment for standard patient"""
        patient = {
            "patient_id": "TEST-001",
            "demography": {"age": 55, "gender": "M"},
            "labs": [{
                "wbc": 6.0, "hgb": 14.0, "platelet": 250,
                "creatinine": 1.0, "albumin": 4.0
            }]
        }
        
        result = self.agent.assess_risk(patient)
        
        self.assertEqual(result["risk_level"], "STANDARD")
        self.assertLess(result["risk_score"], 0.3)
        self.assertFalse(result["requires_review"])
    
    def test_assess_risk_high_age(self):
        """Test risk assessment for elderly patient"""
        patient = {
            "patient_id": "TEST-002",
            "demography": {"age": 78, "gender": "F"},
            "labs": [{
                "wbc": 5.5, "hgb": 13.5, "platelet": 220,
                "creatinine": 1.1, "albumin": 3.8
            }]
        }
        
        result = self.agent.assess_risk(patient)
        
        self.assertTrue(any("Advanced Age" in f for f in result["risk_factors"]))
    
    def test_assess_risk_neutropenia(self):
        """Test risk assessment for neutropenic patient"""
        patient = {
            "patient_id": "TEST-003",
            "demography": {"age": 65, "gender": "M"},
            "labs": [{
                "wbc": 2.5, "hgb": 11.0, "platelet": 150,
                "creatinine": 1.0
            }]
        }
        
        result = self.agent.assess_risk(patient)
        
        self.assertTrue(any("Neutropenia" in f for f in result["risk_factors"]))
        self.assertGreaterEqual(result["risk_score"], 0.25)
    
    def test_assess_risk_critical(self):
        """Test risk assessment for critically ill patient"""
        patient = {
            "patient_id": "TEST-004",
            "demography": {"age": 75, "gender": "F"},
            "labs": [{
                "wbc": 1.5, "hgb": 8.5, "platelet": 40,
                "creatinine": 1.8, "albumin": 2.8,
                "tumor_marker_ca125": 120
            }]
        }
        
        result = self.agent.assess_risk(patient)
        
        self.assertIn(result["risk_level"], ["HIGH_RISK", "CRITICAL"])
        self.assertGreaterEqual(result["risk_score"], 0.5)
        self.assertTrue(result["requires_review"])


class TestTreatmentAgent(unittest.TestCase):
    """Test suite for Treatment Agent"""
    
    def setUp(self):
        self.agent = TreatmentAgent()
    
    def test_treatment_safe_patient(self):
        """Test treatment recommendation for safe patient"""
        patient = {
            "patient_id": "TEST-001",
            "oncology_specific": {
                "cancer_type": "NSCLC",
                "stage": "IV",
                "treatment_intent": "Palliative",
                "performance_status": "ECOG 1"
            },
            "labs": [{
                "wbc": 6.0, "hgb": 13.0, "platelet": 250,
                "creatinine": 1.0
            }],
            "medications": []
        }
        
        risk = {
            "risk_level": "STANDARD",
            "risk_score": 0.15,
            "risk_factors": []
        }
        
        result = self.agent.generate_recommendation(patient, risk)
        
        self.assertNotEqual(result["treatment_recommendation"]["regimen"], "DEFERRED")
        self.assertEqual(result["approval_status"], "AUTO_APPROVED")
    
    def test_treatment_hold_for_low_wbc(self):
        """Test treatment hold for low WBC"""
        patient = {
            "patient_id": "TEST-002",
            "oncology_specific": {
                "cancer_type": "NSCLC",
                "stage": "IV"
            },
            "labs": [{
                "wbc": 2.5, "hgb": 11.0, "platelet": 180,
                "creatinine": 1.0
            }],
            "medications": []
        }
        
        risk = {
            "risk_level": "HIGH_RISK",
            "risk_score": 0.45
        }
        
        result = self.agent.generate_recommendation(patient, risk)
        
        self.assertEqual(result["treatment_recommendation"]["regimen"], "DEFERRED")
        self.assertEqual(result["approval_status"], "HOLD")
    
    def test_treatment_nscil_guidelines(self):
        """Test NCCN guideline compliance for NSCLC"""
        patient = {
            "patient_id": "TEST-003",
            "oncology_specific": {
                "cancer_type": "NSCLC",
                "stage": "IV"
            },
            "labs": [{
                "wbc": 5.5, "hgb": 12.0, "platelet": 200,
                "creatinine": 0.9
            }],
            "medications": []
        }
        
        risk = {"risk_level": "STANDARD", "risk_score": 0.2}
        
        result = self.agent.generate_recommendation(patient, risk)
        
        regimen = result["treatment_recommendation"]["regimen"]
        self.assertIsNotNone(regimen)


class TestAlertAgent(unittest.TestCase):
    """Test suite for Alert Agent"""
    
    def setUp(self):
        self.agent = AlertAgent()
    
    def test_alert_critical_neutropenia(self):
        """Test alert generation for critical neutropenia"""
        patient = {
            "patient_id": "TEST-001",
            "labs": [{
                "wbc": 1.8, "hgb": 10.5, "platelet": 200,
                "creatinine": 1.0
            }]
        }
        
        risk = {"risk_level": "HIGH_RISK", "risk_score": 0.6}
        treatment = {"treatment_recommendation": {"regimen": "Test Regimen"}}
        
        result = self.agent.generate_alerts(patient, risk, treatment)
        
        self.assertGreater(result["summary"]["total_alerts"], 0)
        self.assertTrue(result["summary"]["requires_immediate_action"])
    
    def test_alert_no_issues(self):
        """Test alert generation for patient with no issues"""
        patient = {
            "patient_id": "TEST-002",
            "labs": [{
                "wbc": 6.0, "hgb": 14.0, "platelet": 250
            }],
            "vitals": [{
                "temperature": 37.0, "heart_rate": 72, "sbp": 120, "spo2": 98
            }]
        }
        
        risk = {"risk_level": "STANDARD", "risk_score": 0.1}
        treatment = {"treatment_recommendation": {"regimen": "Standard Care"}}
        
        result = self.agent.generate_alerts(patient, risk, treatment)
        
        self.assertEqual(result["summary"]["critical_count"], 0)
        self.assertFalse(result["summary"]["requires_immediate_action"])
    
    def test_alert_escalation(self):
        """Test alert escalation for critical cases"""
        patient = {
            "patient_id": "TEST-003",
            "labs": [{
                "wbc": 1.5, "hgb": 7.5, "platelet": 45
            }],
            "vitals": [{
                "temperature": 39.0, "heart_rate": 110, "sbp": 85, "spo2": 90
            }]
        }
        
        risk = {"risk_level": "CRITICAL", "risk_score": 0.85}
        treatment = {"treatment_recommendation": {"regimen": "Chemotherapy"}}
        
        result = self.agent.generate_alerts(patient, risk, treatment)
        
        escalation = result["escalation"]
        self.assertEqual(escalation["level"], "IMMEDIATE")


class TestSafetyGuardrails(unittest.TestCase):
    """Test suite for Safety Guardrails"""
    
    def setUp(self):
        self.guardrails = SafetyGuardrails()
    
    def test_safe_labs_pass(self):
        """Test that safe lab values pass"""
        patient = {
            "labs": [{
                "wbc": 6.0, "hgb": 14.0, "platelet": 250,
                "creatinine": 1.0
            }]
        }
        
        result = self.guardrails.check_all("", patient)
        
        self.assertTrue(result.can_proceed)
        self.assertFalse(result.blocked)
    
    def test_critical_wbc_blocked(self):
        """Test that critical WBC blocks treatment"""
        patient = {
            "labs": [{
                "wbc": 1.5, "hgb": 12.0, "platelet": 200
            }]
        }
        
        result = self.guardrails.check_all("", patient)
        
        self.assertTrue(result.blocked)
        self.assertFalse(result.can_proceed)
        self.assertGreater(result.blocked_count, 0)
    
    def test_drug_interaction_detection(self):
        """Test drug interaction detection"""
        has_interaction, severity, recommendation = self.guardrails.check_drug_interaction(
            "CISPLATIN", "AMINOGLYCOSIDE"
        )
        
        self.assertTrue(has_interaction)
        self.assertEqual(severity, SafetyLevel.CRITICAL)
    
    def test_no_interaction_safe_drugs(self):
        """Test that safe drug combinations don't trigger"""
        has_interaction, severity, _ = self.guardrails.check_drug_interaction(
            "Carboplatin", "Ondansetron"
        )
        
        self.assertFalse(has_interaction)


class TestMIMICDataGenerator(unittest.TestCase):
    """Test suite for MIMIC data generator"""
    
    def setUp(self):
        self.generator = MIMICDataGenerator(seed=42)
    
    def test_generate_single_patient(self):
        """Test single patient generation"""
        patient = self.generator.generate_patient("TEST-001")
        
        self.assertEqual(patient.patient_id, "TEST-001")
        self.assertIsNotNone(patient.demography)
        self.assertIsNotNone(patient.labs)
        self.assertGreater(len(patient.labs), 0)
    
    def test_generate_dataset(self):
        """Test dataset generation"""
        dataset = self.generator.generate_dataset(n_patients=100)
        
        self.assertEqual(dataset["metadata"]["total_patients"], 100)
        self.assertEqual(len(dataset["patients"]), 100)
        
        for patient in dataset["patients"][:5]:
            self.assertIn("patient_id", patient)
            self.assertIn("risk_assessment", patient)
    
    def test_risk_distribution(self):
        """Test risk distribution in generated data"""
        dataset = self.generator.generate_dataset(n_patients=500)
        
        high_risk = sum(
            1 for p in dataset["patients"]
            if p["risk_assessment"]["risk_level"] == "HIGH_RISK"
        )
        
        self.assertGreater(high_risk, 5)
        self.assertLess(high_risk, 300)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_full_patient_workflow(self):
        """Test complete patient analysis workflow"""
        from orchestrator import OncologyOrchestrator
        
        orchestrator = OncologyOrchestrator()
        
        patient = {
            "patient_id": "INT-001",
            "demography": {"age": 68, "gender": "M"},
            "labs": [{
                "wbc": 3.5, "hgb": 11.5, "platelet": 180,
                "creatinine": 1.2, "albumin": 3.2
            }],
            "vitals": [{
                "temperature": 37.5, "heart_rate": 82, "sbp": 130
            }],
            "medications": [
                {"drug": "Carboplatin", "class": "platinum"},
                {"drug": "Paclitaxel", "class": "taxane"}
            ],
            "oncology_specific": {
                "cancer_type": "NSCLC",
                "stage": "III",
                "treatment_intent": "Curative",
                "performance_status": "ECOG 1"
            }
        }
        
        result = orchestrator.analyze_patient(patient)
        
        self.assertIsNotNone(result.analysis_id)
        self.assertEqual(result.patient_id, "INT-001")
        self.assertIn(result.overall_status, ["APPROVED", "PENDING", "HOLD"])
        self.assertGreater(len(result.recommendations), 0)


def run_tests():
    """Run all tests and return results"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestRiskAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestTreatmentAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestAlertAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestSafetyGuardrails))
    suite.addTests(loader.loadTestsFromTestCase(TestMIMICDataGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success": result.wasSuccessful()
    }


if __name__ == "__main__":
    results = run_tests()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Run: {results['tests_run']}")
    print(f"Failures: {results['failures']}")
    print(f"Errors: {results['errors']}")
    print(f"Success: {results['success']}")
    print("=" * 60)
