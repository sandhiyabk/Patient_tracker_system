"""
MIMIC-IV Inspired Synthetic Patient Data Generator
Generates realistic oncology patient data with clinical features
"""
import json
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class PatientRecord:
    """Structured patient record format"""
    patient_id: str
    subject_id: int
    hadm_id: int
    
    demography: Dict
    vitals: List[Dict]
    labs: List[Dict]
    medications: List[Dict]
    procedures: List[Dict]
    diagnoses: List[Dict]
    transfers: List[Dict]
    
    oncology_specific: Dict
    risk_assessment: Dict
    
    @classmethod
    def to_dict(cls, record: 'PatientRecord') -> Dict:
        return {
            'patient_id': record.patient_id,
            'subject_id': record.subject_id,
            'hadm_id': record.hadm_id,
            'demography': record.demography,
            'vitals': record.vitals,
            'labs': record.labs,
            'medications': record.medications,
            'procedures': record.procedures,
            'diagnoses': record.diagnoses,
            'transfers': record.transfers,
            'oncology_specific': record.oncology_specific,
            'risk_assessment': record.risk_assessment
        }


class MIMICDataGenerator:
    """Generate MIMIC-IV inspired synthetic oncology patient data"""
    
    CANCER_TYPES = {
        "NSCLC": {"icd9": "162.9", "icd10": "C34.90", "prevalence": 0.25},
        "Breast Cancer": {"icd9": "174.9", "icd10": "C50.90", "prevalence": 0.22},
        "Colorectal": {"icd9": "153.9", "icd10": "C18.90", "prevalence": 0.18},
        "Prostate": {"icd9": "185", "icd10": "C61", "prevalence": 0.15},
        "Pancreatic": {"icd9": "157.9", "icd10": "C25.90", "prevalence": 0.08},
        "Ovarian": {"icd9": "183.0", "icd10": "C56.90", "prevalence": 0.07},
        "Melanoma": {"icd9": "172.9", "icd10": "C43.90", "prevalence": 0.05}
    }
    
    GENDERS = ["M", "F"]
    ETHNICITIES = ["white", "black", "hispanic", "asian", "other"]
    
    CHEMOTHERAPY_DRUGS = {
        "Carboplatin": {"class": "platinum", "myelosuppression": 0.7, "nephrotoxicity": 0.2},
        "Cisplatin": {"class": "platinum", "myelosuppression": 0.8, "nephrotoxicity": 0.5},
        "Paclitaxel": {"class": "taxane", "myelosuppression": 0.6, "neurotoxicity": 0.4},
        "Docetaxel": {"class": "taxane", "myelosuppression": 0.65, "neurotoxicity": 0.35},
        "Pembrolizumab": {"class": "immunotherapy", "myelosuppression": 0.2, "immunotoxicity": 0.3},
        "Nivolumab": {"class": "immunotherapy", "myelosuppression": 0.15, "immunotoxicity": 0.25},
        "Trastuzumab": {"class": "targeted", "cardiotoxicity": 0.05},
        "Bevacizumab": {"class": "targeted", "hypertension": 0.25, "bleeding": 0.15},
        "Gemcitabine": {"class": "antimetabolite", "myelosuppression": 0.5},
        "FOLFIRI": {"class": "combination", "myelosuppression": 0.6, "gi_toxicity": 0.4},
        "FOLFOX": {"class": "combination", "myelosuppression": 0.55, "neurotoxicity": 0.3},
        "Doxorubicin": {"class": "anthracycline", "cardiotoxicity": 0.15, "myelosuppression": 0.7},
        "Olaparib": {"class": "parp_inhibitor", "myelosuppression": 0.4, "mds_risk": 0.02}
    }
    
    SUPPORTIVE_CARE = [
        "Ondansetron", "Granisetron", "Dexamethasone",
        "Filgrastim", "Pegfilgrastim", "Epoetin alfa",
        "Transfusion PRBC", "Transfusion Platelets"
    ]
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.patient_counter = 100000
        self.admission_counter = 200000
    
    def _generate_subject_id(self) -> int:
        """Generate MIMIC-style subject ID"""
        self.patient_counter += 1
        return self.patient_counter
    
    def _generate_hadm_id(self) -> int:
        """Generate MIMIC-style hospital admission ID"""
        self.admission_counter += 1
        return self.admission_counter
    
    def _generate_age(self, cancer_type: str) -> int:
        """Generate age based on cancer type distribution"""
        base_ages = {
            "NSCLC": (65, 80),
            "Breast Cancer": (50, 75),
            "Colorectal": (55, 78),
            "Prostate": (65, 85),
            "Pancreatic": (60, 80),
            "Ovarian": (50, 75),
            "Melanoma": (45, 70)
        }
        low, high = base_ages.get(cancer_type, (55, 75))
        return random.randint(low, high)
    
    def _generate_demographics(self, subject_id: int, cancer_type: str) -> Dict:
        """Generate patient demographics"""
        gender = random.choice(self.GENDERS)
        age = self._generate_age(cancer_type)
        
        ethnicity_weights = [0.60, 0.12, 0.12, 0.06, 0.10]
        ethnicity = random.choices(self.ETHNICITIES, weights=ethnicity_weights)[0]
        
        dob = datetime.now() - timedelta(days=age*365 + random.randint(0, 365))
        
        return {
            "subject_id": subject_id,
            "gender": gender,
            "dob": dob.strftime("%Y-%m-%d"),
            "age": age,
            "ethnicity": ethnicity,
            "insurance": random.choice(["Medicare", "Medicaid", "Private", "Self-Pay"]),
            "marital_status": random.choice(["Married", "Single", "Divorced", "Widowed"])
        }
    
    def _generate_vitals(self, admit_time: datetime, days: int = 14) -> List[Dict]:
        """Generate vital sign observations"""
        vitals = []
        base_hr = random.randint(70, 85)
        base_sbp = random.randint(115, 135)
        base_dbp = random.randint(70, 85)
        base_temp = 36.8
        base_spo2 = 97
        
        for day in range(days):
            vitals.append({
                "charttime": (admit_time + timedelta(hours=day*6)).strftime("%Y-%m-%d %H:%M:%S"),
                "heart_rate": base_hr + random.randint(-10, 15),
                "sbp": base_sbp + random.randint(-15, 20),
                "dbp": base_dbp + random.randint(-10, 10),
                "temperature": round(base_temp + random.uniform(-0.5, 1.0), 1),
                "spo2": min(100, base_spo2 + random.randint(-3, 2)),
                "respiratory_rate": random.randint(14, 20)
            })
        
        return vitals
    
    def _generate_labs(self, admit_time: datetime, cancer_type: str, days: int = 14) -> List[Dict]:
        """Generate laboratory results"""
        labs = []
        
        base_wbc = random.uniform(5.0, 10.0)
        base_hgb = random.uniform(12.0, 15.0) if random.choice(["M", "F"]) == "F" else random.uniform(13.0, 17.0)
        base_platelet = random.uniform(150, 400)
        base_creatinine = random.uniform(0.7, 1.2)
        base_alt = random.uniform(20, 40)
        base_ast = random.uniform(15, 35)
        base_albumin = random.uniform(3.5, 5.0)
        base_tumor_marker = random.uniform(5, 50)
        
        high_risk = random.random() < 0.35
        
        for day in range(days):
            wbc_variation = random.uniform(-1.5, 1.5)
            hgb_variation = random.uniform(-1.0, 1.0)
            
            if high_risk and day > 5:
                wbc_variation -= random.uniform(2, 4)
                hgb_variation -= random.uniform(1, 3)
            
            tumor_trend = (day * random.uniform(0.5, 2.0)) if high_risk else random.uniform(-1, 1)
            
            labs.append({
                "charttime": (admit_time + timedelta(hours=day*8)).strftime("%Y-%m-%d %H:%M:%S"),
                "wbc": round(max(1.0, base_wbc + wbc_variation), 1),
                "hgb": round(max(6.0, base_hgb + hgb_variation), 1),
                "platelet": round(max(50, base_platelet + random.uniform(-30, 30)), 0),
                "creatinine": round(base_creatinine + random.uniform(-0.1, 0.2), 2),
                "alt": round(max(5, base_alt + random.uniform(-5, 15)), 0),
                "ast": round(max(5, base_ast + random.uniform(-5, 10)), 0),
                "albumin": round(max(2.0, base_albumin + random.uniform(-0.3, 0.3)), 1),
                "tumor_marker_ca125": round(base_tumor_marker + tumor_trend + random.uniform(-5, 5), 1),
                "tumor_marker_cea": round((base_tumor_marker * 0.5) + tumor_trend * 0.3 + random.uniform(-2, 2), 1),
                "tumor_marker_afp": round(random.uniform(3, 15) + tumor_trend * 0.2, 1)
            })
        
        return labs
    
    def _generate_medications(self, admit_time: datetime, cancer_type: str, days: int = 14) -> List[Dict]:
        """Generate medication administrations"""
        meds = []
        
        chemo_drugs = [d for d in self.CHEMOTHERAPY_DRUGS.keys() if self.CHEMOTHERAPY_DRUGS[d]["class"] != "parp_inhibitor"]
        
        num_chemo = random.randint(1, 3)
        selected_chemos = random.sample(chemo_drugs, k=min(num_chemo, len(chemo_drugs)))
        
        chemo_start_day = random.randint(1, 3)
        
        for chemo in selected_chemos:
            chemo_info = self.CHEMOTHERAPY_DRUGS[chemo]
            dose = random.choice([1.0, 1.5, 2.0]) * 100
            
            meds.append({
                "starttime": (admit_time + timedelta(days=chemo_start_day)).strftime("%Y-%m-%d %H:%M:%S"),
                "endtime": (admit_time + timedelta(days=chemo_start_day + random.randint(0, 2))).strftime("%Y-%m-%d %H:%M:%S"),
                "drug": chemo,
                "dose": dose,
                "unit": "mg",
                "route": "IV",
                "class": chemo_info["class"]
            })
        
        supportive = random.sample(self.SUPPORTIVE_CARE, k=random.randint(2, 4))
        for med in supportive:
            meds.append({
                "starttime": (admit_time + timedelta(hours=random.randint(1, 48))).strftime("%Y-%m-%d %H:%M:%S"),
                "endtime": None,
                "drug": med,
                "dose": random.randint(1, 10) * 100,
                "unit": "mg",
                "route": random.choice(["IV", "PO", "SC"]),
                "class": "supportive"
            })
        
        return meds
    
    def _generate_procedures(self, admit_time: datetime, cancer_type: str) -> List[Dict]:
        """Generate procedures"""
        procedures = []
        
        procedures.append({
            "starttime": (admit_time + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "procedure": "CT Chest Abdomen Pelvis",
            "code": "71250",
            "category": "imaging"
        })
        
        if random.random() < 0.7:
            procedures.append({
                "starttime": (admit_time + timedelta(days=random.randint(2, 5))).strftime("%Y-%m-%d %H:%M:%S"),
                "procedure": "Tumor Biopsy",
                "code": "11111",
                "category": "diagnostic"
            })
        
        if random.random() < 0.5:
            procedures.append({
                "starttime": (admit_time + timedelta(days=random.randint(3, 7))).strftime("%Y-%m-%d %H:%M:%S"),
                "procedure": "Port-a-Cath Placement",
                "code": "36561",
                "category": "surgical"
            })
        
        return procedures
    
    def _generate_diagnoses(self, cancer_type: str) -> List[Dict]:
        """Generate diagnoses"""
        cancer_info = self.CANCER_TYPES.get(cancer_type, self.CANCER_TYPES["NSCLC"])
        
        diagnoses = [
            {
                "icd9_code": cancer_info["icd9"],
                "icd10_code": cancer_info["icd10"],
                "diagnosis": f"Primary Malignant Neoplasm of {cancer_type}",
                "type": "PRIMARY",
                "seq_num": 1
            }
        ]
        
        common_complications = [
            ("285.9", "D63.8", "Anemia in neoplastic disease"),
            ("199.1", "C80.1", "Malignant neoplasm without specification of site"),
            ("287.5", "D69.6", "Thrombocytopenia"),
            ("288.0", "D70.9", "Neutropenia")
        ]
        
        for icd9, icd10, dx in random.sample(common_complications, k=random.randint(1, 3)):
            diagnoses.append({
                "icd9_code": icd9,
                "icd10_code": icd10,
                "diagnosis": dx,
                "type": "SECONDARY",
                "seq_num": len(diagnoses) + 1
            })
        
        return diagnoses
    
    def _generate_transfers(self, admit_time: datetime) -> List[Dict]:
        """Generate unit transfers"""
        transfers = [
            {
                "intime": admit_time.strftime("%Y-%m-%d %H:%M:%S"),
                "outtime": (admit_time + timedelta(days=random.randint(1, 3))).strftime("%Y-%m-%d %H:%M:%S"),
                "careunit": "Medical Oncology",
                "wardid": random.randint(100, 150)
            }
        ]
        
        if random.random() < 0.3:
            transfers.append({
                "intime": transfers[0]["outtime"],
                "outtime": (admit_time + timedelta(days=random.randint(4, 10))).strftime("%Y-%m-%d %H:%M:%S"),
                "careunit": "Intensive Care Unit",
                "wardid": random.randint(200, 210)
            })
        
        return transfers
    
    def _assess_risk(self, labs: List[Dict], meds: List[Dict], demographics: Dict) -> Dict:
        """Assess patient risk level"""
        if not labs:
            return {"risk_level": "STANDARD", "pre_score": 0.3, "risk_factors": []}
        
        latest = labs[-1]
        
        risk_score = 0.0
        risk_factors = []
        
        if demographics.get("age", 60) > 70:
            risk_score += 0.15
            risk_factors.append("Advanced Age")
        
        if latest.get("wbc", 7) < 3.5:
            risk_score += 0.25
            risk_factors.append("Neutropenia")
        
        if latest.get("hgb", 13) < 10:
            risk_score += 0.20
            risk_factors.append("Anemia")
        
        if latest.get("platelet", 200) < 100:
            risk_score += 0.15
            risk_factors.append("Thrombocytopenia")
        
        if latest.get("tumor_marker_ca125", 20) > 50:
            risk_score += 0.15
            risk_factors.append("Elevated Tumor Markers")
        
        chemo_intensity = sum(1 for m in meds if m["class"] != "supportive")
        if chemo_intensity > 2:
            risk_score += 0.10
            risk_factors.append("High Chemo Intensity")
        
        risk_score = min(risk_score, 1.0)
        
        if risk_score >= 0.5:
            risk_level = "HIGH_RISK"
        elif risk_score >= 0.3:
            risk_level = "MODERATE_RISK"
        else:
            risk_level = "STANDARD"
        
        return {
            "risk_level": risk_level,
            "pre_score": round(risk_score, 3),
            "risk_factors": risk_factors,
            "ground_truth_high_risk": 1 if risk_level == "HIGH_RISK" else 0
        }
    
    def generate_patient(self, patient_id: str) -> PatientRecord:
        """Generate a single patient record"""
        subject_id = self._generate_subject_id()
        hadm_id = self._generate_hadm_id()
        
        cancer_type = random.choices(
            list(self.CANCER_TYPES.keys()),
            weights=[v["prevalence"] for v in self.CANCER_TYPES.values()]
        )[0]
        
        admit_time = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 300))
        
        demographics = self._generate_demographics(subject_id, cancer_type)
        vitals = self._generate_vitals(admit_time)
        labs = self._generate_labs(admit_time, cancer_type)
        medications = self._generate_medications(admit_time, cancer_type)
        procedures = self._generate_procedures(admit_time, cancer_type)
        diagnoses = self._generate_diagnoses(cancer_type)
        transfers = self._generate_transfers(admit_time)
        
        risk = self._assess_risk(labs, medications, demographics)
        
        oncology_specific = {
            "cancer_type": cancer_type,
            "stage": random.choice(["I", "II", "III", "IV"]),
            "grade": random.choice(["G1", "G2", "G3", "G4"]),
            "er_status": random.choice(["Positive", "Negative", "Not Applicable"]) if cancer_type == "Breast Cancer" else "Not Applicable",
            "pr_status": random.choice(["Positive", "Negative", "Not Applicable"]) if cancer_type == "Breast Cancer" else "Not Applicable",
            "her2_status": random.choice(["Positive", "Negative", "Not Applicable"]) if cancer_type in ["Breast Cancer", "Gastric"] else "Not Applicable",
            "kras_mutation": random.choice(["Mutated", "Wild-type", "Not Tested"]) if cancer_type in ["NSCLC", "Colorectal"] else "Not Tested",
            "treatment_intent": random.choice(["Curative", "Palliative"]),
            "performance_status": random.choice(["ECOG 0", "ECOG 1", "ECOG 2", "ECOG 3"])
        }
        
        return PatientRecord(
            patient_id=patient_id,
            subject_id=subject_id,
            hadm_id=hadm_id,
            demography=demographics,
            vitals=vitals,
            labs=labs,
            medications=medications,
            procedures=procedures,
            diagnoses=diagnoses,
            transfers=transfers,
            oncology_specific=oncology_specific,
            risk_assessment=risk
        )
    
    def generate_dataset(self, n_patients: int = 1000) -> Dict:
        """Generate a complete dataset"""
        print(f"Generating {n_patients} MIMIC-IV inspired patient records...")
        
        patients = []
        high_risk_count = 0
        
        for i in range(n_patients):
            patient_id = f"PID-{i+1:05d}"
            patient = self.generate_patient(patient_id)
            patients.append(PatientRecord.to_dict(patient))
            
            if patient.risk_assessment["risk_level"] == "HIGH_RISK":
                high_risk_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{n_patients}")
        
        return {
            "metadata": {
                "total_patients": n_patients,
                "high_risk_count": high_risk_count,
                "standard_count": n_patients - high_risk_count,
                "generated_date": datetime.now().isoformat(),
                "data_source": "MIMIC-IV Inspired Synthetic",
                "version": "1.0.0"
            },
            "patients": patients
        }


def generate_mimic_dataset(output_file: str = "mimic_patients.json", n_patients: int = 1000) -> Dict:
    """Generate and save MIMIC-IV inspired dataset"""
    generator = MIMICDataGenerator(seed=42)
    dataset = generator.generate_dataset(n_patients)
    
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\nDataset saved to {output_file}")
    print(f"Total patients: {dataset['metadata']['total_patients']}")
    print(f"High risk: {dataset['metadata']['high_risk_count']}")
    print(f"Standard: {dataset['metadata']['standard_count']}")
    
    return dataset


if __name__ == "__main__":
    generate_mimic_dataset("mimic_patients.json", 1000)
