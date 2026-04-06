import json
import random
from datetime import datetime, timedelta

random.seed(42)

CANCER_TYPES = ["NSCLC", "Breast Cancer", "Colorectal", "Prostate", "Melanoma", "Pancreatic", "Ovarian"]
GENDERS = ["Male", "Female"]
MEDICATIONS = ["Carboplatin", "Paclitaxel", "Pembrolizumab", "Trastuzumab", "FOLFOX", "Doxorubicin", "Gemcitabine", "Olaparib"]

def generate_patient(patient_id):
    age = random.randint(35, 85)
    gender = random.choice(GENDERS)
    cancer_type = random.choice(CANCER_TYPES)
    
    lab_results = []
    base_wbc = random.uniform(4.5, 11.0)
    base_hemoglobin = random.uniform(12.0, 17.5)
    base_tumor_marker = random.uniform(0, 20)
    
    for month in range(12):
        date = datetime(2025, 1, 1) + timedelta(days=30 * month)
        
        wbc_variation = random.uniform(-2, 2)
        hb_variation = random.uniform(-1.5, 1.5)
        tm_variation = random.uniform(-5, 10)
        
        if random.random() < 0.15:
            wbc_variation += random.uniform(3, 8)
        
        lab_results.append({
            "Month": month + 1,
            "Date": date.strftime("%Y-%m-%d"),
            "White_Blood_Cell_Count": round(base_wbc + wbc_variation, 2),
            "Hemoglobin": round(base_hemoglobin + hb_variation, 2),
            "Tumor_Markers": round(max(0, base_tumor_marker + tm_variation), 2)
        })
    
    num_cycles = random.randint(2, 8) if cancer_type in ["NSCLC", "Breast Cancer", "Colorectal"] else random.randint(1, 6)
    chemo_cycles = []
    for i in range(num_cycles):
        chemo_cycles.append({
            "Cycle_Number": i + 1,
            "Drug": random.sample(MEDICATIONS, k=random.randint(1, 3)),
            "Date": (datetime(2025, 2, 1) + timedelta(days=42 * i)).strftime("%Y-%m-%d"),
            "Response": random.choice(["Complete", "Partial", "Stable", "Progressive"])
        })
    
    num_radiation = random.randint(0, 30)
    radiation_doses = []
    if num_radiation > 0:
        for i in range(num_radiation):
            radiation_doses.append({
                "Session_Number": i + 1,
                "Dose_Gy": random.choice([2.0, 2.5, 3.0, 3.5]),
                "Date": (datetime(2025, 2, 1) + timedelta(days=i)).strftime("%Y-%m-%d"),
                "Site": random.choice(["Primary", "Metastatic", "Lymph Nodes", "Bone"])
            })
    
    medication_history = []
    for med in random.sample(MEDICATIONS, k=random.randint(2, 5)):
        medication_history.append({
            "Medication": med,
            "Start_Date": (datetime(2025, 1, 15) + timedelta(days=random.randint(0, 180))).strftime("%Y-%m-%d"),
            "End_Date": (datetime(2025, 7, 1) + timedelta(days=random.randint(0, 180))).strftime("%Y-%m-%d") if random.random() > 0.3 else None,
            "Ongoing": random.random() > 0.5
        })
    
    risk_factors = []
    if age > 65:
        risk_factors.append("Age > 65")
    if any(lab["White_Blood_Cell_Count"] > 15 for lab in lab_results):
        risk_factors.append("Elevated WBC")
    if any(lab["Tumor_Markers"] > 30 for lab in lab_results):
        risk_factors.append("High Tumor Markers")
    if len(chemo_cycles) > 5:
        risk_factors.append("Multiple Chemo Cycles")
    
    high_risk = 1 if (
        random.random() < 0.25 or 
        (age > 70 and random.random() < 0.4) or
        (len(risk_factors) >= 2 and random.random() < 0.5)
    ) else 0
    
    return {
        "Patient_Master": {
            "Patient_ID": f"PID-{patient_id:05d}",
            "Age": age,
            "Gender": gender,
            "Cancer_Type": cancer_type
        },
        "Lab_Results": lab_results,
        "Treatment_Logs": {
            "Chemo_Cycles": chemo_cycles,
            "Radiation_Doses": radiation_doses,
            "Medication_History": medication_history
        },
        "Outcome": {
            "High_Risk_Flag": high_risk,
            "Risk_Factors": risk_factors
        }
    }

patients = [generate_patient(i + 1) for i in range(1000)]

output = {
    "metadata": {
        "total_patients": 1000,
        "generated_date": datetime.now().strftime("%Y-%m-%d"),
        "description": "Synthetic Patient Journey Dataset for Oncology"
    },
    "patients": patients
}

with open("patient_journeys.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Generated {len(patients)} patient records")
print(f"High risk patients: {sum(p['Outcome']['High_Risk_Flag'] for p in patients)}")
