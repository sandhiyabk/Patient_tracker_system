/*
dbt Model: GOLD_PATIENT_RISK
============================
Transforms RAW_PATIENT_DATA into risk-scored patient records.

Risk Score Formula:
  PRE_SCORE = (0.3 * AGE_NORM) + (0.3 * TUMOR_NORM) + (0.2 * CYCLE_NORM) + (0.2 * WBC_SLOPE_NORM)

Thresholds:
  - HIGH_RISK: PRE_SCORE > 0.4
  - STANDARD: PRE_SCORE <= 0.4

Normalization:
  - Age: Age / 100 (max 1.0)
  - Tumor Markers: Tumor_Markers_Avg / 100 (max 1.0)
  - Chemo Cycles: Chemo_Cycle_Count / 10 (max 1.0)
  - WBC Slope Severity: Based on cumulative slope direction and magnitude
*/

WITH base_data AS (
    SELECT 
        PATIENT_ID,
        AGE,
        GENDER,
        CANCER_TYPE,
        WBC_AVG,
        WBC_SLOPE_TOTAL,
        TUMOR_MARKERS_AVG,
        COALESCE(CHEMO_CYCLE_COUNT, 0) AS CHEMO_CYCLE_COUNT,
        HIGH_RISK_FLAG,
        RISK_FACTORS
    FROM {{ source('RAW', 'RAW_PATIENT_DATA') }}
),

normalized_scores AS (
    SELECT 
        PATIENT_ID,
        AGE,
        GENDER,
        CANCER_TYPE,
        WBC_AVG,
        WBC_SLOPE_TOTAL,
        TUMOR_MARKERS_AVG,
        CHEMO_CYCLE_COUNT,
        HIGH_RISK_FLAG,
        RISK_FACTORS,
        
        -- Normalize Age (max 100)
        LEAST(AGE / 100.0, 1.0) AS AGE_NORM,
        
        -- Normalize Tumor Markers (max 100)
        LEAST(COALESCE(TUMOR_MARKERS_AVG, 0) / 100.0, 1.0) AS TUMOR_NORM,
        
        -- Normalize Chemo Cycles (max 10)
        LEAST(CHEMO_CYCLE_COUNT / 10.0, 1.0) AS CYCLE_NORM,
        
        -- Normalize WBC Slope Severity
        -- 0.0 = stable, 0.3 = slight decline, 0.6 = moderate decline, 0.8 = severe decline
        CASE 
            WHEN WBC_SLOPE_TOTAL > 10 THEN 0.3  -- Increasing significantly (possibly infection)
            WHEN WBC_SLOPE_TOTAL > 5 THEN 0.2   -- Slight increase
            WHEN WBC_SLOPE_TOTAL > 0 THEN 0.1    -- Stable/slight increase
            WHEN WBC_SLOPE_TOTAL > -5 THEN 0.2   -- Slight decline
            WHEN WBC_SLOPE_TOTAL > -10 THEN 0.5   -- Moderate decline
            ELSE 0.8                             -- Severe decline (bone marrow concern)
        END AS WBC_SLOPE_NORM
        
    FROM base_data
),

risk_calculation AS (
    SELECT 
        PATIENT_ID,
        AGE,
        GENDER,
        CANCER_TYPE,
        WBC_AVG,
        WBC_SLOPE_TOTAL,
        TUMOR_MARKERS_AVG,
        CHEMO_CYCLE_COUNT,
        HIGH_RISK_FLAG,
        RISK_FACTORS,
        AGE_NORM,
        TUMOR_NORM,
        CYCLE_NORM,
        WBC_SLOPE_NORM,
        
        -- Calculate PRE_SCORE using the formula
        ROUND(
            (0.3 * AGE_NORM) + 
            (0.3 * TUMOR_NORM) + 
            (0.2 * CYCLE_NORM) + 
            (0.2 * WBC_SLOPE_NORM),
            3
        ) AS PRE_SCORE
        
    FROM normalized_scores
),

final_risk AS (
    SELECT 
        PATIENT_ID,
        AGE,
        GENDER,
        CANCER_TYPE,
        WBC_AVG,
        WBC_SLOPE_TOTAL,
        TUMOR_MARKERS_AVG,
        CHEMO_CYCLE_COUNT,
        HIGH_RISK_FLAG,
        RISK_FACTORS,
        AGE_NORM,
        TUMOR_NORM,
        CYCLE_NORM,
        WBC_SLOPE_NORM,
        PRE_SCORE,
        
        -- Flag as HIGH_Risk if PRE_SCORE > 0.4
        CASE 
            WHEN PRE_SCORE > 0.4 THEN 'HIGH_RISK'
            ELSE 'STANDARD'
        END AS RISK_LEVEL,
        
        -- Additional derived risk factors
        CASE WHEN AGE > 70 THEN 'Age > 70' END AS AGE_RISK,
        CASE WHEN TUMOR_MARKERS_AVG > 30 THEN 'Elevated Tumor Markers' END AS TUMOR_RISK,
        CASE WHEN CHEMO_CYCLE_COUNT > 5 THEN 'Multiple Chemo Cycles' END AS CYCLE_RISK,
        CASE WHEN WBC_SLOPE_TOTAL < -5 THEN 'WBC Declining' END AS WBC_RISK
        
    FROM risk_calculation
)

SELECT 
    PATIENT_ID,
    AGE,
    GENDER,
    CANCER_TYPE,
    WBC_AVG,
    WBC_SLOPE_TOTAL,
    TUMOR_MARKERS_AVG,
    CHEMO_CYCLE_COUNT,
    AGE_NORM,
    TUMOR_NORM,
    CYCLE_NORM,
    WBC_SLOPE_NORM,
    PRE_SCORE,
    RISK_LEVEL,
    HIGH_RISK_FLAG,
    
    -- Combine all identified risk factors
    CONCAT_WS(', ', 
        AGE_RISK, 
        TUMOR_RISK, 
        CYCLE_RISK, 
        WBC_RISK
    ) AS RISK_FACTORS_DERIVED,
    
    CURRENT_TIMESTAMP() AS ETL_LOADED_AT
    
FROM final_risk

ORDER BY PRE_SCORE DESC
