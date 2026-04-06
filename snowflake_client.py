"""
Snowflake Connection Utility
"""
import os
import snowflake.connector
from typing import Optional, Dict, List, Any
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

class SnowflakeClient:
    def __init__(self):
        self.conn = None
        self.config = {
            "user": "SANDHIYABK",
            "password": "k66T4jKv_LQDHXe",
            "account": "rwcfeut-wb78109",
            "warehouse": "COMPUTE_WH",
            "database": "ONCOLOGY_DB",
            "schema": "GOLD"
        }
    
    def connect(self):
        if self.conn is None or self.conn.is_closed():
            self.conn = snowflake.connector.connect(**self.config)
        return self.conn
    
    def close(self):
        if self.conn and not self.conn.is_closed():
            self.conn.close()
    
    @contextmanager
    def cursor(self):
        conn = self.connect()
        cursor = conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        with self.cursor() as cursor:
            cursor.execute(query, params or ())
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            return results
    
    def execute_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        results = self.execute_query(query, params)
        return results[0] if results else None

snowflake_client = SnowflakeClient()

def get_patient_from_snowflake(patient_id: str) -> Optional[Dict]:
    """Query a specific patient from GOLD_PATIENT_RISK table."""
    query = """
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
            RISK_FACTORS_DERIVED
        FROM GOLD_PATIENT_RISK
        WHERE PATIENT_ID = %s
    """
    return snowflake_client.execute_one(query, (patient_id,))

def get_dashboard_stats_from_snowflake() -> Dict:
    """Get aggregated dashboard stats from Snowflake."""
    query = """
        SELECT 
            COUNT(*) AS total_patients,
            SUM(CASE WHEN RISK_LEVEL = 'HIGH_RISK' THEN 1 ELSE 0 END) AS high_risk_count,
            SUM(CASE WHEN RISK_LEVEL = 'STANDARD' THEN 1 ELSE 0 END) AS standard_risk_count,
            AVG(PRE_SCORE) AS avg_risk_score,
            MAX(PRE_SCORE) AS max_risk_score,
            MIN(PRE_SCORE) AS min_risk_score
        FROM GOLD_PATIENT_RISK
    """
    return snowflake_client.execute_one(query)

def get_risk_factors_aggregation() -> List[Dict]:
    """Get aggregated risk factors from Snowflake."""
    query = """
        SELECT 
            RISK_FACTOR,
            COUNT(*) AS count
        FROM (
            SELECT TRIM(value) AS RISK_FACTOR
            FROM GOLD_PATIENT_RISK,
            LATERAL SPLIT_TO_TABLE(RISK_FACTORS_DERIVED, ',')
            WHERE RISK_FACTORS_DERIVED IS NOT NULL
        )
        GROUP BY RISK_FACTOR
        ORDER BY count DESC
        LIMIT 10
    """
    return snowflake_client.execute_query(query)

def get_high_risk_patients(limit: int = 100) -> List[Dict]:
    """Get high risk patients for the alerts endpoint."""
    query = """
        SELECT 
            PATIENT_ID,
            AGE,
            GENDER,
            CANCER_TYPE,
            PRE_SCORE,
            RISK_LEVEL,
            WBC_AVG,
            WBC_SLOPE_TOTAL,
            TUMOR_MARKERS_AVG,
            CHEMO_CYCLE_COUNT,
            HIGH_RISK_FLAG,
            RISK_FACTORS_DERIVED
        FROM GOLD_PATIENT_RISK
        WHERE RISK_LEVEL = 'HIGH_RISK'
        ORDER BY PRE_SCORE DESC
        LIMIT %s
    """
    return snowflake_client.execute_query(query, (limit,))

def init_snowflake_connection():
    """Initialize Snowflake connection on app startup."""
    try:
        snowflake_client.connect()
        print("Snowflake connected successfully")
    except Exception as e:
        print(f"Warning: Could not connect to Snowflake: {e}")
        print("Falling back to local JSON data")
        return False
    return True

def close_snowflake_connection():
    """Close Snowflake connection on app shutdown."""
    snowflake_client.close()
    print("Snowflake connection closed")
