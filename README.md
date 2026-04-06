# Patient Journey AI System

A robust, multi-agent AI system for oncology clinical decision support that leverages PySpark, Snowflake, dbt, FastAPI, and LangGraph.

## Architecture

*   **Ingestion:** Scrambles and ingests synthetic healthcare data using PySpark.
*   **Data Warehouse:** Snowflake serves as the data lake layer, with `RAW` databases populated by PySpark.
*   **Transformation:** `dbt` transforms raw data into `GOLD_PATIENT_RISK` layers for modeling.
*   **Backend & LLM Orchestration:** A FastAPI system running LangGraph agents to run logic and reflexions, supporting Human-In-The-Loop.
*   **Dashboard:** Built with React/Vite and styled with Tailwind to visualize safe limits and approvals.

## Setup

1.  **Environment Setup**: Copy `.env.example` to `.env` and fill in your Snowflake credentials.
    ```bash
    cp .env.example .env
    ```
2.  **Dependencies**: Install Python dependencies.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Frontend**: Move into the `oncology-dashboard` directory and install the necessary dependencies:
    ```bash
    cd oncology-dashboard
    npm install
    npm run dev
    ```
4.  **Backend API**: In the root directory, start FastAPI:
    ```bash
    fastapi dev main.py
    ```

## Development

- Use `python spark_ingest.py` to test your Spark ingestion locally.
- Adjust logic for the agents under `langgraph_oncology.py`.
