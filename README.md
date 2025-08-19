# Automated Transit Data Pipeline with ML Integration
This project builds a complete pipeline for transit delay analysis and prediction. 

## Overview
This project loads GTFS data into PostgreSQL, generates synthetic delays, builds a warehouse layer, trains a machine learning model, automates workflows using Airflow and visualizes results in a Streamlit dashboard.

## Key features
1. Data Engineering Pipeline
- Real GTFS (General Transit Feed Specification) data integration
- Star schema data warehouse 
- Comprehensive ETL pipeline with data validation

2. Machine Learning
- **XGBoost regression model** with 40+ engineered features
- **88.80% test accuracy** (±10 min prediction window)
- Feature engineering: temporal, weather, historical, interaction features

3. Automated Orchestration
- **Apache Airflow** for pipeline automation
- Weekly model retraining (Sundays at 3 AM) and Daily predictions (2 AM)
- Built-in monitoring and alerting framework

4. Performance Optimization
- Strategic database indexing 
- Materialized views for dashboard 
- Database maintenance automation 

5. Interactive Dashboard
- **Streamlit** web application
- Route performance comparison
- Weather impact analysis
- Hourly pattern visualization

## System Architechture
```bash
DATA SOURCES (GTFS, Delays, Weather)
           ↓
    POSTGRESQL DATABASE
    ┌─────────────────┐
    │ Operational     │ → Raw GTFS + delays
    │ Warehouse       │ → Star schema (fact + dims)
    │ ML Schema       │ → Features
    │ Analytics       │ → Materialized views
    └─────────────────┘
           ↓
     MACHINE LEARNING
    ┌─────────────────┐
    │ Feature Eng     │ → 40+ features
    │ XGBoost Model   │ → Training
    │ Predictions     │ → Risk levels
    └─────────────────┘
            ↓
    ┌────────────────────┐
    ↓                    ↓
   Airflow            Streamlit
(Automation)       (Visualization)
```

## Project Structure
```bash
transit-data-pipeline/
│
├── sql/
│   ├── create_schema.sql                # Database schema creation 
│   ├── load_gtfs_data.sql               # GTFS data import
│   ├── data_warehouse.sql               # Star schema creation
│   ├── analysis_queries.sql             # Warehouse ETL
│   ├── ml_feature_engineering.sql       # ML features
│   └── performance_optimization.sql     # Database optimization
│
├── airflow/
│   ├── dags/
│   │   ├── outputs                      # Predicted outputs are stored here (Airflow)
│   │   ├─── ml_pipeline_dag.py          # Airflow DAG definition
│   │   └─── predict_delays.py           # Modified for Airflow 
│   ├── docker-compose.yaml              # Airflow Docker setup
│   ├── Dockefile
│   ├── requirements.txt
│   └── models/                          # Model artifacts
│       ├── xgboost_delay_model.pkl      # Trained model
│       ├── scaler.pkl
│       └── encoders.pkl
│
├── generate_synthetic_data.py          # Delay event generation
├── train_model.py                      # XGBoost training script
├── predict_delays.py                   # Prediction generation
├── streamlit_dashboard.py              # Interactive dashboard
│
├── models/                             # Trained model artifacts
│   ├── xgboost_delay_model.pkl         # Final XGBoost model
│   ├── scaler.pkl                      
│   └── encoders.pkl                    
│
├── outputs/                            # Predicted outputs are stored here (Local setup)
│
├── requirements.txt 
│
└── README.md                            
```

## How to run

### 1. Set Up the Database
Place GTFS files under: data/gtfs/ <br>
Create schema and load GTFS:
```bash
psql -U postgres -d transit_delay_optimization -f sql/create_schema.sql
psql -U postgres -d transit_delay_optimization -f sql/load_gtfs_data.sql
```

### 2. Generate Synthetic Weather & Delay Data
NOTE: Skip this step if your dataset already includes weather and delay events. <br>
Install Python requirements:
```bash
pip install -r requirements.txt
```
Run the generator:
```bash
python scripts/generate_synthetic_data.py
```

### 3. Build Warehouse Tables
```bash
psql -U postgres -d transit_delay_optimization -f sql/fact_dim_tables.sql
```
(Optional) Run analytical queries:
```bash
psql -U postgres -d transit_delay_optimization -f sql/analysis_queries.sql
```
This helps understand patterns such as rush hour trends, weather effects and route performance.

### 4. Create ML Features in PostgreSQL
```bash
psql -U postgres -d transit_delay_optimization -f sql/06_ml_feature_engineering.sql
```
This step builds the ML feature table and train/test views.

### 5. Train the Model
```bash
python scripts/train_delay_model.py
```
This script trains an XGBoost model, evaluates it and saves all artifacts under models/.

### 6. Run Predictions
Test predictions:
```bash
python scripts/predict_delays.py --mode test
```
Future predictions:
```bash
python scripts/predict_delays.py --mode future
```

### 7. Start Airflow Automation (Docker)
NOTE: I am using Docker Desktop for Airflow, but you can use any setup you’re comfortable with for running Airflow. <br>
Build and start Airflow:
```bash
docker-compose build
docker-compose up airflow-init
docker-compose up -d
```
If Airflow has no users on first run, run this:
```bash
docker-compose exec airflow-webserver airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
```
Airflow UI:
```bash
http://localhost:8080
```
Daily and weekly pipelines will appear automatically in the UI after placing the DAG under:
```bash
airflow/dags/ml_pipeline_dag.py
```

### 8. Apply Database Optimization
```bash
psql -U postgres -d transit_delay_optimization -f database/performance_optimization.sql
```

### 9. Run the Dashboard
```bash
streamlit run streamlit_dashboard.py
```
The dashboard shows various features like key delay metrics, daily trends, route comparison, weather impact and model accuracy and feature importance.

## Notes
- Update database credentials inside Python and SQL files before running.
- Model Location: Copy trained models to `airflow/models/` directory for pipeline to access
- Dashboard loads fresh data every time, no caching is used.
