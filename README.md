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
