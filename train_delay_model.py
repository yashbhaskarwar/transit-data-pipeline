import psycopg2
import xgboost as xgb

# CONFIGURATION
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'transit_delay_optimization',
    'user': 'postgres',      # UPDATE THIS
    'password': 'postgres'   # UPDATE THIS
}

MODEL_CONFIG = {
    'model_path': 'models/xgboost_delay_model.pkl',
    'scaler_path': 'models/scaler.pkl',
    'encoders_path': 'models/encoders.pkl',
    'metrics_path': 'models/training_metrics.json',
    'feature_importance_path': 'models/feature_importance.csv',
    'random_state': 42
}

# Model type
MODEL_TYPE = 'regression'  # Predicting delay minutes

# DATABASE CONNECTION
def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("Connected to database")
        return conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        raise
