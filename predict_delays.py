import pickle

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
}

# MODEL LOADING

def load_model_artifacts():
    print("Loading model artifacts...")
    
    try:
        with open(MODEL_CONFIG['model_path'], 'rb') as f:
            model = pickle.load(f)
        
        with open(MODEL_CONFIG['scaler_path'], 'rb') as f:
            scaler = pickle.load(f)
        
        with open(MODEL_CONFIG['encoders_path'], 'rb') as f:
            encoders = pickle.load(f)
        
        print("Model artifacts loaded successfully")
        return model, scaler, encoders
    
    except FileNotFoundError as e:
        print("Model files not found.")
        raise

