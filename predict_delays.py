import pickle
import psycopg2
import pandas as pd
import numpy as np
from tabulate import tabulate

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

# DATABASE CONNECTION

def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        raise

# PREDICTION ON TEST SET
def predict_test_set(conn, model, scaler, encoders):
    print("\nMODE: Test Set Prediction")
    # Load test data with same features as training
    query = """
        SELECT 
            trip_id,
            stop_id,
            route_id,
            
            -- Actual delay (target)
            delay_minutes as actual_delay,
            
            -- Features (same as training)
            hour_of_day,
            day_of_week,
            day_of_month,
            month,
            week_of_year,
            is_weekend::INTEGER as is_weekend,
            is_holiday::INTEGER as is_holiday,
            is_rush_hour::INTEGER as is_rush_hour,
            season,
            
            route_type,
            route_total_stops,
            stop_sequence,
            stops_remaining,
            
            is_major_hub::INTEGER as is_major_hub,
            stop_area,
            
            weather_condition,
            weather_severity,
            COALESCE(temperature, 15.0) as temperature,
            COALESCE(precipitation, 0.0) as precipitation,
            COALESCE(wind_speed, 5.0) as wind_speed,
            -- COALESCE(visibility, 8.0) as visibility,
            
            COALESCE(avg_delay_same_route_stop_7d, 0) as avg_delay_same_route_stop_7d,
            COALESCE(avg_delay_same_route_stop_30d, 0) as avg_delay_same_route_stop_30d,
            COALESCE(delay_count_same_route_stop_7d, 0) as delay_count_same_route_stop_7d,
            COALESCE(max_delay_same_route_stop_7d, 0) as max_delay_same_route_stop_7d,
            
            COALESCE(avg_delay_route_7d, 0) as avg_delay_route_7d,
            COALESCE(avg_delay_route_30d, 0) as avg_delay_route_30d,
            COALESCE(stddev_delay_route_7d, 0) as stddev_delay_route_7d,
            
            COALESCE(avg_delay_stop_7d, 0) as avg_delay_stop_7d,
            COALESCE(avg_delay_stop_30d, 0) as avg_delay_stop_30d,
            
            COALESCE(avg_delay_same_hour_7d, 0) as avg_delay_same_hour_7d,
            COALESCE(avg_delay_same_hour_30d, 0) as avg_delay_same_hour_30d,
            
            COALESCE(avg_delay_same_dow_7d, 0) as avg_delay_same_dow_7d,
            COALESCE(avg_delay_same_weather_7d, 0) as avg_delay_same_weather_7d,
            
            COALESCE(delay_trend_7d, 0) as delay_trend_7d,
            COALESCE(delay_volatility_7d, 0) as delay_volatility_7d,
            
            COALESCE(prev_stop_delay, 0) as prev_stop_delay,
            COALESCE(prev_stop_avg_delay_7d, 0) as prev_stop_avg_delay_7d,
            
            rush_hour_delay_multiplier,
            weather_rush_hour_interaction,
            weekend_weather_interaction
            
            -- actual_arrival_timestamp
            
        FROM ml.test_features
        WHERE delay_minutes IS NOT NULL
        LIMIT 1000  
    """
    
    print("Loading test data...")
    df = pd.read_sql(query, conn)
    print(f"Loaded {len(df):,} test records")
    
    # Separate identifiers and actuals
    identifiers = df[['trip_id', 'stop_id', 'route_id']]
    actual_delays = df['actual_delay'].values
    
    # Prepare features
    X = df.drop(['trip_id', 'stop_id', 'route_id', 'actual_delay'], axis=1)
    
    # Generate predictions
    print("Generating predictions...")
    predicted_delays = model.predict(X)
    predicted_delays = np.round(predicted_delays).astype(int)
    
    # Calculate metrics
    mae = np.mean(np.abs(actual_delays - predicted_delays))
    rmse = np.sqrt(np.mean((actual_delays - predicted_delays) ** 2))
    accuracy_5min = np.mean(np.abs(actual_delays - predicted_delays) <= 5) * 100
    accuracy_10min = np.mean(np.abs(actual_delays - predicted_delays) <= 10) * 100
    
    print("PREDICTION RESULTS")
    print(f"MAE: {mae:.2f} minutes")
    print(f"RMSE: {rmse:.2f} minutes")
    print(f"Accuracy within 5 min: {accuracy_5min:.2f}%")
    print(f"Accuracy within 10 min: {accuracy_10min:.2f}%")
    
    # Create results dataframe
    results = identifiers.copy()
    results['actual_delay'] = actual_delays
    results['predicted_delay'] = predicted_delays
    results['error'] = actual_delays - predicted_delays
    results['abs_error'] = np.abs(results['error'])
    results['within_5min'] = results['abs_error'] <= 5
    results['within_10min'] = results['abs_error'] <= 10
    
    # Display sample predictions
    print("SAMPLE PREDICTIONS (Best and Worst)")
    
    # Best predictions
    best = results.nsmallest(10, 'abs_error')[['trip_id', 'stop_id', 'actual_delay', 'predicted_delay', 'error']]
    print("\nBest Predictions:")
    print(tabulate(best, headers='keys', tablefmt='grid', showindex=False))
    
    # Worst predictions
    worst = results.nlargest(10, 'abs_error')[['trip_id', 'stop_id', 'actual_delay', 'predicted_delay', 'error']]
    print("\nWorst Predictions:")
    print(tabulate(worst, headers='keys', tablefmt='grid', showindex=False))
    
    # Save results
    results.to_csv('outputs/test_predictions.csv', index=False)
    print(f"\nFull results saved to outputs/test_predictions.csv")
    
    return results