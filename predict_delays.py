import pickle
import psycopg2
import pandas as pd
import numpy as np
from tabulate import tabulate
from datetime import datetime, timedelta
import argparse
import warnings
warnings.filterwarnings('ignore')

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

    # Preprocess
    X = preprocess_features(X, scaler, encoders)
    
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

# PREDICTION ON FUTURE DATA

def predict_future_delays(conn, model, scaler, encoders, target_date=None):
    print("\nMODE: Future Prediction")
    
    if target_date is None:
        # Predict for tomorrow
        target_date = (datetime.now() + timedelta(days=1)).date()
    
    print(f"Predicting delays for: {target_date}")
    
    # Query future trip data
    # This requires future weather forecasts and trip schedules, so we'll use recent data with modified dates to predict future delays
    
    query = f"""
        WITH future_trips AS (
            SELECT DISTINCT
                t.trip_id,
                t.route_id,
                st.stop_id,
                st.stop_sequence,
                st.arrival_time
            FROM operational.trips t
            INNER JOIN operational.stop_times st ON t.trip_id = st.trip_id
            INNER JOIN operational.calendar c ON t.service_id = c.service_id
            WHERE c.{target_date.strftime('%A').lower()} = TRUE
              AND c.start_date <= '{target_date}'
              AND c.end_date >= '{target_date}'
            LIMIT 100  
        )
        SELECT 
            ft.trip_id,
            ft.stop_id,
            ft.route_id,
            
            -- Temporal features (for target date)
            EXTRACT(HOUR FROM ft.arrival_time)::INTEGER as hour_of_day,
            EXTRACT(DOW FROM DATE '{target_date}')::INTEGER as day_of_week,
            EXTRACT(DAY FROM DATE '{target_date}')::INTEGER as day_of_month,
            EXTRACT(MONTH FROM DATE '{target_date}')::INTEGER as month,
            EXTRACT(WEEK FROM DATE '{target_date}')::INTEGER as week_of_year,
            CASE WHEN EXTRACT(DOW FROM DATE '{target_date}') IN (0, 6) THEN 1 ELSE 0 END as is_weekend,
            0 as is_holiday,
            CASE WHEN EXTRACT(HOUR FROM ft.arrival_time) BETWEEN 7 AND 9 
                   OR EXTRACT(HOUR FROM ft.arrival_time) BETWEEN 17 AND 19 
                 THEN 1 ELSE 0 END as is_rush_hour,
            CASE 
                WHEN EXTRACT(MONTH FROM DATE '{target_date}') IN (12, 1, 2) THEN 'Winter'
                WHEN EXTRACT(MONTH FROM DATE '{target_date}') IN (3, 4, 5) THEN 'Spring'
                WHEN EXTRACT(MONTH FROM DATE '{target_date}') IN (6, 7, 8) THEN 'Summer'
                ELSE 'Fall'
            END as season,
            
            -- Route features
            dr.route_type,
            dtri.total_stops as route_total_stops,
            ft.stop_sequence,
            dtri.total_stops - ft.stop_sequence as stops_remaining,
            
            -- Stop features
            ds.is_major_hub::INTEGER as is_major_hub,
            ds.stop_area,
            
            -- Weather features (using recent average or forecast)
            'clear' as weather_condition,  -- Would use forecast API
            1 as weather_severity,
            15.0 as temperature,
            0.0 as precipitation,
            5.0 as wind_speed,
            -- 8.0 as visibility,
            
            -- Historical features (from last 7-30 days)
            COALESCE((
                SELECT AVG(delay_minutes)
                FROM ml.delay_features
                WHERE route_id = ft.route_id
                  AND stop_id = ft.stop_id
                  -- AND actual_arrival_timestamp >= CURRENT_DATE - INTERVAL '7 days'
            ), 0) as avg_delay_same_route_stop_7d,
            
            COALESCE((
                SELECT AVG(delay_minutes)
                FROM ml.delay_features
                WHERE route_id = ft.route_id
                  AND stop_id = ft.stop_id
                  -- AND actual_arrival_timestamp >= CURRENT_DATE - INTERVAL '30 days'
            ), 0) as avg_delay_same_route_stop_30d,
            
            COALESCE((
                SELECT COUNT(*)
                FROM ml.delay_features
                WHERE route_id = ft.route_id
                  AND stop_id = ft.stop_id
                  -- AND actual_arrival_timestamp >= CURRENT_DATE - INTERVAL '7 days'
            ), 0) as delay_count_same_route_stop_7d,
            
            COALESCE((
                SELECT MAX(delay_minutes)
                FROM ml.delay_features
                WHERE route_id = ft.route_id
                  AND stop_id = ft.stop_id
                  -- AND actual_arrival_timestamp >= CURRENT_DATE - INTERVAL '7 days'
            ), 0) as max_delay_same_route_stop_7d,
            
            -- Other historical features 
            0.0 as avg_delay_route_7d,
            0.0 as avg_delay_route_30d,
            0.0 as stddev_delay_route_7d,
            0.0 as avg_delay_stop_7d,
            0.0 as avg_delay_stop_30d,
            0.0 as avg_delay_same_hour_7d,
            0.0 as avg_delay_same_hour_30d,
            0.0 as avg_delay_same_dow_7d,
            0.0 as avg_delay_same_weather_7d,
            0.0 as delay_trend_7d,
            0.0 as delay_volatility_7d,
            0 as prev_stop_delay,
            0.0 as prev_stop_avg_delay_7d,
            
            -- Interaction features
            CASE WHEN EXTRACT(HOUR FROM ft.arrival_time) BETWEEN 7 AND 9 
                   OR EXTRACT(HOUR FROM ft.arrival_time) BETWEEN 17 AND 19 
                 THEN 1.5 ELSE 1.0 END as rush_hour_delay_multiplier,
            1.0 as weather_rush_hour_interaction,
            1.0 as weekend_weather_interaction
            
        FROM future_trips ft
        INNER JOIN warehouse.dim_stop ds ON ft.stop_id = ds.stop_id
        INNER JOIN warehouse.dim_route dr ON ft.route_id = dr.route_id
        INNER JOIN warehouse.dim_trip dtri ON ft.trip_id = dtri.trip_id
    """
    
    print("Loading future trip data...")
    df = pd.read_sql(query, conn)
    print(f" Loaded {len(df):,} future trips")
    
    # Separate identifiers
    identifiers = df[['trip_id', 'stop_id', 'route_id']]
    
    # Prepare features
    X = df.drop(['trip_id', 'stop_id', 'route_id'], axis=1)

    # Preprocess
    X = preprocess_features(X, scaler, encoders)
    
    # Generate predictions
    print("Generating predictions...")
    predicted_delays = model.predict(X)
    predicted_delays = np.round(predicted_delays).astype(int)
    
    # Create results
    results = identifiers.copy()
    results['predicted_delay'] = predicted_delays
    results['prediction_date'] = target_date
    results['risk_level'] = pd.cut(
        predicted_delays,
        bins=[-np.inf, 5, 10, 20, np.inf],
        labels=['Low', 'Medium', 'High', 'Severe']
    )
    
    # Display summary
    print("PREDICTION SUMMARY")
    print(f"Total trips: {len(results):,}")
    print(f"Avg predicted delay: {predicted_delays.mean():.2f} minutes")
    print(f"Max predicted delay: {predicted_delays.max()} minutes")
    print("\nRisk Distribution:")
    print(results['risk_level'].value_counts().to_string())
    
    # Display high-risk trips
    print("HIGH-RISK TRIPS (Predicted delay > 15 minutes)")
    high_risk = results[results['predicted_delay'] > 15].sort_values('predicted_delay', ascending=False)
    print(tabulate(high_risk.head(20), headers='keys', tablefmt='grid', showindex=False))
    
    # Save results
    import os
    os.makedirs('outputs', exist_ok=True)
    results.to_csv(f'outputs/future_predictions_{target_date}.csv', index=False)
    print(f"\n Full results saved to outputs/future_predictions_{target_date}.csv")
    
    return results

# FEATURE PREPROCESSING
def preprocess_features(X, scaler, encoders):
    
    # Identify categorical and numerical columns
    categorical_cols = ['season', 'stop_area', 'weather_condition']
    boolean_cols = [col for col in X.columns if X[col].dtype == 'bool']
    numerical_cols = [col for col in X.columns if col not in categorical_cols and col not in boolean_cols]
    
    # Convert boolean to int
    for col in boolean_cols:
        X[col] = X[col].astype(int)
    
    # Encode categorical features
    for col in categorical_cols:
        if col in encoders:
            le = encoders[col]
            X[col] = X[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # Scale numerical features
    X[numerical_cols] = scaler.transform(X[numerical_cols])
    
    return X

# MAIN EXECUTION
def main():
    parser = argparse.ArgumentParser(description='Generate delay predictions')
    parser.add_argument('--mode', type=str, default='test', 
                       choices=['test', 'future'],
                       help='Prediction mode: test (evaluate) or future (predict)')
    parser.add_argument('--date', type=str, default=None,
                       help='Target date for future predictions (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    print("TRANSIT DELAY PREDICTION")
    
    # Load model
    model, scaler, encoders = load_model_artifacts()
    
    # Connect to database
    conn = get_db_connection()
    
    try:
        if args.mode == 'test':
            results = predict_test_set(conn, model, scaler, encoders)
        else:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').date() if args.date else None
            results = predict_future_delays(conn, model, scaler, encoders, target_date)
        
        print("PREDICTION COMPLETE!")
        
    except Exception as e:
        print(f"\nError during prediction: {e}")
        raise
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()
