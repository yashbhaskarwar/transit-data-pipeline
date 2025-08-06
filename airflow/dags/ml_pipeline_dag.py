from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
import pickle
import numpy as np

# DAG DEFAULT ARGUMENTS
default_args = {
    'owner': 'transit_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email': ['EMAIL'],  # UPDATE THIS
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# DAG DEFINITIONS

# Daily prediction pipeline
dag_daily = DAG(
    'transit_delay_prediction_daily',
    default_args=default_args,
    description='Daily delay prediction pipeline',
    schedule_interval='0 2 * * *',  # 2 AM daily
    catchup=False,
    tags=['transit', 'ml', 'daily', 'prediction']
)

# Weekly model retraining
dag_weekly = DAG(
    'transit_delay_model_retraining',
    default_args=default_args,
    description='Weekly model retraining pipeline',
    schedule_interval='0 3 * * 0',  # 3 AM every Sunday
    catchup=False,
    tags=['transit', 'ml', 'weekly', 'training']
)

# DATA QUALITY CHECKS

def check_data_quality(**context):
    hook = PostgresHook(postgres_conn_id='transit_db')
    conn = hook.get_conn()
    cursor = conn.cursor()
    
    checks = []
    
    # Check 1: Recent delay events exist
    cursor.execute("""
        SELECT COUNT(*) 
        FROM operational.delay_events 
        WHERE recorded_at >= CURRENT_DATE - INTERVAL '7 days'
    """)
    recent_delays = cursor.fetchone()[0]
    checks.append(('Recent delays', recent_delays > 0, f"{recent_delays} records"))
    
    # Check 2: No null critical features
    cursor.execute("""
        SELECT COUNT(*) 
        FROM ml.delay_features 
        WHERE delay_minutes IS NULL 
           OR hour_of_day IS NULL
    """)
    null_features = cursor.fetchone()[0]
    checks.append(('Feature completeness', null_features == 0, f"{null_features} nulls"))
    
    # Check 3: Feature table updated recently
    cursor.execute("""
        SELECT MAX(created_at) 
        FROM ml.delay_features
    """)
    last_update = cursor.fetchone()[0]
    checks.append(('Feature freshness', last_update is not None, str(last_update)))
    
    cursor.close()
    conn.close()
    
    # Evaluate checks
    print("\nData Quality Check Results:")
    all_passed = True
    for check_name, passed, detail in checks:
        status = "PASS" if passed else "✗ FAIL"
        print(f"{status} | {check_name}: {detail}")
        if not passed:
            all_passed = False
    
    if not all_passed:
        raise ValueError("Data quality checks failed!")
    
    print("All data quality checks passed")
    
    return True

# UPDATE ML FEATURES
def update_ml_features(**context):
    hook = PostgresHook(postgres_conn_id='transit_db')
    conn = hook.get_conn()
    cursor = conn.cursor()
    
    # Get latest date in features table (not timestamp!)
    cursor.execute("SELECT MAX(date) FROM ml.delay_features")
    last_feature_date = cursor.fetchone()[0]
    
    if last_feature_date:
        print(f"Last feature date: {last_feature_date}")
        
        # Check if there are new delays to process
        cursor.execute(f"""
            SELECT COUNT(*)
            FROM operational.delay_events
            WHERE DATE(actual_arrival) > '{last_feature_date}'
        """)
        new_delays = cursor.fetchone()[0]
        
        if new_delays == 0:
            print("No new delays to process")
            cursor.close()
            conn.close()
            context['task_instance'].xcom_push(key='features_updated', value=False)
            return True
        
        print(f"Processing {new_delays} new delay events...")
        
        # Insert only new records
        cursor.execute(f"""
            INSERT INTO ml.delay_features (
                -- Identifiers
                trip_id, stop_id, route_id,
                
                -- Target
                delay_minutes, delay_category,
                
                -- Temporal (11 columns)
                date, day_of_week, day_of_month, hour_of_day, minute_of_hour,
                week_of_year, is_weekend, is_holiday, is_rush_hour, month, season,
                
                -- Route (4 columns)
                route_type, route_total_stops, stop_sequence, stops_remaining,
                
                -- Weather (5 columns)
                temperature, precipitation, wind_speed,
                weather_condition, weather_severity,
                
                -- Historical 7-day (9 columns)
                avg_delay_same_route_stop_7d, delay_count_same_route_stop_7d,
                max_delay_same_route_stop_7d,
                avg_delay_route_7d, stddev_delay_route_7d,
                avg_delay_stop_7d,
                avg_delay_same_hour_7d, avg_delay_same_dow_7d,
                avg_delay_same_weather_7d,
                
                -- Historical 30-day (4 columns)
                avg_delay_same_route_stop_30d, avg_delay_route_30d,
                avg_delay_stop_30d, avg_delay_same_hour_30d,
                
                -- Trend (2 columns)
                delay_trend_7d, delay_volatility_7d,
                
                -- Cascade (2 columns)
                prev_stop_delay, prev_stop_avg_delay_7d,
                
                -- Stop (2 columns)
                is_major_hub, stop_area,
                
                -- Interactions (3 columns)
                rush_hour_delay_multiplier,
                weather_rush_hour_interaction,
                weekend_weather_interaction
            )
            SELECT 
                -- Identifiers
                de.trip_id,
                de.stop_id,
                t.route_id,
                
                -- Target
                de.delay_minutes,
                CASE 
                    WHEN de.delay_minutes <= 5 THEN 'Minor'
                    WHEN de.delay_minutes <= 15 THEN 'Moderate'
                    WHEN de.delay_minutes <= 30 THEN 'Severe'
                    ELSE 'Extreme'
                END,
                
                -- Temporal features
                DATE(de.actual_arrival),
                de.day_of_week,
                EXTRACT(DAY FROM de.actual_arrival)::INTEGER,
                EXTRACT(HOUR FROM de.actual_arrival)::INTEGER,
                EXTRACT(MINUTE FROM de.actual_arrival)::INTEGER,
                EXTRACT(WEEK FROM de.actual_arrival)::INTEGER,
                CASE WHEN de.day_of_week IN (5, 6) THEN TRUE ELSE FALSE END,
                de.is_holiday,
                CASE 
                    WHEN EXTRACT(HOUR FROM de.actual_arrival) BETWEEN 7 AND 9 
                      OR EXTRACT(HOUR FROM de.actual_arrival) BETWEEN 17 AND 19 
                    THEN TRUE ELSE FALSE 
                END,
                EXTRACT(MONTH FROM de.actual_arrival)::INTEGER,
                CASE 
                    WHEN EXTRACT(MONTH FROM de.actual_arrival) IN (12, 1, 2) THEN 'Winter'
                    WHEN EXTRACT(MONTH FROM de.actual_arrival) IN (3, 4, 5) THEN 'Spring'
                    WHEN EXTRACT(MONTH FROM de.actual_arrival) IN (6, 7, 8) THEN 'Summer'
                    ELSE 'Fall'
                END,
                
                -- Route features
                r.route_type,
                (SELECT COUNT(*) FROM operational.stop_times WHERE trip_id = de.trip_id),
                COALESCE(st.stop_sequence, 0),
                GREATEST(0, (SELECT MAX(stop_sequence) FROM operational.stop_times WHERE trip_id = de.trip_id) - COALESCE(st.stop_sequence, 0)),
                
                -- Weather features
                COALESCE(
                    (SELECT temperature FROM operational.weather_data 
                     WHERE DATE_TRUNC('hour', recorded_at) = DATE_TRUNC('hour', de.actual_arrival)
                     LIMIT 1), 
                    15.0
                ),
                COALESCE(
                    (SELECT precipitation FROM operational.weather_data 
                     WHERE DATE_TRUNC('hour', recorded_at) = DATE_TRUNC('hour', de.actual_arrival)
                     LIMIT 1), 
                    0.0
                ),
                COALESCE(
                    (SELECT wind_speed FROM operational.weather_data 
                     WHERE DATE_TRUNC('hour', recorded_at) = DATE_TRUNC('hour', de.actual_arrival)
                     LIMIT 1), 
                    5.0
                ),
                de.weather_condition,
                CASE 
                    WHEN de.weather_condition IN ('clear', 'partly_cloudy', 'cloudy') THEN 1
                    WHEN de.weather_condition IN ('rainy', 'fog', 'windy') THEN 2
                    ELSE 3
                END,
                
                -- Historical features (simplified - defaults for incremental)
                -- In production, you'd compute these from recent data
                0.0, 0, 0,  -- 7d route-stop
                0.0, 0.0,   -- 7d route
                0.0,        -- 7d stop
                0.0, 0.0, 0.0,  -- 7d hour/dow/weather
                0.0, 0.0, 0.0, 0.0,  -- 30d
                0.0, 0.0,   -- trends
                0, 0.0,     -- cascade
                
                -- Stop features
                COALESCE(s.is_major_hub, FALSE),
                COALESCE(s.stop_area, 'Unknown'),
                
                -- Interaction features
                CASE 
                    WHEN EXTRACT(HOUR FROM de.actual_arrival) BETWEEN 7 AND 9 
                      OR EXTRACT(HOUR FROM de.actual_arrival) BETWEEN 17 AND 19 THEN 1.5
                    ELSE 1.0
                END,
                CASE 
                    WHEN (EXTRACT(HOUR FROM de.actual_arrival) BETWEEN 7 AND 9 
                          OR EXTRACT(HOUR FROM de.actual_arrival) BETWEEN 17 AND 19)
                      AND de.weather_condition IN ('rainy', 'heavy_rain', 'snow') THEN 3
                    WHEN (EXTRACT(HOUR FROM de.actual_arrival) BETWEEN 7 AND 9 
                          OR EXTRACT(HOUR FROM de.actual_arrival) BETWEEN 17 AND 19) THEN 1
                    ELSE 0
                END,
                CASE 
                    WHEN de.day_of_week IN (5, 6) 
                      AND de.weather_condition IN ('rainy', 'heavy_rain', 'snow') THEN 2
                    ELSE 0
                END
                
            FROM operational.delay_events de
            INNER JOIN operational.trips t ON de.trip_id = t.trip_id
            INNER JOIN operational.routes r ON t.route_id = r.route_id
            LEFT JOIN operational.stop_times st ON de.trip_id = st.trip_id AND de.stop_id = st.stop_id
            LEFT JOIN warehouse.dim_stop s ON de.stop_id = s.stop_id
            WHERE DATE(de.actual_arrival) > '{last_feature_date}'
        """)
        
        rows_inserted = cursor.rowcount
        conn.commit()
        print(f"Inserted {rows_inserted} new feature records")
        context['task_instance'].xcom_push(key='features_updated', value=True)
    else:
        print("No existing features found, skipping incremental update")
        context['task_instance'].xcom_push(key='features_updated', value=False)
    
    cursor.close()
    conn.close()
    
    return True

# GENERATE PREDICTIONS

def generate_predictions(**context):
    """Generate predictions by calling the working predict_delays.py script"""
    import subprocess
    
    result = subprocess.run(
        ['python3', '/opt/airflow/dags/predict_delays.py', '--mode', 'test'],
        capture_output=True,
        text=True,
        cwd='/opt/airflow/dags',
        timeout=300
    )
    
    if result.returncode == 0:
        print("Predictions generated successfully!")
        print(result.stdout)
        
        # Try to extract prediction count
        import re
        match = re.search(r'Total\s+(?:trips|predictions):\s*(\d+)', result.stdout, re.IGNORECASE)
        if match:
            pred_count = int(match.group(1))
            context['task_instance'].xcom_push(key='trip_count', value=pred_count)
        
        return True
    else:
        print("Prediction script failed")
        print(result.stderr)
        raise Exception(f"Prediction failed: {result.stderr}")
    
# MONITORING & ALERTS

def monitor_predictions(**context):
    hook = PostgresHook(postgres_conn_id='transit_db')
    conn = hook.get_conn()
    cursor = conn.cursor()
    
    # Check if predictions table exists
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'ml' 
            AND table_name = 'daily_predictions'
        )
    """)
    table_exists = cursor.fetchone()[0]
    
    if not table_exists:
        print("No predictions table yet")
        cursor.close()
        conn.close()
        return True
    
    # Check if there are any predictions
    cursor.execute("SELECT COUNT(*) FROM ml.daily_predictions")
    pred_count = cursor.fetchone()[0]
    
    if pred_count == 0:
        print("No predictions to monitor yet")
        cursor.close()
        conn.close()
        return True
    
    # Monitor predictions
    cursor.execute("""
        SELECT 
            COUNT(*) as total_predictions,
            AVG(ABS(p.predicted_delay - f.delay_minutes)) as mae,
            COUNT(*) FILTER (WHERE ABS(p.predicted_delay - f.delay_minutes) <= 10) * 100.0 / COUNT(*) as accuracy_10min
        FROM ml.daily_predictions p
        INNER JOIN ml.delay_features f ON 
            p.trip_id = f.trip_id 
            AND p.stop_id = f.stop_id
            AND f.date = p.prediction_date
        WHERE p.created_at >= CURRENT_DATE - INTERVAL '7 days'
    """)
    
    result = cursor.fetchone()
    
    if result and result[0] > 0:
        total, mae, accuracy = result
        print(f"\nPrediction Performance (Last 7 days):")
        print(f"  Total predictions validated: {total}")
        print(f"  MAE: {mae:.2f} minutes")
        print(f"  Accuracy (10min): {accuracy:.2f}%")
        
        # Alert if accuracy drops
        if accuracy < 75:
            print("WARNING: Model accuracy below 75%!")
            context['task_instance'].xcom_push(key='alert_needed', value=True)
    else:
        print("No predictions matched with actual delays yet")
    
    cursor.close()
    conn.close()
    return True

def send_high_risk_alert(**context):
    ti = context['task_instance']
    high_risk_count = ti.xcom_pull(task_ids='generate_predictions', key='high_risk_count')
    
    if not high_risk_count:
        print("No high-risk count from predictions (check if predictions ran)")
        return True
    
    print(f"Found {high_risk_count} high-risk predictions from generate_predictions task")
    
    if high_risk_count > 10:
        print(f"HIGH-RISK ALERT: {high_risk_count} trips with >20min predicted delay")
        print("  This exceeds the threshold of 10 trips")
        print("  In production, this would trigger alerts to operations team")
        
        # Try to get details if table exists
        hook = PostgresHook(postgres_conn_id='transit_db')
        conn = hook.get_conn()
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'ml' 
                AND table_name = 'daily_predictions'
            )
        """)
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            cursor.execute("""
                SELECT route_id, COUNT(*) as trip_count, AVG(predicted_delay) as avg_delay
                FROM ml.daily_predictions
                WHERE prediction_date = CURRENT_DATE + INTERVAL '1 day'
                  AND predicted_delay > 20
                GROUP BY route_id
                ORDER BY trip_count DESC
                LIMIT 10
            """)
            
            routes = cursor.fetchall()
            if routes:
                print("\nTop Affected Routes:")
                print(f"{'Route':<10} {'Trips':<10} {'Avg Delay':<15}")
                for route_id, count, avg_delay in routes:
                    print(f"{route_id:<10} {count:<10} {avg_delay:.1f} min")
        
        cursor.close()
        conn.close()
    else:
        print(f"{high_risk_count} high-risk trips (below alert threshold of 10)")
    
    return True

# DAILY PIPELINE TASKS

# Data quality checks
task_quality_check = PythonOperator(
    task_id='data_quality_check',
    python_callable=check_data_quality,
    dag=dag_daily,
)

# Update features
task_update_features = PythonOperator(
    task_id='update_ml_features',
    python_callable=update_ml_features,
    dag=dag_daily,
)

# Generate predictions
task_predict = PythonOperator(
    task_id='generate_predictions',
    python_callable=generate_predictions,
    dag=dag_daily,
)

# Monitor performance
task_monitor = PythonOperator(
    task_id='monitor_predictions',
    python_callable=monitor_predictions,
    dag=dag_daily,
)

# Send alerts
task_alert = PythonOperator(
    task_id='send_high_risk_alert',
    python_callable=send_high_risk_alert,
    dag=dag_daily,
)

# Cleanup old predictions
task_cleanup = PostgresOperator(
    task_id='cleanup_old_predictions',
    postgres_conn_id='transit_db',
    sql="""
        -- Create predictions table if it doesn't exist
        CREATE TABLE IF NOT EXISTS ml.daily_predictions (
            prediction_id SERIAL PRIMARY KEY,
            trip_id VARCHAR(50),
            stop_id VARCHAR(50),
            route_id VARCHAR(50),
            predicted_delay INTEGER,
            prediction_date DATE,
            risk_level VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Clean up old predictions (won't fail if table is empty)
        DELETE FROM ml.daily_predictions
        WHERE created_at < CURRENT_DATE - INTERVAL '30 days';
        
        -- Log the result
        DO $$
        DECLARE
            deleted_count INTEGER;
        BEGIN
            GET DIAGNOSTICS deleted_count = ROW_COUNT;
            RAISE NOTICE 'Cleaned up % old predictions', deleted_count;
        END $$;
    """,
    dag=dag_daily,
)

# Daily pipeline dependencies
task_quality_check >> task_update_features >> task_predict >> task_monitor >> task_alert >> task_cleanup

# Training

# TRAIN MODEL (Weekly)
def train_model(**context):
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    # Load data
    hook = PostgresHook(postgres_conn_id='transit_db')
    query = """
        SELECT * FROM ml.train_features
        WHERE actual_arrival_timestamp >= CURRENT_DATE - INTERVAL '60 days'
        ORDER BY RANDOM()
        LIMIT 50000
    """
    df = hook.get_pandas_df(query)
    
    print(f"Loaded {len(df)} training records")
    
    # Prepare features (simplified preprocessing)
    y = df['delay_minutes'].values
    X = df.drop(['feature_id', 'trip_id', 'stop_id', 'route_id', 
                  'delay_minutes', 'delay_category', 'actual_arrival_timestamp',
                  'created_at'], axis=1, errors='ignore')
    
    # Basic preprocessing
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    X = X.fillna(0)
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    accuracy_10min = np.mean(np.abs(y_val - y_pred) <= 10) * 100
    
    print(f"\nModel Performance:")
    print(f"  MAE: {mae:.2f} minutes")
    print(f"  RMSE: {rmse:.2f} minutes")
    print(f"  Accuracy (10min): {accuracy_10min:.2f}%")
    
    # Save model
    import os
    os.makedirs('/opt/airflow/models', exist_ok=True)
    
    model_path = '/opt/airflow/models/xgboost_delay_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")
    
    # Push metrics to XCom
    context['task_instance'].xcom_push(key='model_mae', value=mae)
    context['task_instance'].xcom_push(key='model_accuracy', value=accuracy_10min)
    
    return {'mae': mae, 'rmse': rmse, 'accuracy': accuracy_10min}

# WEEKLY RETRAINING TASKS

# Task: Train model
task_train = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag_weekly,
)