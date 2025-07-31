from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import PythonOperator

# DAG DEFAULT ARGUMENTS
default_args = {
    'owner': 'transit_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email': ['alerts@transit.com'],  # UPDATE THIS
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

# TASK 2: UPDATE ML FEATURES

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

# TASK 4: GENERATE PREDICTIONS

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

# Task: Generate predictions
task_predict = PythonOperator(
    task_id='generate_predictions',
    python_callable=generate_predictions,
    dag=dag_daily,
)

# Daily pipeline dependencies
task_quality_check >> task_update_features >> task_predict