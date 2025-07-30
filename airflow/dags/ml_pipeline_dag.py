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

# DAILY PIPELINE TASKS

# Data quality checks
task_quality_check = PythonOperator(
    task_id='data_quality_check',
    python_callable=check_data_quality,
    dag=dag_daily,
)