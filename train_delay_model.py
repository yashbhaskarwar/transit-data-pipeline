import psycopg2
import pandas as pd
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

# DATA LOADING
def load_training_data(conn):
    print("\nLoading training data...")
    
    query = """
        SELECT 
            -- Target
            delay_minutes,
            delay_category,
            
            -- Temporal features
            hour_of_day,
            day_of_week,
            day_of_month,
            month,
            week_of_year,
            is_weekend,
            is_holiday,
            is_rush_hour,
            season,
            
            -- Route features
            route_type,
            route_total_stops,
            stop_sequence,
            stops_remaining,
            
            -- Stop features
            is_major_hub,
            stop_area,
            
            -- Weather features
            weather_condition,
            weather_severity,
            COALESCE(temperature, 15.0) as temperature,
            COALESCE(precipitation, 0.0) as precipitation,
            COALESCE(wind_speed, 5.0) as wind_speed,
            -- COALESCE(visibility, 8.0) as visibility,
            
            -- Historical features
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
            
            -- Trend features
            COALESCE(delay_trend_7d, 0) as delay_trend_7d,
            COALESCE(delay_volatility_7d, 0) as delay_volatility_7d,
            
            -- Cascade features
            COALESCE(prev_stop_delay, 0) as prev_stop_delay,
            COALESCE(prev_stop_avg_delay_7d, 0) as prev_stop_avg_delay_7d,
            
            -- Interaction features
            rush_hour_delay_multiplier,
            weather_rush_hour_interaction,
            weekend_weather_interaction
            
        FROM ml.train_features
        WHERE delay_minutes IS NOT NULL
        ORDER BY RANDOM()  
        LIMIT 100000  -- remove LIMIT to train the full dataset
    """
    
    df = pd.read_sql(query, conn)
    print(f"  Loaded {len(df):,} training records")
    print(f"  Features: {df.shape[1] - 2}")  
    print(f"  Delay range: {df['delay_minutes'].min()}-{df['delay_minutes'].max()} minutes")
    print(f"  Avg delay: {df['delay_minutes'].mean():.2f} minutes")
    
    return df


def load_test_data(conn):
    print("\nLoading test data...")
    
    query = """
        SELECT 
            -- Target
            delay_minutes,
            delay_category,
            
            -- Temporal features
            hour_of_day,
            day_of_week,
            day_of_month,
            month,
            week_of_year,
            is_weekend,
            is_holiday,
            is_rush_hour,
            season,
            
            -- Route features
            route_type,
            route_total_stops,
            stop_sequence,
            stops_remaining,
            
            -- Stop features
            is_major_hub,
            stop_area,
            
            -- Weather features
            weather_condition,
            weather_severity,
            COALESCE(temperature, 15.0) as temperature,
            COALESCE(precipitation, 0.0) as precipitation,
            COALESCE(wind_speed, 5.0) as wind_speed,
            -- COALESCE(visibility, 8.0) as visibility,
            
            -- Historical features
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
            
            -- Trend features
            COALESCE(delay_trend_7d, 0) as delay_trend_7d,
            COALESCE(delay_volatility_7d, 0) as delay_volatility_7d,
            
            -- Cascade features
            COALESCE(prev_stop_delay, 0) as prev_stop_delay,
            COALESCE(prev_stop_avg_delay_7d, 0) as prev_stop_avg_delay_7d,
            
            -- Interaction features
            rush_hour_delay_multiplier,
            weather_rush_hour_interaction,
            weekend_weather_interaction
            
        FROM ml.test_features
        WHERE delay_minutes IS NOT NULL
    """
    
    df = pd.read_sql(query, conn)
    print(f"Loaded {len(df):,} test records")
    
    return df