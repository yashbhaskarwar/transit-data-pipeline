import psycopg2
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

# FEATURE ENGINEERING

def preprocess_features(df_train, df_test=None):
    print("\n[Preprocessing features...")
    
    # Separate target from features
    if MODEL_TYPE == 'regression':
        y_train = df_train['delay_minutes'].values
        y_test = df_test['delay_minutes'].values if df_test is not None else None
    else:
        y_train = df_train['delay_category'].values
        y_test = df_test['delay_category'].values if df_test is not None else None
    
    # Drop target columns
    X_train = df_train.drop(['delay_minutes', 'delay_category'], axis=1)
    X_test = df_test.drop(['delay_minutes', 'delay_category'], axis=1) if df_test is not None else None
    
    # Identify categorical and numerical columns
    categorical_cols = ['season', 'stop_area', 'weather_condition']
    numerical_cols = [col for col in X_train.columns if col not in categorical_cols]
    boolean_cols = [col for col in X_train.columns if X_train[col].dtype == 'bool']
    
    # Convert boolean to int
    for col in boolean_cols:
        X_train[col] = X_train[col].astype(int)
        if X_test is not None:
            X_test[col] = X_test[col].astype(int)
    
    # Encode categorical features
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        encoders[col] = le
        
        if X_test is not None:
            # Handle unseen categories in test set
            X_test[col] = X_test[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    
    if X_test is not None:
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print(f"  Preprocessed features")
    print(f"  Categorical: {len(categorical_cols)}")
    print(f"  Numerical: {len(numerical_cols)}")
    print(f"  Boolean: {len(boolean_cols)}")
    
    if df_test is not None:
        return X_train, X_test, y_train, y_test, encoders, scaler
    else:
        return X_train, y_train, encoders, scaler
