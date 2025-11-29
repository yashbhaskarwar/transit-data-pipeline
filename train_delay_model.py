import psycopg2
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, mean_squared_error,
    mean_absolute_error, r2_score
)
import pickle
import json
from datetime import datetime
import time
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

# MODEL TRAINING
def train_xgboost_model(X_train, y_train):
    print("\nTraining XGBoost model...")
    
    if MODEL_TYPE == 'regression':
        # XGBoost Regressor
        print("  Model type: Regression")
        
        # Initial model for quick validation
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=MODEL_CONFIG['random_state'],
            n_jobs=-1
        )
        
        # Hyperparameter grid (reduced for speed)
        param_grid = {
            'max_depth': [6, 8],
            'learning_rate': [0.1],
            'n_estimators': [200],
            'min_child_weight': [1, 3],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
        
        print("  Running hyperparameter tuning...")
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        print(f"  Model trained")
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  CV Score (MSE): {-grid_search.best_score_:.2f}")
        
    else:
        # XGBoost Classifier
        print("  Model type: Classification (predicting delay category)")
        
        base_model = xgb.XGBClassifier(
            objective='multi:softmax',
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=MODEL_CONFIG['random_state'],
            n_jobs=-1
        )
        
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [150, 200, 250]
        }
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        print(f"  Model trained")
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  CV Accuracy: {grid_search.best_score_:.4f}")
    
    return best_model

# MODEL EVALUATION
def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model...")
    
    y_pred = model.predict(X_test)
    
    metrics = {}
    
    if MODEL_TYPE == 'regression':
        # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate accuracy within tolerance
        tolerance_5min = np.mean(np.abs(y_test - y_pred) <= 5)
        tolerance_10min = np.mean(np.abs(y_test - y_pred) <= 10)
        
        metrics = {
            'model_type': 'regression',
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'accuracy_within_5min': float(tolerance_5min),
            'accuracy_within_10min': float(tolerance_10min)
        }
        
        print(f"  Regression Metrics:")
        print(f"  RMSE: {rmse:.2f} minutes")
        print(f"  MAE: {mae:.2f} minutes")
        print(f"  R² Score: {r2:.4f}")
        print(f"  Accuracy within 5 min: {tolerance_5min*100:.2f}%")
        print(f"  Accuracy within 10 min: {tolerance_10min*100:.2f}%")
        
    else:
        # Classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'model_type': 'classification',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        print(f"  Classification Metrics:")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    return metrics, y_pred

# FEATURE IMPORTANCE
def analyze_feature_importance(model, feature_names):
    print("\nAnalyzing feature importance...")
    
    importance_scores = model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    return feature_importance

# MODEL PERSISTENCE

def save_model_artifacts(model, scaler, encoders, metrics, feature_importance):
    print("\nSaving model artifacts...")
    
    import os
    os.makedirs('models', exist_ok=True)
    
    # Save model
    with open(MODEL_CONFIG['model_path'], 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_CONFIG['model_path']}")
    
    # Save scaler
    with open(MODEL_CONFIG['scaler_path'], 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {MODEL_CONFIG['scaler_path']}")
    
    # Save encoders
    with open(MODEL_CONFIG['encoders_path'], 'wb') as f:
        pickle.dump(encoders, f)
    print(f"Encoders saved to {MODEL_CONFIG['encoders_path']}")
    
    # Save metrics
    metrics['training_date'] = datetime.now().isoformat()
    with open(MODEL_CONFIG['metrics_path'], 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {MODEL_CONFIG['metrics_path']}")
    
    # Save feature importance
    feature_importance.to_csv(MODEL_CONFIG['feature_importance_path'], index=False)
    print(f"Feature importance saved to {MODEL_CONFIG['feature_importance_path']}")

# Save metrics to DB
def save_metrics_to_db(metrics_dict, db_config):
    try:
        conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password']
        )
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml.model_metrics (
                metric_id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) DEFAULT 'xgboost_delay_model',
                model_version VARCHAR(50),
                training_accuracy DECIMAL(5,2),
                training_mae DECIMAL(10,2),
                training_samples INTEGER,
                test_accuracy DECIMAL(5,2),
                test_mae DECIMAL(10,2),
                test_samples INTEGER,
                num_features INTEGER,
                top_feature VARCHAR(100),
                top_feature_importance DECIMAL(5,2),
                trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                training_duration_seconds INTEGER,
                notes TEXT
            )
        """)
        
        # Generate version
        version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Insert metrics
        cursor.execute("""
            INSERT INTO ml.model_metrics (
                model_version, training_accuracy, training_mae, training_samples,
                test_accuracy, test_mae, test_samples, num_features,
                top_feature, top_feature_importance, training_duration_seconds, notes
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            version,
            metrics_dict['training_accuracy'],
            metrics_dict['training_mae'],
            metrics_dict['training_samples'],
            metrics_dict['test_accuracy'],
            metrics_dict['test_mae'],
            metrics_dict['test_samples'],
            metrics_dict['num_features'],
            metrics_dict['top_feature'],
            metrics_dict['top_feature_importance'],
            metrics_dict['training_duration'],
            metrics_dict.get('notes', 'Automated training run')
        ))
        
        conn.commit()
        print(f"Metrics saved with version: {version}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error saving metrics: {e}")

# MAIN EXECUTION
def main():
    start_time = time.time()
    # Connect to database
    conn = get_db_connection()
    
    try:
        # Load data
        df_train = load_training_data(conn)
        df_test = load_test_data(conn)
        
        # Preprocess
        X_train, X_test, y_train, y_test, encoders, scaler = preprocess_features(df_train, df_test)
        
        # Train model
        model = train_xgboost_model(X_train, y_train)
        
        # Evaluate model
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        
        # Feature importance
        feature_importance = analyze_feature_importance(model, X_train.columns.tolist())
        
        # Calculate training duration
        training_time = int(time.time() - start_time)

        # Metrics
        metrics.update({
            'training_accuracy': metrics.get('accuracy_within_10min', 0) * 100,
            'training_mae': metrics.get('mae', 0),
            'training_samples': len(X_train),
            
            'test_accuracy': metrics.get('accuracy_within_10min', 0) * 100,
            'test_mae': metrics.get('mae', 0),
            'test_samples': len(X_test),
            
            # Additional fields
            'num_features': len(X_train.columns),
            'top_feature': feature_importance.iloc[0]['feature'] if not feature_importance.empty else 'unknown',
            'top_feature_importance': float(feature_importance.iloc[0]['importance']) if not feature_importance.empty else 0.0,
            'training_duration': training_time,
})

        # Save artifacts
        save_model_artifacts(model, scaler, encoders, metrics, feature_importance)

        # Save to DB
        save_metrics_to_db(metrics, DB_CONFIG)
        
        if MODEL_TYPE == 'regression':
            if metrics['accuracy_within_10min'] >= 0.85:
                print(f"SUCCESS: Model achieves {metrics['accuracy_within_10min']*100:.2f}%")
            else:
                print(f"Model accuracy: {metrics['accuracy_within_10min']*100:.2f}%")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    
    finally:
        conn.close()

if __name__ == "__main__":
    main()
