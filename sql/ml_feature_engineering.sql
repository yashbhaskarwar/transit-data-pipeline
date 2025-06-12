DROP SCHEMA IF EXISTS ml CASCADE;
CREATE SCHEMA ml;

-- ML FEATURES TABLE 

CREATE TABLE ml.delay_features (
    feature_id SERIAL PRIMARY KEY,
    
    -- Identifiers
    trip_id VARCHAR(50) NOT NULL,
    stop_id VARCHAR(50) NOT NULL,
    route_id VARCHAR(50) NOT NULL,
    
    -- Target Variable
    delay_minutes INTEGER NOT NULL,
    delay_category VARCHAR(20),
    
    -- Temporal Features
    date DATE NOT NULL,
    day_of_week INTEGER NOT NULL,
    day_of_month INTEGER NOT NULL,
    hour_of_day INTEGER NOT NULL,
    minute_of_hour INTEGER NOT NULL,
    week_of_year INTEGER NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    is_holiday BOOLEAN DEFAULT FALSE,
    is_rush_hour BOOLEAN NOT NULL,
    month INTEGER NOT NULL,
    season VARCHAR(20) NOT NULL,
    
    -- Route Features
    route_type INTEGER NOT NULL,
    route_total_stops INTEGER,
    stop_sequence INTEGER,
    stops_remaining INTEGER,
    
    -- Weather Features
    temperature DECIMAL(5, 2),
    precipitation DECIMAL(5, 2),
    wind_speed DECIMAL(5, 2),
    weather_condition VARCHAR(50),
    weather_severity INTEGER,
    
    -- Historical Features 
    avg_delay_same_route_stop_7d DECIMAL(10, 2) DEFAULT 0,
    delay_count_same_route_stop_7d INTEGER DEFAULT 0,
    max_delay_same_route_stop_7d INTEGER DEFAULT 0,
    
    avg_delay_route_7d DECIMAL(10, 2) DEFAULT 0,
    stddev_delay_route_7d DECIMAL(10, 2) DEFAULT 0,
    
    avg_delay_stop_7d DECIMAL(10, 2) DEFAULT 0,
    
    avg_delay_same_hour_7d DECIMAL(10, 2) DEFAULT 0,
    avg_delay_same_dow_7d DECIMAL(10, 2) DEFAULT 0,
    avg_delay_same_weather_7d DECIMAL(10, 2) DEFAULT 0,
    
    -- Historical Features 
    avg_delay_same_route_stop_30d DECIMAL(10, 2) DEFAULT 0,
    avg_delay_route_30d DECIMAL(10, 2) DEFAULT 0,
    avg_delay_stop_30d DECIMAL(10, 2) DEFAULT 0,
    avg_delay_same_hour_30d DECIMAL(10, 2) DEFAULT 0,
    
    -- Trend Features
    delay_trend_7d DECIMAL(10, 2) DEFAULT 0,
    delay_volatility_7d DECIMAL(10, 2) DEFAULT 0,
    
    -- Cascade Features
    prev_stop_delay INTEGER DEFAULT 0,
    prev_stop_avg_delay_7d DECIMAL(10, 2) DEFAULT 0,
    
    -- Stop Features
    is_major_hub BOOLEAN,
    stop_area VARCHAR(50),
    
    -- Interaction Features
    rush_hour_delay_multiplier DECIMAL(5, 2) DEFAULT 1.0,
    weather_rush_hour_interaction INTEGER DEFAULT 0,
    weekend_weather_interaction INTEGER DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

