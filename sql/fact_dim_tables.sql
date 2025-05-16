DROP SCHEMA IF EXISTS warehouse CASCADE;
CREATE SCHEMA warehouse;


-- DIMENSION TABLES

-- Date
CREATE TABLE warehouse.dim_date (
    date_key INTEGER PRIMARY KEY,
    full_date DATE NOT NULL UNIQUE,
    year INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    month INTEGER NOT NULL,
    month_name VARCHAR(20) NOT NULL,
    week_of_year INTEGER NOT NULL,
    day_of_month INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL,
    day_name VARCHAR(20) NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    is_holiday BOOLEAN NOT NULL,
    season VARCHAR(20) NOT NULL
);

-- Time of Day
CREATE TABLE warehouse.dim_time (
    time_key INTEGER PRIMARY KEY,
    hour INTEGER NOT NULL,
    minute INTEGER NOT NULL,
    time_of_day VARCHAR(20) NOT NULL,
    is_rush_hour BOOLEAN NOT NULL,
    rush_hour_period VARCHAR(20)
);

-- Stop 
CREATE TABLE warehouse.dim_stop (
    stop_key SERIAL PRIMARY KEY,
    stop_id VARCHAR(50) NOT NULL UNIQUE,
    stop_name VARCHAR(255) NOT NULL,
    stop_lat DECIMAL(10, 8) NOT NULL,
    stop_lon DECIMAL(11, 8) NOT NULL,
    platform_code VARCHAR(50),  
    location_type INTEGER DEFAULT 0,
    stop_area VARCHAR(100),
    is_major_hub BOOLEAN DEFAULT FALSE
);

-- Route
CREATE TABLE warehouse.dim_route (
    route_key SERIAL PRIMARY KEY,
    route_id VARCHAR(50) NOT NULL UNIQUE,
    route_short_name VARCHAR(50),
    route_long_name VARCHAR(255),
    route_type INTEGER NOT NULL,
    route_type_desc VARCHAR(50),
    route_color VARCHAR(6),
    route_sort_order INTEGER, 
    avg_trip_duration INTEGER
);

-- Trip
CREATE TABLE warehouse.dim_trip (
    trip_key SERIAL PRIMARY KEY,
    trip_id VARCHAR(50) NOT NULL UNIQUE,
    route_key INTEGER REFERENCES warehouse.dim_route(route_key),
    service_id VARCHAR(50),
    direction_id INTEGER,
    trip_headsign VARCHAR(255),
    total_stops INTEGER
);

-- Weather
CREATE TABLE warehouse.dim_weather (
    weather_key SERIAL PRIMARY KEY,
    weather_condition VARCHAR(50) NOT NULL,
    severity_level INTEGER NOT NULL,
    impact_category VARCHAR(50),
    description TEXT
);

-- FACT TABLE

-- Delay Events
CREATE TABLE warehouse.fact_delay_events (
    delay_event_key SERIAL PRIMARY KEY,
    
    -- Dimension Foreign Keys
    date_key INTEGER NOT NULL REFERENCES warehouse.dim_date(date_key),
    time_key INTEGER NOT NULL REFERENCES warehouse.dim_time(time_key),
    stop_key INTEGER NOT NULL REFERENCES warehouse.dim_stop(stop_key),
    route_key INTEGER NOT NULL REFERENCES warehouse.dim_route(route_key),
    trip_key INTEGER NOT NULL REFERENCES warehouse.dim_trip(trip_key),
    weather_key INTEGER NOT NULL REFERENCES warehouse.dim_weather(weather_key),
    
    -- Degenerate Dimensions
    trip_id VARCHAR(50) NOT NULL,
    stop_id VARCHAR(50) NOT NULL,
    stop_sequence INTEGER,
    
    -- Measures
    delay_minutes INTEGER NOT NULL,
    scheduled_arrival_seconds INTEGER,
    actual_arrival_timestamp TIMESTAMP NOT NULL,
    
    -- Derived Measures
    is_significant_delay BOOLEAN,
    is_severe_delay BOOLEAN,
    delay_category VARCHAR(20),
    
    -- Metadata
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- AGGREGATE FACT TABLES

-- Daily Aggregates by Route
CREATE TABLE warehouse.fact_daily_route_performance (
    date_key INTEGER NOT NULL REFERENCES warehouse.dim_date(date_key),
    route_key INTEGER NOT NULL REFERENCES warehouse.dim_route(route_key),
    
    -- Metrics
    total_trips INTEGER,
    total_delays INTEGER,
    total_delay_minutes INTEGER,
    avg_delay_minutes DECIMAL(10, 2),
    max_delay_minutes INTEGER,
    on_time_percentage DECIMAL(6, 2),  
    
    -- Delay Categories
    minor_delays INTEGER,
    moderate_delays INTEGER,
    severe_delays INTEGER,
    extreme_delays INTEGER,
    
    PRIMARY KEY (date_key, route_key)
);

-- Hourly Aggregates by Stop
CREATE TABLE warehouse.fact_hourly_stop_performance (
    date_key INTEGER NOT NULL REFERENCES warehouse.dim_date(date_key),
    time_key INTEGER NOT NULL REFERENCES warehouse.dim_time(time_key),
    stop_key INTEGER NOT NULL REFERENCES warehouse.dim_stop(stop_key),
    
    -- Metrics
    total_arrivals INTEGER,
    total_delays INTEGER,
    avg_delay_minutes DECIMAL(10, 2),
    delay_rate DECIMAL(5, 2),
    
    PRIMARY KEY (date_key, time_key, stop_key)
);