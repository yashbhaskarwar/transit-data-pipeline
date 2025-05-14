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