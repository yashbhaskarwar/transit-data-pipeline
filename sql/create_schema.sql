-- Create database 
CREATE DATABASE transit_delay_optimization;

CREATE SCHEMA staging;
CREATE SCHEMA operational;

-- stops.txt 
CREATE TABLE staging.stops (
    stop_id VARCHAR(50),
    stop_code VARCHAR(50),
    stop_name VARCHAR(255),
    stop_lat DECIMAL(10, 8),
    stop_lon DECIMAL(11, 8),
    wheelchair_boarding INTEGER,
    platform_code VARCHAR(50),
    stop_url TEXT
);

-- routes.txt 
CREATE TABLE staging.routes (
    route_id VARCHAR(50),
    agency_id VARCHAR(50),
    route_short_name VARCHAR(50),
    route_long_name VARCHAR(255),
    route_url TEXT,
    route_desc TEXT,
    route_type INTEGER,
    route_color VARCHAR(6),
    route_text_color VARCHAR(6),
    route_sort_order INTEGER
);

-- trips.txt 
CREATE TABLE staging.trips (
    route_id VARCHAR(50),
    service_id VARCHAR(50),
    trip_id VARCHAR(50),
    trip_headsign VARCHAR(255),
    direction_id INTEGER,
    block_id VARCHAR(50),
    shape_id VARCHAR(50),
    wheelchair_accessible INTEGER,
    bikes_allowed INTEGER
);

-- stop_times.txt 
CREATE TABLE staging.stop_times (
    trip_id VARCHAR(50),
    arrival_time VARCHAR(8),
    departure_time VARCHAR(8),
    stop_id VARCHAR(50),
    stop_sequence INTEGER,
    stop_headsign VARCHAR(255),
    shape_dist_traveled DECIMAL(10, 2),
    timepoint INTEGER
);

-- calendar.txt 
CREATE TABLE staging.calendar (
    service_id VARCHAR(50),
    monday INTEGER,
    tuesday INTEGER,
    wednesday INTEGER,
    thursday INTEGER,
    friday INTEGER,
    saturday INTEGER,
    sunday INTEGER,
    start_date DATE,
    end_date DATE
);

-- OPERATIONAL LAYER: Cleaned and Optimized Tables

-- Operational: Stops 
CREATE TABLE operational.stops (
    stop_id VARCHAR(50) PRIMARY KEY,
    stop_code VARCHAR(50),
    stop_name VARCHAR(255) NOT NULL,
    stop_lat DECIMAL(10, 8) NOT NULL,
    stop_lon DECIMAL(11, 8) NOT NULL,
    wheelchair_boarding INTEGER DEFAULT 0,
    platform_code VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Operational: Routes
CREATE TABLE operational.routes (
    route_id VARCHAR(50) PRIMARY KEY,
    agency_id VARCHAR(50),
    route_short_name VARCHAR(50),
    route_long_name VARCHAR(255),
    route_type INTEGER NOT NULL,
    route_color VARCHAR(6) DEFAULT 'FFFFFF',
    route_text_color VARCHAR(6) DEFAULT '000000',
    route_sort_order INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Operational: Trips
CREATE TABLE operational.trips (
    trip_id VARCHAR(50) PRIMARY KEY,
    route_id VARCHAR(50) NOT NULL,
    service_id VARCHAR(50) NOT NULL,
    trip_headsign VARCHAR(255),
    direction_id INTEGER DEFAULT 0,
    block_id VARCHAR(50),
    shape_id VARCHAR(50),
    wheelchair_accessible INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (route_id) REFERENCES operational.routes(route_id)
);

-- Operational: Stop Times 	
CREATE TABLE operational.stop_times (
    id SERIAL PRIMARY KEY,
    trip_id VARCHAR(50) NOT NULL,
    stop_id VARCHAR(50) NOT NULL,
    stop_sequence INTEGER NOT NULL,
    arrival_time INTERVAL,
    departure_time INTERVAL,
    stop_headsign VARCHAR(255),
    shape_dist_traveled DECIMAL(10, 2),
    timepoint INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trip_id) REFERENCES operational.trips(trip_id),
    FOREIGN KEY (stop_id) REFERENCES operational.stops(stop_id),
    UNIQUE (trip_id, stop_sequence)
);

-- Operational: Calendar
CREATE TABLE operational.calendar (
    service_id VARCHAR(50) PRIMARY KEY,
    monday BOOLEAN NOT NULL,
    tuesday BOOLEAN NOT NULL,
    wednesday BOOLEAN NOT NULL,
    thursday BOOLEAN NOT NULL,
    friday BOOLEAN NOT NULL,
    saturday BOOLEAN NOT NULL,
    sunday BOOLEAN NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ML FEATURE TABLES

-- Real-time delay tracking (ML training)
CREATE TABLE operational.delay_events (
    id SERIAL PRIMARY KEY,
    trip_id VARCHAR(50) NOT NULL,
    stop_id VARCHAR(50) NOT NULL,
    scheduled_arrival INTERVAL,
    actual_arrival TIMESTAMP,
    delay_minutes INTEGER,
    weather_condition VARCHAR(50),
    day_of_week INTEGER,
    is_holiday BOOLEAN DEFAULT FALSE,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trip_id) REFERENCES operational.trips(trip_id),
    FOREIGN KEY (stop_id) REFERENCES operational.stops(stop_id)
);

-- Weather data integration (ML features)
CREATE TABLE operational.weather_data (
    id SERIAL PRIMARY KEY,
    recorded_at TIMESTAMP NOT NULL,
    temperature DECIMAL(5, 2),
    precipitation DECIMAL(5, 2),
    wind_speed DECIMAL(5, 2),
    visibility DECIMAL(5, 2),
    weather_condition VARCHAR(50),
    UNIQUE (recorded_at)
);

-- INDEXES FOR PERFORMANCE

-- Staging indexes 
CREATE INDEX idx_staging_stops_id ON staging.stops(stop_id);
CREATE INDEX idx_staging_routes_id ON staging.routes(route_id);
CREATE INDEX idx_staging_trips_id ON staging.trips(trip_id);
CREATE INDEX idx_staging_stop_times_trip ON staging.stop_times(trip_id);

-- Operational indexes
CREATE INDEX idx_ops_stops_location ON operational.stops(stop_lat, stop_lon);
CREATE INDEX idx_ops_trips_route ON operational.trips(route_id);
CREATE INDEX idx_ops_trips_service ON operational.trips(service_id);
CREATE INDEX idx_ops_stop_times_trip ON operational.stop_times(trip_id);
CREATE INDEX idx_ops_stop_times_stop ON operational.stop_times(stop_id);
CREATE INDEX idx_ops_stop_times_sequence ON operational.stop_times(trip_id, stop_sequence);
CREATE INDEX idx_ops_calendar_dates ON operational.calendar(start_date, end_date);
CREATE INDEX idx_ops_delay_events_trip ON operational.delay_events(trip_id);
CREATE INDEX idx_ops_delay_events_stop ON operational.delay_events(stop_id);
CREATE INDEX idx_ops_delay_events_time ON operational.delay_events(recorded_at);

-- Convert GTFS time format 
CREATE OR REPLACE FUNCTION staging.gtfs_time_to_interval(time_str VARCHAR)
RETURNS INTERVAL AS $$
DECLARE
    parts TEXT[];
    hours INTEGER;
    minutes INTEGER;
    seconds INTEGER;
BEGIN
    IF time_str IS NULL OR time_str = '' THEN
        RETURN NULL;
    END IF;
    
    parts := string_to_array(time_str, ':');
    hours := parts[1]::INTEGER;
    minutes := parts[2]::INTEGER;
    seconds := parts[3]::INTEGER;
    
    RETURN make_interval(hours => hours, mins => minutes, secs => seconds);
EXCEPTION
    WHEN OTHERS THEN
        RETURN NULL;
END;
$$ LANGUAGE plpgsql;