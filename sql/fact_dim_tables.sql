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

-- POPULATE DIMENSION TABLES

-- Populate dim_date
INSERT INTO warehouse.dim_date (
    date_key, full_date, year, quarter, month, month_name,
    week_of_year, day_of_month, day_of_week, day_name,
    is_weekend, is_holiday, season
)
SELECT 
    TO_CHAR(d, 'YYYYMMDD')::INTEGER as date_key,
    d as full_date,
    EXTRACT(YEAR FROM d)::INTEGER as year,
    EXTRACT(QUARTER FROM d)::INTEGER as quarter,
    EXTRACT(MONTH FROM d)::INTEGER as month,
    TO_CHAR(d, 'Month') as month_name,
    EXTRACT(WEEK FROM d)::INTEGER as week_of_year,
    EXTRACT(DAY FROM d)::INTEGER as day_of_month,
    EXTRACT(DOW FROM d)::INTEGER as day_of_week,
    TO_CHAR(d, 'Day') as day_name,
    CASE WHEN EXTRACT(DOW FROM d) IN (0, 6) THEN TRUE ELSE FALSE END as is_weekend,
    FALSE as is_holiday,
    CASE 
        WHEN EXTRACT(MONTH FROM d) IN (12, 1, 2) THEN 'Winter'
        WHEN EXTRACT(MONTH FROM d) IN (3, 4, 5) THEN 'Spring'
        WHEN EXTRACT(MONTH FROM d) IN (6, 7, 8) THEN 'Summer'
        ELSE 'Fall'
    END as season
FROM generate_series(
    (SELECT MIN(DATE(actual_arrival)) FROM operational.delay_events),
    (SELECT MAX(DATE(actual_arrival)) FROM operational.delay_events),
    '1 day'::interval
) d;

-- Update holidays based on actual data
UPDATE warehouse.dim_date
SET is_holiday = TRUE
WHERE date_key IN (
    SELECT DISTINCT TO_CHAR(DATE(actual_arrival), 'YYYYMMDD')::INTEGER
    FROM operational.delay_events
    WHERE is_holiday = TRUE
);

-- Populate dim_time
INSERT INTO warehouse.dim_time (time_key, hour, minute, time_of_day, is_rush_hour, rush_hour_period)
SELECT 
    hour * 100 + minute as time_key,
    hour,
    minute,
    CASE 
        WHEN hour BETWEEN 5 AND 11 THEN 'Morning'
        WHEN hour BETWEEN 12 AND 16 THEN 'Afternoon'
        WHEN hour BETWEEN 17 AND 20 THEN 'Evening'
        ELSE 'Night'
    END as time_of_day,
    CASE 
        WHEN hour BETWEEN 7 AND 9 OR hour BETWEEN 17 AND 19 THEN TRUE
        ELSE FALSE
    END as is_rush_hour,
    CASE 
        WHEN hour BETWEEN 7 AND 9 THEN 'Morning Rush'
        WHEN hour BETWEEN 17 AND 19 THEN 'Evening Rush'
        ELSE 'Off Peak'
    END as rush_hour_period
FROM generate_series(0, 23) hour
CROSS JOIN generate_series(0, 59) minute
WHERE minute IN (0, 15, 30, 45);

-- Populate dim_stop
INSERT INTO warehouse.dim_stop (
    stop_id, stop_name, stop_lat, stop_lon, platform_code,
    location_type, stop_area, is_major_hub
)
SELECT 
    s.stop_id,
    s.stop_name,
    s.stop_lat,
    s.stop_lon,
    s.platform_code, 
    0 as location_type,  -- Default value is set to 0 as field is not present in GTFS dataset
    -- Derive stop area based on location
    CASE 
        WHEN s.stop_lat > (SELECT AVG(stop_lat) FROM operational.stops) THEN 'North'
        WHEN s.stop_lat < (SELECT AVG(stop_lat) FROM operational.stops) THEN 'South'
        ELSE 'Central'
    END as stop_area,
    CASE WHEN (
        SELECT COUNT(DISTINCT t.route_id)
        FROM operational.trips t
        INNER JOIN operational.stop_times st ON t.trip_id = st.trip_id
        WHERE st.stop_id = s.stop_id
    ) >= 5 THEN TRUE ELSE FALSE END as is_major_hub
FROM operational.stops s;

-- Populate dim_route 
INSERT INTO warehouse.dim_route (
    route_id, route_short_name, route_long_name, route_type, 
    route_type_desc, route_color, route_sort_order, avg_trip_duration
)
SELECT 
    r.route_id,
    r.route_short_name,
    r.route_long_name,
    r.route_type,
    CASE r.route_type
        WHEN 0 THEN 'Tram/Light Rail'
        WHEN 1 THEN 'Subway/Metro'
        WHEN 2 THEN 'Rail'
        WHEN 3 THEN 'Bus'
        WHEN 4 THEN 'Ferry'
        WHEN 5 THEN 'Cable Car'
        WHEN 6 THEN 'Gondola'
        WHEN 7 THEN 'Funicular'
        ELSE 'Other'
    END as route_type_desc,
    r.route_color,
    r.route_sort_order, 
    (SELECT COALESCE(AVG(duration_seconds), 0)::INTEGER / 60
     FROM (
         SELECT 
             st.trip_id,
             MAX(EXTRACT(EPOCH FROM st.arrival_time)) - MIN(EXTRACT(EPOCH FROM st.arrival_time)) as duration_seconds
         FROM operational.trips t
         INNER JOIN operational.stop_times st ON t.trip_id = st.trip_id
         WHERE t.route_id = r.route_id
           AND st.arrival_time IS NOT NULL
         GROUP BY st.trip_id
         HAVING MAX(EXTRACT(EPOCH FROM st.arrival_time)) - MIN(EXTRACT(EPOCH FROM st.arrival_time)) > 0
     ) trip_durations
    ) as avg_trip_duration
FROM operational.routes r;

-- Populate dim_trip
INSERT INTO warehouse.dim_trip (
    trip_id, route_key, service_id, direction_id, 
    trip_headsign, total_stops
)
SELECT 
    t.trip_id,
    dr.route_key,
    t.service_id,
    t.direction_id,
    t.trip_headsign,
    (SELECT COUNT(*) FROM operational.stop_times WHERE trip_id = t.trip_id) as total_stops
FROM operational.trips t
INNER JOIN warehouse.dim_route dr ON t.route_id = dr.route_id;

-- Populate dim_weather
INSERT INTO warehouse.dim_weather (weather_condition, severity_level, impact_category, description)
VALUES
    ('clear', 1, 'Low Impact', 'Clear skies, minimal transit impact'),
    ('partly_cloudy', 1, 'Low Impact', 'Partly cloudy, minimal transit impact'),
    ('cloudy', 1, 'Low Impact', 'Overcast, minimal transit impact'),
    ('rainy', 2, 'Medium Impact', 'Rain, moderate delays expected'),
    ('heavy_rain', 3, 'High Impact', 'Heavy rain, significant delays expected'),
    ('snow', 3, 'High Impact', 'Snow conditions, major delays expected'),
    ('fog', 2, 'Medium Impact', 'Foggy conditions, moderate delays'),
    ('windy', 2, 'Medium Impact', 'High winds, moderate delays possible');

-- POPULATE FACT TABLE

INSERT INTO warehouse.fact_delay_events (
    date_key, time_key, stop_key, route_key, trip_key, weather_key,
    trip_id, stop_id, stop_sequence,
    delay_minutes, scheduled_arrival_seconds, actual_arrival_timestamp,
    is_significant_delay, is_severe_delay, delay_category
)
SELECT 
    TO_CHAR(DATE(de.actual_arrival), 'YYYYMMDD')::INTEGER as date_key,
    (EXTRACT(HOUR FROM de.actual_arrival)::INTEGER * 100 + 
     (EXTRACT(MINUTE FROM de.actual_arrival)::INTEGER / 15) * 15) as time_key,
    ds.stop_key,
    dr.route_key,
    dt.trip_key,
    dw.weather_key,
    de.trip_id,
    de.stop_id,
    (SELECT stop_sequence FROM operational.stop_times 
     WHERE trip_id = de.trip_id AND stop_id = de.stop_id LIMIT 1) as stop_sequence,
    de.delay_minutes,
    EXTRACT(EPOCH FROM de.scheduled_arrival)::INTEGER as scheduled_arrival_seconds,
    de.actual_arrival,
    CASE WHEN de.delay_minutes > 10 THEN TRUE ELSE FALSE END as is_significant_delay,
    CASE WHEN de.delay_minutes > 30 THEN TRUE ELSE FALSE END as is_severe_delay,
    CASE 
        WHEN de.delay_minutes BETWEEN 1 AND 5 THEN 'Minor'
        WHEN de.delay_minutes BETWEEN 6 AND 15 THEN 'Moderate'
        WHEN de.delay_minutes BETWEEN 16 AND 30 THEN 'Severe'
        ELSE 'Extreme'
    END as delay_category
FROM operational.delay_events de
INNER JOIN warehouse.dim_stop ds ON de.stop_id = ds.stop_id
INNER JOIN warehouse.dim_trip dt ON de.trip_id = dt.trip_id
INNER JOIN warehouse.dim_route dr ON dt.route_key = dr.route_key
INNER JOIN warehouse.dim_weather dw ON de.weather_condition = dw.weather_condition;

-- POPULATE AGGREGATE FACT TABLES

-- Daily Route Performance
INSERT INTO warehouse.fact_daily_route_performance
SELECT 
    fde.date_key,
    fde.route_key,
    COUNT(DISTINCT fde.trip_id) as total_trips,
    COUNT(*) as total_delays,
    SUM(fde.delay_minutes) as total_delay_minutes,
    AVG(fde.delay_minutes)::DECIMAL(10,2) as avg_delay_minutes,
    MAX(fde.delay_minutes) as max_delay_minutes,
    LEAST(100.0, GREATEST(0.0, 
        100.0 - (COUNT(*)::DECIMAL / GREATEST(COUNT(DISTINCT fde.trip_id), 1) * 100)
    ))::DECIMAL(6,2) as on_time_percentage,
    SUM(CASE WHEN fde.delay_category = 'Minor' THEN 1 ELSE 0 END) as minor_delays,
    SUM(CASE WHEN fde.delay_category = 'Moderate' THEN 1 ELSE 0 END) as moderate_delays,
    SUM(CASE WHEN fde.delay_category = 'Severe' THEN 1 ELSE 0 END) as severe_delays,
    SUM(CASE WHEN fde.delay_category = 'Extreme' THEN 1 ELSE 0 END) as extreme_delays
FROM warehouse.fact_delay_events fde
GROUP BY fde.date_key, fde.route_key;

-- Hourly Stop Performance
INSERT INTO warehouse.fact_hourly_stop_performance
SELECT 
    fde.date_key,
    fde.time_key,
    fde.stop_key,
    COUNT(*) as total_arrivals,
    COUNT(*) as total_delays,
    AVG(fde.delay_minutes)::DECIMAL(10,2) as avg_delay_minutes,
    100.0::DECIMAL(5,2) as delay_rate
FROM warehouse.fact_delay_events fde
GROUP BY fde.date_key, fde.time_key, fde.stop_key;

-- CREATE INDEXES FOR PERFORMANCE

-- Fact table indexes
CREATE INDEX idx_fact_delay_date ON warehouse.fact_delay_events(date_key);
CREATE INDEX idx_fact_delay_time ON warehouse.fact_delay_events(time_key);
CREATE INDEX idx_fact_delay_stop ON warehouse.fact_delay_events(stop_key);
CREATE INDEX idx_fact_delay_route ON warehouse.fact_delay_events(route_key);
CREATE INDEX idx_fact_delay_trip ON warehouse.fact_delay_events(trip_key);
CREATE INDEX idx_fact_delay_weather ON warehouse.fact_delay_events(weather_key);
CREATE INDEX idx_fact_delay_category ON warehouse.fact_delay_events(delay_category);
CREATE INDEX idx_fact_delay_significant ON warehouse.fact_delay_events(is_significant_delay);

-- Dimension indexes
CREATE INDEX idx_dim_date_full ON warehouse.dim_date(full_date);
CREATE INDEX idx_dim_date_dow ON warehouse.dim_date(day_of_week);
CREATE INDEX idx_dim_date_weekend ON warehouse.dim_date(is_weekend);
CREATE INDEX idx_dim_time_rush ON warehouse.dim_time(is_rush_hour);
CREATE INDEX idx_dim_stop_hub ON warehouse.dim_stop(is_major_hub);
CREATE INDEX idx_dim_route_type ON warehouse.dim_route(route_type);
