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

-- BASIC FEATURES
INSERT INTO ml.delay_features (
    trip_id, stop_id, route_id,
    delay_minutes, delay_category,
    date, day_of_week, day_of_month, hour_of_day, minute_of_hour,
    week_of_year, is_weekend, is_holiday, is_rush_hour, month, season,
    route_type, route_total_stops, stop_sequence, stops_remaining,
    temperature, precipitation, wind_speed,
    weather_condition, weather_severity,
    is_major_hub, stop_area,
    rush_hour_delay_multiplier, weather_rush_hour_interaction, weekend_weather_interaction
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
    END as delay_category,
    
    -- Temporal Features
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
    
    -- Route Features
    r.route_type,
    (SELECT COUNT(*) FROM operational.stop_times WHERE trip_id = de.trip_id),
    COALESCE(st.stop_sequence, 0),
    GREATEST(0, (SELECT MAX(stop_sequence) FROM operational.stop_times WHERE trip_id = de.trip_id) - COALESCE(st.stop_sequence, 0)),
    
    -- Weather Features (from weather_data table)
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
        CASE 
            WHEN de.weather_condition IN ('rainy', 'heavy_rain') THEN 5.0
            WHEN de.weather_condition = 'snow' THEN 3.0
            ELSE 0.0
        END
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
    
    -- Stop Features
    COALESCE(s.is_major_hub, FALSE),
    COALESCE(s.stop_area, 'Unknown'),
    
    -- Interaction Features
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
LEFT JOIN warehouse.dim_stop s ON de.stop_id = s.stop_id;

-- PRE-COMPUTE HISTORICAL AGGREGATIONS
-- 7-day route-stop averages
CREATE TEMP TABLE hist_route_stop_7d AS
SELECT 
    de.trip_id,
    de.stop_id,
    AVG(de.delay_minutes) as avg_delay,
    COUNT(*) as delay_count,
    MAX(de.delay_minutes) as max_delay
FROM operational.delay_events de
WHERE de.actual_arrival >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY de.trip_id, de.stop_id;

CREATE INDEX idx_hist_rs7d ON hist_route_stop_7d(trip_id, stop_id);

-- 30-day route-stop averages
CREATE TEMP TABLE hist_route_stop_30d AS
SELECT 
    de.trip_id,
    de.stop_id,
    AVG(de.delay_minutes) as avg_delay
FROM operational.delay_events de
WHERE de.actual_arrival >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY de.trip_id, de.stop_id;

CREATE INDEX idx_hist_rs30d ON hist_route_stop_30d(trip_id, stop_id);

-- 7-day route averages
CREATE TEMP TABLE hist_route_7d AS
SELECT 
    t.route_id,
    AVG(de.delay_minutes) as avg_delay,
    STDDEV(de.delay_minutes) as stddev_delay
FROM operational.delay_events de
INNER JOIN operational.trips t ON de.trip_id = t.trip_id
WHERE de.actual_arrival >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY t.route_id;

CREATE INDEX idx_hist_r7d ON hist_route_7d(route_id);

-- 30-day route averages
CREATE TEMP TABLE hist_route_30d AS
SELECT 
    t.route_id,
    AVG(de.delay_minutes) as avg_delay
FROM operational.delay_events de
INNER JOIN operational.trips t ON de.trip_id = t.trip_id
WHERE de.actual_arrival >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY t.route_id;

CREATE INDEX idx_hist_r30d ON hist_route_30d(route_id);

-- 7-day stop averages
CREATE TEMP TABLE hist_stop_7d AS
SELECT 
    stop_id,
    AVG(delay_minutes) as avg_delay
FROM operational.delay_events
WHERE actual_arrival >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY stop_id;

CREATE INDEX idx_hist_s7d ON hist_stop_7d(stop_id);

-- 30-day stop averages
CREATE TEMP TABLE hist_stop_30d AS
SELECT 
    stop_id,
    AVG(delay_minutes) as avg_delay
FROM operational.delay_events
WHERE actual_arrival >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY stop_id;

CREATE INDEX idx_hist_s30d ON hist_stop_30d(stop_id);

-- Hour-of-day averages (7-day)
CREATE TEMP TABLE hist_hour_7d AS
SELECT 
    EXTRACT(HOUR FROM actual_arrival)::INTEGER as hour,
    AVG(delay_minutes) as avg_delay
FROM operational.delay_events
WHERE actual_arrival >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY EXTRACT(HOUR FROM actual_arrival)::INTEGER;

CREATE INDEX idx_hist_h7d ON hist_hour_7d(hour);

-- Hour-of-day averages (30-day)
CREATE TEMP TABLE hist_hour_30d AS
SELECT 
    EXTRACT(HOUR FROM actual_arrival)::INTEGER as hour,
    AVG(delay_minutes) as avg_delay
FROM operational.delay_events
WHERE actual_arrival >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY EXTRACT(HOUR FROM actual_arrival)::INTEGER;

CREATE INDEX idx_hist_h30d ON hist_hour_30d(hour);

-- Day-of-week averages (7-day)
CREATE TEMP TABLE hist_dow_7d AS
SELECT 
    day_of_week,
    AVG(delay_minutes) as avg_delay
FROM operational.delay_events
WHERE actual_arrival >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY day_of_week;

CREATE INDEX idx_hist_dow7d ON hist_dow_7d(day_of_week);

-- Weather averages (7-day)
CREATE TEMP TABLE hist_weather_7d AS
SELECT 
    weather_condition,
    AVG(delay_minutes) as avg_delay
FROM operational.delay_events
WHERE actual_arrival >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY weather_condition;

CREATE INDEX idx_hist_w7d ON hist_weather_7d(weather_condition);

-- Previous stop delays
CREATE TEMP TABLE prev_stop_delays AS
SELECT 
    de.trip_id,
    st.stop_sequence,
    de.delay_minutes,
    AVG(de.delay_minutes) OVER (
        PARTITION BY de.trip_id 
        ORDER BY st.stop_sequence 
        ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING
    ) as prev_delay
FROM operational.delay_events de
INNER JOIN operational.stop_times st ON de.trip_id = st.trip_id AND de.stop_id = st.stop_id;

CREATE INDEX idx_prev_stop ON prev_stop_delays(trip_id, stop_sequence);

-- UPDATE WITH HISTORICAL FEATURES
-- Update 7-day route-stop features
UPDATE ml.delay_features df
SET 
    avg_delay_same_route_stop_7d = COALESCE(h.avg_delay, 0),
    delay_count_same_route_stop_7d = COALESCE(h.delay_count, 0),
    max_delay_same_route_stop_7d = COALESCE(h.max_delay, 0)
FROM hist_route_stop_7d h
WHERE df.trip_id = h.trip_id AND df.stop_id = h.stop_id;

-- Update 30-day route-stop features
UPDATE ml.delay_features df
SET avg_delay_same_route_stop_30d = COALESCE(h.avg_delay, 0)
FROM hist_route_stop_30d h
WHERE df.trip_id = h.trip_id AND df.stop_id = h.stop_id;

-- Update 7-day route features
UPDATE ml.delay_features df
SET 
    avg_delay_route_7d = COALESCE(h.avg_delay, 0),
    stddev_delay_route_7d = COALESCE(h.stddev_delay, 0)
FROM hist_route_7d h
WHERE df.route_id = h.route_id;

-- Update 30-day route features
UPDATE ml.delay_features df
SET avg_delay_route_30d = COALESCE(h.avg_delay, 0)
FROM hist_route_30d h
WHERE df.route_id = h.route_id;

-- Update stop features
UPDATE ml.delay_features df
SET avg_delay_stop_7d = COALESCE(h.avg_delay, 0)
FROM hist_stop_7d h
WHERE df.stop_id = h.stop_id;

UPDATE ml.delay_features df
SET avg_delay_stop_30d = COALESCE(h.avg_delay, 0)
FROM hist_stop_30d h
WHERE df.stop_id = h.stop_id;

-- Update hour features
UPDATE ml.delay_features df
SET avg_delay_same_hour_7d = COALESCE(h.avg_delay, 0)
FROM hist_hour_7d h
WHERE df.hour_of_day = h.hour;

UPDATE ml.delay_features df
SET avg_delay_same_hour_30d = COALESCE(h.avg_delay, 0)
FROM hist_hour_30d h
WHERE df.hour_of_day = h.hour;

-- Update day-of-week features
UPDATE ml.delay_features df
SET avg_delay_same_dow_7d = COALESCE(h.avg_delay, 0)
FROM hist_dow_7d h
WHERE df.day_of_week = h.day_of_week;

-- Update weather features
UPDATE ml.delay_features df
SET avg_delay_same_weather_7d = COALESCE(h.avg_delay, 0)
FROM hist_weather_7d h
WHERE df.weather_condition = h.weather_condition;

-- Update previous stop delays
UPDATE ml.delay_features df
SET prev_stop_delay = COALESCE(p.prev_delay::INTEGER, 0)
FROM prev_stop_delays p
WHERE df.trip_id = p.trip_id AND df.stop_sequence = p.stop_sequence;

-- Compute volatility (using stddev from route)
UPDATE ml.delay_features
SET delay_volatility_7d = stddev_delay_route_7d;