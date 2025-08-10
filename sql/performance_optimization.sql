CREATE SCHEMA IF NOT EXISTS analytics;

-- STRATEGIC INDEXES 

-- Index 1: Speed up delay queries by date
CREATE INDEX IF NOT EXISTS idx_delay_events_date 
ON operational.delay_events(actual_arrival);

-- Index 2: Speed up route analysis
CREATE INDEX IF NOT EXISTS idx_delay_events_route_date 
ON operational.delay_events(trip_id, actual_arrival);

-- Index 3: Speed up feature queries by route/stop
CREATE INDEX IF NOT EXISTS idx_ml_features_route_stop 
ON ml.delay_features(route_id, stop_id);

-- Index 4: Speed up warehouse fact table queries
CREATE INDEX IF NOT EXISTS idx_fact_delay_date 
ON warehouse.fact_delay_events(date_key);

-- Index 5: Speed up warehouse queries by route
CREATE INDEX IF NOT EXISTS idx_fact_delay_route 
ON warehouse.fact_delay_events(route_key, date_key);

-- MATERIALIZED VIEWS 

-- View 1: Daily route performance summary
DROP MATERIALIZED VIEW IF EXISTS analytics.mv_daily_route_performance CASCADE;
CREATE MATERIALIZED VIEW analytics.mv_daily_route_performance AS
SELECT 
    dr.route_id,
    dr.route_short_name,
    dd.full_date,
    COUNT(*) as total_delays,
    AVG(fde.delay_minutes) as avg_delay_minutes,
    MAX(fde.delay_minutes) as max_delay_minutes,
    COUNT(*) FILTER (WHERE fde.delay_minutes > 15) as severe_delays,
    ROUND(AVG(fde.delay_minutes)::NUMERIC, 2) as avg_delay
FROM warehouse.fact_delay_events fde
INNER JOIN warehouse.dim_route dr ON fde.route_key = dr.route_key
INNER JOIN warehouse.dim_date dd ON fde.date_key = dd.date_key
GROUP BY dr.route_id, dr.route_short_name, dd.full_date
ORDER BY dd.full_date DESC, dr.route_id;

CREATE INDEX idx_mv_daily_route_date ON analytics.mv_daily_route_performance(full_date);
CREATE INDEX idx_mv_daily_route_id ON analytics.mv_daily_route_performance(route_id);

-- View 2: Hourly pattern analysis
DROP MATERIALIZED VIEW IF EXISTS analytics.mv_hourly_patterns CASCADE;
CREATE MATERIALIZED VIEW analytics.mv_hourly_patterns AS
SELECT 
    dt.hour,
    dt.is_rush_hour,
    COUNT(*) as delay_count,
    AVG(fde.delay_minutes) as avg_delay,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY fde.delay_minutes) as median_delay,
    COUNT(*) FILTER (WHERE fde.delay_minutes > 20) as severe_count
FROM warehouse.fact_delay_events fde
INNER JOIN warehouse.dim_time dt ON fde.time_key = dt.time_key
GROUP BY dt.hour, dt.is_rush_hour
ORDER BY dt.hour;

CREATE INDEX idx_mv_hourly_hour ON analytics.mv_hourly_patterns(hour);

-- View 3: Weather impact summary
DROP MATERIALIZED VIEW IF EXISTS analytics.mv_weather_impact CASCADE;
CREATE MATERIALIZED VIEW analytics.mv_weather_impact AS
SELECT 
    dw.weather_condition,
    dw.severity_level,
    COUNT(*) as delay_count,
    AVG(fde.delay_minutes) as avg_delay,
    MAX(fde.delay_minutes) as max_delay,
    COUNT(*) FILTER (WHERE fde.delay_minutes > 15) as severe_delays
FROM warehouse.fact_delay_events fde
INNER JOIN warehouse.dim_weather dw ON fde.weather_key = dw.weather_key
GROUP BY dw.weather_condition, dw.severity_level
ORDER BY avg_delay DESC;

CREATE INDEX idx_mv_weather_condition ON analytics.mv_weather_impact(weather_condition);

-- Function to refresh all materialized views
CREATE OR REPLACE FUNCTION analytics.refresh_all_views()
RETURNS TEXT AS $$
BEGIN
    REFRESH MATERIALIZED VIEW analytics.mv_daily_route_performance;
    REFRESH MATERIALIZED VIEW analytics.mv_hourly_patterns;
    REFRESH MATERIALIZED VIEW analytics.mv_weather_impact;
    RETURN 'All materialized views refreshed successfully';
END;
$$ LANGUAGE plpgsql;

-- VACUUM & ANALYZE 

-- VACUUM: Reclaim storage and improve performance
VACUUM ANALYZE operational.delay_events;
VACUUM ANALYZE operational.trips;
VACUUM ANALYZE operational.stop_times;
VACUUM ANALYZE warehouse.fact_delay_events;
VACUUM ANALYZE ml.delay_features;

-- ANALYZE: Update statistics for query optimizer
ANALYZE operational.delay_events;
ANALYZE operational.trips;
ANALYZE operational.routes;
ANALYZE warehouse.fact_delay_events;
ANALYZE warehouse.dim_route;
ANALYZE warehouse.dim_stop;
ANALYZE ml.delay_features;


