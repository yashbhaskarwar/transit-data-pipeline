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
