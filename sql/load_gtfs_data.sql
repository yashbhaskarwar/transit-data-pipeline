-- stops.txt 
COPY staging.stops (
    stop_id,
    stop_code,
    stop_name,
    stop_lat,
    stop_lon,
    wheelchair_boarding,
    platform_code,
    stop_url
)
FROM '\data\gtfsstops.txt'
DELIMITER ',' 
CSV HEADER
NULL AS '';

-- routes.txt 
COPY staging.routes (
    route_id,
    agency_id,
    route_short_name,
    route_long_name,
    route_url,
    route_desc,
    route_type,
    route_color,
    route_text_color,
    route_sort_order
)
FROM '\data\gtfs\routes.txt'
DELIMITER ',' 
CSV HEADER
NULL AS '';

-- trips.txt
COPY staging.trips (
    route_id,
    service_id,
    trip_id,
    trip_headsign,
    direction_id,
    block_id,
    shape_id,
    wheelchair_accessible,
    bikes_allowed
)
FROM '\data\gtfs\trips.txt'
DELIMITER ',' 
CSV HEADER
NULL AS '';

-- stop_times.txt
COPY staging.stop_times (
    trip_id,
    arrival_time,
    departure_time,
    stop_id,
    stop_sequence,
    stop_headsign,
    shape_dist_traveled,
    timepoint
)
FROM '\data\gtfs\stop_times.txt'
DELIMITER ',' 
CSV HEADER
NULL AS '';

-- calendar.txt 
COPY staging.calendar (
    service_id,
    monday,
    tuesday,
    wednesday,
    thursday,
    friday,
    saturday,
    sunday,
    start_date,
    end_date
)
FROM '\data\gtfs\calendar.txt'
DELIMITER ',' 
CSV HEADER
NULL AS '';

-- DATA QUALITY CHECKS

-- null coordinates in stops
SELECT 
    'Stops with null coordinates' AS check_type,
    COUNT(*) AS count
FROM staging.stops
WHERE stop_lat IS NULL OR stop_lon IS NULL;

-- trips without routes
SELECT 
    'Trips without matching routes' AS check_type,
    COUNT(*) AS count
FROM staging.trips t
LEFT JOIN staging.routes r ON t.route_id = r.route_id
WHERE r.route_id IS NULL;

-- invalid stop_times
SELECT 
    'Stop times with null trip_id or stop_id' AS check_type,
    COUNT(*) AS count
FROM staging.stop_times
WHERE trip_id IS NULL OR stop_id IS NULL OR arrival_time IS NULL;

-- calendar size
SELECT 
    'Calendar entries (services)' AS check_type,
    COUNT(*) AS count
FROM staging.calendar;

-- STEP 3: TRANSFORM AND LOAD INTO OPERATIONAL TABLES

-- operational.stops
INSERT INTO operational.stops (
    stop_id,
    stop_code,
    stop_name,
    stop_lat,
    stop_lon,
    wheelchair_boarding,
    platform_code
)
SELECT 
    stop_id,
    stop_code,
    stop_name,
    stop_lat,
    stop_lon,
    COALESCE(wheelchair_boarding, 0),
    platform_code
FROM staging.stops
WHERE stop_lat IS NOT NULL 
  AND stop_lon IS NOT NULL
ON CONFLICT (stop_id) DO NOTHING;

-- operational.routes 
INSERT INTO operational.routes (
    route_id,
    agency_id,
    route_short_name,
    route_long_name,
    route_type,
    route_color,
    route_text_color,
    route_sort_order
)
SELECT 
    route_id,
    agency_id,
    route_short_name,
    route_long_name,
    route_type,
    COALESCE(route_color, 'FFFFFF'),
    COALESCE(route_text_color, '000000'),
    route_sort_order
FROM staging.routes
ON CONFLICT (route_id) DO NOTHING;

-- operational.trips
INSERT INTO operational.trips (
    trip_id,
    route_id,
    service_id,
    trip_headsign,
    direction_id,
    block_id,
    shape_id,
    wheelchair_accessible
)
SELECT 
    t.trip_id,
    t.route_id,
    t.service_id,
    t.trip_headsign,
    COALESCE(t.direction_id, 0),
    t.block_id,
    t.shape_id,
    COALESCE(t.wheelchair_accessible, 0)
FROM staging.trips t
INNER JOIN operational.routes r ON t.route_id = r.route_id
ON CONFLICT (trip_id) DO NOTHING;

-- operational.stop_times 
INSERT INTO operational.stop_times (
    trip_id,
    stop_id,
    stop_sequence,
    arrival_time,
    departure_time,
    stop_headsign,
    shape_dist_traveled,
    timepoint
)
SELECT 
    st.trip_id,
    st.stop_id,
    st.stop_sequence,
    staging.gtfs_time_to_interval(st.arrival_time),
    staging.gtfs_time_to_interval(st.departure_time),
    st.stop_headsign,
    st.shape_dist_traveled,
    COALESCE(st.timepoint, 1)
FROM staging.stop_times st
INNER JOIN operational.trips t ON st.trip_id = t.trip_id
INNER JOIN operational.stops s ON st.stop_id = s.stop_id
WHERE st.arrival_time IS NOT NULL
ON CONFLICT (trip_id, stop_sequence) DO NOTHING;

-- operational.calendar 
INSERT INTO operational.calendar (
    service_id,
    monday,
    tuesday,
    wednesday,
    thursday,
    friday,
    saturday,
    sunday,
    start_date,
    end_date
)
SELECT 
    service_id,
    monday::BOOLEAN,
    tuesday::BOOLEAN,
    wednesday::BOOLEAN,
    thursday::BOOLEAN,
    friday::BOOLEAN,
    saturday::BOOLEAN,
    sunday::BOOLEAN,
    start_date,
    end_date
FROM staging.calendar
ON CONFLICT (service_id) DO NOTHING;

-- POST-LOAD VERIFICATION

-- Count records in all tables
SELECT 
    'STAGING LAYER' AS layer,
    'stops' AS table_name,
    COUNT(*) AS record_count
FROM staging.stops
UNION ALL
SELECT 'STAGING LAYER', 'routes', COUNT(*) FROM staging.routes
UNION ALL
SELECT 'STAGING LAYER', 'trips', COUNT(*) FROM staging.trips
UNION ALL
SELECT 'STAGING LAYER', 'stop_times', COUNT(*) FROM staging.stop_times
UNION ALL
SELECT 'STAGING LAYER', 'calendar', COUNT(*) FROM staging.calendar
UNION ALL
SELECT 'OPERATIONAL LAYER', 'stops', COUNT(*) FROM operational.stops
UNION ALL
SELECT 'OPERATIONAL LAYER', 'routes', COUNT(*) FROM operational.routes
UNION ALL
SELECT 'OPERATIONAL LAYER', 'trips', COUNT(*) FROM operational.trips
UNION ALL
SELECT 'OPERATIONAL LAYER', 'stop_times', COUNT(*) FROM operational.stop_times
UNION ALL
SELECT 'OPERATIONAL LAYER', 'calendar', COUNT(*) FROM operational.calendar
ORDER BY layer, table_name;

-- sample data from operational layer
SELECT stop_id, stop_name, stop_lat, stop_lon 
FROM operational.stops 
LIMIT 5;

SELECT route_id, route_short_name, route_long_name, route_type 
FROM operational.routes 
LIMIT 5;
