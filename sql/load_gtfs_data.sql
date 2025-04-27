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