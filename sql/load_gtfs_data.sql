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

