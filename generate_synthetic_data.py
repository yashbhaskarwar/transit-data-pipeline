import psycopg2
from datetime import datetime, timedelta
from typing import List, Tuple
import sys

# DATABASE CONFIGURATION

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'transit_delay_optimization',
    'user': 'postgres',      
    'password': 'postgres'   
}

# DATA GENERATION PARAMETERS

GENERATION_PARAMS = {
    'days_to_generate': 60,
    'base_delay_probability': 0.15,  # delays for 15% of trips
    'weather_conditions': [
        'clear', 'partly_cloudy', 'cloudy', 'rainy', 
        'heavy_rain', 'snow', 'fog', 'windy'
    ],
    'incident_types': [
        'traffic_accident', 'vehicle_breakdown', 'road_construction',
        'special_event', 'medical_emergency', 'power_outage'
    ]
}

# DATABASE CONNECTION

def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("Connected")
        return conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        sys.exit(1)

# DATE RANGE DETECTION

def detect_date_range(conn) -> Tuple[datetime, datetime]:
    cursor = conn.cursor()
    # Get active service date ranges
    cursor.execute("""
        SELECT 
            MIN(start_date) as earliest_start,
            MAX(end_date) as latest_end
        FROM operational.calendar
        WHERE end_date >= CURRENT_DATE - INTERVAL '90 days'
    """)
    
    result = cursor.fetchone()
    earliest_start, latest_end = result
    
    if not earliest_start or not latest_end:
        print("No valid service dates found in calendar table")
        sys.exit(1)
    
    end_date = min(datetime.now().date(), latest_end)
    start_date = end_date - timedelta(days=59)
    
    if start_date < earliest_start:
        start_date = earliest_start
        end_date = start_date + timedelta(days=59)
    
    cursor.close()
    
    print(f"Detected date range: {start_date} to {end_date}")
    return start_date, end_date

# SERVICE PATTERN DETECTION

def get_active_services(conn, target_date: datetime.date) -> List[str]:
    cursor = conn.cursor()
    
    day_of_week = target_date.strftime('%A').lower()
    
    cursor.execute(f"""
        SELECT service_id
        FROM operational.calendar
        WHERE {day_of_week} = TRUE
          AND start_date <= %s
          AND end_date >= %s
    """, (target_date, target_date))
    
    services = [row[0] for row in cursor.fetchall()]
    cursor.close()
    
    return services


def get_trips_for_date(conn, target_date: datetime.date) -> List[Tuple]:
    services = get_active_services(conn, target_date)
    
    if not services:
        return []
    
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            t.trip_id,
            t.route_id,
            st.stop_id,
            st.stop_sequence,
            st.arrival_time,
            st.departure_time
        FROM operational.trips t
        INNER JOIN operational.stop_times st ON t.trip_id = st.trip_id
        WHERE t.service_id = ANY(%s)
        ORDER BY t.trip_id, st.stop_sequence
    """, (services,))
    
    trips = cursor.fetchall()
    cursor.close()
    
    return trips
