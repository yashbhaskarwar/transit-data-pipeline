import psycopg2
import numpy as np
from datetime import datetime, timedelta, time
import random
from typing import List, Tuple, Dict
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

# WEATHER DATA GENERATION

def generate_weather_data(start_date: datetime.date, end_date: datetime.date) -> List[Dict]:
    weather_data = []
    current_date = datetime.combine(start_date, time(0, 0))
    end_datetime = datetime.combine(end_date, time(23, 0))
    
    # Generate weather clusters 
    weather_cluster_duration = 0
    current_weather = 'clear'
    
    while current_date <= end_datetime:
        # Change weather cluster periodically
        if weather_cluster_duration <= 0:
            current_weather = random.choice(GENERATION_PARAMS['weather_conditions'])
            weather_cluster_duration = random.randint(6, 48)  # 6-48 hours
        
        weather_cluster_duration -= 1
        
        day_of_year = current_date.timetuple().tm_yday
        hour_of_day = current_date.hour
        
        # Seasonal temperature 
        seasonal_temp = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Daily temperature variation
        daily_variation = 8 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        temperature = seasonal_temp + daily_variation + random.gauss(0, 2)
        
        # Precipitation based on weather condition
        precip_map = {
            'clear': 0, 'partly_cloudy': 0, 'cloudy': 0,
            'rainy': random.uniform(1, 5), 
            'heavy_rain': random.uniform(5, 15),
            'snow': random.uniform(2, 8),
            'fog': random.uniform(0, 0.5),
            'windy': 0
        }
        precipitation = precip_map.get(current_weather, 0)
        
        # Wind speed
        wind_base = 5 if current_weather != 'windy' else 20
        wind_speed = max(0, wind_base + random.gauss(0, 5))
        
        # Visibility
        visibility_map = {
            'clear': 10, 'partly_cloudy': 9, 'cloudy': 8,
            'rainy': 5, 'heavy_rain': 3, 'snow': 4,
            'fog': 1, 'windy': 7
        }
        visibility = visibility_map.get(current_weather, 8) + random.gauss(0, 1)
        visibility = max(0.5, min(10, visibility))
        
        weather_data.append({
            'recorded_at': current_date,
            'temperature': round(temperature, 2),
            'precipitation': round(precipitation, 2),
            'wind_speed': round(wind_speed, 2),
            'visibility': round(visibility, 2),
            'weather_condition': current_weather
        })
        
        current_date += timedelta(hours=1)
    
    print(f"Generated {len(weather_data)} hourly weather records")
    return weather_data

# DELAY DATA GENERATION

def calculate_delay_probability(hour: int, weather: str, is_holiday: bool) -> float:
    base_prob = GENERATION_PARAMS['base_delay_probability']
    
    # Delays increased in rush hour
    if hour in [7, 8, 9, 17, 18, 19]:
        base_prob *= 1.8
    
    # Weather impact
    weather_multipliers = {
        'clear': 1.0,
        'partly_cloudy': 1.1,
        'cloudy': 1.2,
        'rainy': 1.5,
        'heavy_rain': 2.0,
        'snow': 2.5,
        'fog': 1.8,
        'windy': 1.3
    }
    base_prob *= weather_multipliers.get(weather, 1.0)
    
    # Holiday impact
    if is_holiday:
        base_prob *= 0.7  
    
    return min(base_prob, 0.8)  # Cap at 80%


def generate_delay_minutes(weather: str, hour: int) -> int:
    # Base delay distribution (most delays are short)
    if random.random() < 0.6:
        base_delay = random.randint(1, 5)  
    elif random.random() < 0.85:
        base_delay = random.randint(6, 15)  
    else:
        base_delay = random.randint(16, 45) 
    
    # Weather amplification
    weather_amplifiers = {
        'clear': 1.0, 'partly_cloudy': 1.1, 'cloudy': 1.2,
        'rainy': 1.4, 'heavy_rain': 1.8, 'snow': 2.0,
        'fog': 1.5, 'windy': 1.2
    }
    
    delay = int(base_delay * weather_amplifiers.get(weather, 1.0))
    
    if hour in [8, 9, 17, 18]:
        delay = int(delay * random.uniform(1.0, 1.5))
    
    return max(1, min(delay, 120))  # Cap at 2 hours


def generate_delay_events(conn, start_date: datetime.date, end_date: datetime.date, 
                         weather_data: List[Dict]) -> List[Dict]:
    
    weather_lookup = {w['recorded_at']: w for w in weather_data}
    
    delay_events = []
    current_date = start_date
    
    # Simple holiday detection 
    holidays = set()
    temp_date = start_date
    while temp_date <= end_date:
        if random.random() < 0.05: 
            holidays.add(temp_date)
        temp_date += timedelta(days=1)
    
    day_count = 0
    while current_date <= end_date:
        day_count += 1
        if day_count % 10 == 0:
            print(f"  Processing day {day_count}...")
        
        is_holiday = current_date in holidays
        trips = get_trips_for_date(conn, current_date)
        
        if not trips:
            current_date += timedelta(days=1)
            continue
        
        # Process trips for this day
        for trip in trips:
            trip_id, route_id, stop_id, stop_seq, arrival_time, departure_time = trip
            
            if not arrival_time:
                continue
            
            # Convert interval to hour of day
            total_seconds = arrival_time.total_seconds()
            hour = int((total_seconds // 3600) % 24)
            
            # Get weather for this hour
            trip_datetime = datetime.combine(current_date, time(hour, 0))
            weather_record = weather_lookup.get(trip_datetime)
            weather_condition = weather_record['weather_condition'] if weather_record else 'clear'
            
            # Determine if delay occurs
            delay_prob = calculate_delay_probability(hour, weather_condition, is_holiday)
            
            if random.random() < delay_prob:
                delay_minutes = generate_delay_minutes(weather_condition, hour)
                
                # Calculate actual arrival time
                actual_arrival = trip_datetime + timedelta(
                    seconds=total_seconds % 3600,
                    minutes=delay_minutes
                )
                
                delay_events.append({
                    'trip_id': trip_id,
                    'stop_id': stop_id,
                    'scheduled_arrival': arrival_time,
                    'actual_arrival': actual_arrival,
                    'delay_minutes': delay_minutes,
                    'weather_condition': weather_condition,
                    'day_of_week': current_date.weekday(),
                    'is_holiday': is_holiday
                })
        
        current_date += timedelta(days=1)
    
    print(f"Generated {len(delay_events)} delay events")
    return delay_events
