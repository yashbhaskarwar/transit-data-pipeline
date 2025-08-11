import streamlit as st
import pandas as pd
import psycopg2
from datetime import datetime, timedelta

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Transit Delay Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# DATABASE CONNECTION 

def get_db_connection():
    return psycopg2.connect(
        host='localhost',
        port=5432,
        database='transit_delay_optimization',
        user='postgres',
        password='postgres'  # UPDATE
    )

def run_query(query):
    conn = get_db_connection()
    try:
        df = pd.read_sql(query, conn)
        return df
    finally:
        conn.close()

# HEADER
st.title("Transit Delay Prediction Dashboard")
st.markdown("Real-time monitoring and ML predictions for transit delays")

# SIDEBAR
st.sidebar.header("Filters")
# Get actual date range from data
try:
    date_range_query = """
        SELECT 
            MIN(DATE(actual_arrival)) as min_date,
            MAX(DATE(actual_arrival)) as max_date
        FROM operational.delay_events
    """
    date_info = run_query(date_range_query)
    min_date_available = date_info['min_date'].iloc[0]
    max_date_available = date_info['max_date'].iloc[0]
except:
    min_date_available = datetime(2025, 9, 1).date()
    max_date_available = datetime.now().date()

# Date range filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(max_date_available - timedelta(days=7), max_date_available),
    min_value=min_date_available,
    max_value=max_date_available
)

# Convert to start and end dates
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = date_range
    end_date = date_range

# Route filter
try:
    routes_df = run_query("SELECT DISTINCT route_id FROM warehouse.dim_route ORDER BY route_id")
    route_options = ['All'] + routes_df['route_id'].tolist()
    selected_route = st.sidebar.selectbox("Select Route", route_options)
except Exception as e:
    st.sidebar.error(f"Could not load routes: {e}")
    selected_route = 'All'

# FOOTER
st.markdown("**Transit Delay Prediction System**")