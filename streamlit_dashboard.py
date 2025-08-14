import streamlit as st
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

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

# KEY METRICS

st.header("Key Performance Metrics")
col1, col2, col3, col4 = st.columns(4)
try:
    total_delays_query = f"""
        SELECT COUNT(*) as total
        FROM operational.delay_events
        WHERE DATE(actual_arrival) BETWEEN '{start_date}' AND '{end_date}'
    """
    total_delays = run_query(total_delays_query)['total'].iloc[0]
    
    # Average delay
    avg_delay_query = f"""
        SELECT AVG(delay_minutes) as avg_delay
        FROM operational.delay_events
        WHERE DATE(actual_arrival) BETWEEN '{start_date}' AND '{end_date}'
    """
    avg_result = run_query(avg_delay_query)
    avg_delay = avg_result['avg_delay'].iloc[0] if avg_result['avg_delay'].iloc[0] is not None else 0
    
    # Model accuracy 
    model_accuracy = 88.80  # test accuracy
    
    # High-risk delays
    high_risk_query = f"""
        SELECT COUNT(*) as high_risk
        FROM operational.delay_events
        WHERE DATE(actual_arrival) BETWEEN '{start_date}' AND '{end_date}'
          AND delay_minutes > 20
    """
    high_risk = run_query(high_risk_query)['high_risk'].iloc[0]
    
    with col1:
        st.metric("Total Delays (7d)", f"{total_delays:,}")
    
    with col2:
        st.metric("Avg Delay", f"{avg_delay:.1f} min")
    
    with col3:
        st.metric("Model Accuracy", f"{model_accuracy:.1f}%", delta="3.13%")
    
    with col4:
        st.metric("High-Risk (>20min)", f"{high_risk:,}")

except Exception as e:
    st.error(f"Error loading metrics: {e}")

# DELAY TRENDS

st.header("Delay Trends Over Time")
try:
    # Use selected date range
    trend_query = f"""
        SELECT 
            DATE(actual_arrival) as date,
            COUNT(*) as delay_count,
            AVG(delay_minutes) as avg_delay,
            MAX(delay_minutes) as max_delay
        FROM operational.delay_events
        WHERE DATE(actual_arrival) BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY DATE(actual_arrival)
        ORDER BY date
    """
    df_trend = run_query(trend_query)
    
    if not df_trend.empty:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_trend['date'],
            y=df_trend['avg_delay'],
            mode='lines+markers',
            name='Avg Delay',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df_trend['date'],
            y=df_trend['max_delay'],
            mode='lines',
            name='Max Delay',
            line=dict(color='#ff7f0e', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title="Daily Delay Patterns (Last 30 Days)",
            xaxis_title="Date",
            yaxis_title="Delay (minutes)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trend data available")

except Exception as e:
    st.error(f"Error loading trends: {e}")

# FOOTER
st.markdown("**Transit Delay Prediction System**")

# ROUTE PERFORMANCE
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Top 10 Routes by Delays")
    
    try:
        route_query = f"""
            SELECT 
                t.route_id,
                COUNT(*) as delay_count,
                AVG(de.delay_minutes) as avg_delay,
                MAX(de.delay_minutes) as max_delay
            FROM operational.delay_events de
            INNER JOIN operational.trips t ON de.trip_id = t.trip_id
            WHERE DATE(de.actual_arrival) BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY t.route_id
            ORDER BY delay_count DESC
            LIMIT 10
        """
        df_routes = run_query(route_query)
        
        if not df_routes.empty:
            fig = px.bar(
                df_routes,
                x='route_id',
                y='delay_count',
                title='Delays by Route (Last 7 Days)',
                labels={'delay_count': 'Number of Delays', 'route_id': 'Route'},
                color='avg_delay',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            st.dataframe(
                df_routes.style.format({
                    'avg_delay': '{:.1f}',
                    'max_delay': '{:.0f}'
                }),
                use_container_width=True
            )
        else:
            st.info("No route data available")
    
    except Exception as e:
        st.error(f"Error loading route data: {e}")

with col_right:
    st.subheader("Hourly Delay Patterns")
    
    try:
        hourly_query = f"""
            SELECT 
                EXTRACT(HOUR FROM actual_arrival)::INTEGER as hour,
                COUNT(*) as delay_count,
                AVG(delay_minutes) as avg_delay
            FROM operational.delay_events
            WHERE DATE(actual_arrival) BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY EXTRACT(HOUR FROM actual_arrival)::INTEGER
            ORDER BY hour
        """
        df_hourly = run_query(hourly_query)
        
        if not df_hourly.empty:
            fig = px.line(
                df_hourly,
                x='hour',
                y='avg_delay',
                markers=True,
                title='Average Delay by Hour of Day',
                labels={'hour': 'Hour of Day', 'avg_delay': 'Avg Delay (min)'}
            )
            
            # Highlight rush hours
            fig.add_vrect(x0=7, x1=9, fillcolor="red", opacity=0.1, annotation_text="AM Rush")
            fig.add_vrect(x0=17, x1=19, fillcolor="red", opacity=0.1, annotation_text="PM Rush")
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hourly data available")
    
    except Exception as e:
        st.error(f"Error loading hourly data: {e}")
