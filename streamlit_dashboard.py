import streamlit as st
import pandas as pd
import psycopg2

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
        