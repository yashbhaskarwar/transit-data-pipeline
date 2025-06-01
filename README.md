# Automated Transit Data Pipeline with ML Integration

This project analyzes public transit delay patterns using GTFS data and a PostgreSQL data warehouse.

## Phase 1: Database Foundation

### How to run

1. Create the schema:
```bash
psql -U postgres -d transit_delay_optimization -f sql/create_schema.sql
```
2. Download available GTFS dataset and place the text files in: data/gtfs/

3. Open sql/load_gtfs_data.sql and update the file paths to match your GTFS dataset location.

4. Load the data:
```bash
psql -U postgres -d transit_delay_optimization -f sql/load_gtfs_data.sql
```
This loads the raw files, cleans them, fills the operational tables and runs basic checks.

## Phase 2: Synthetic Weather and Delay Data

My GTFS dataset had only scheduled times and didn't had any weather and delay data. So, this phase adds a synthetic generator to create the missing data needed for analysis and ML training.

### How to run

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```
2. Update database credentials inside generate_synthetic_data.py

3. Execute the script
```bash
python generate_synthetic_data.py
```

## Phase 3 – Data Warehouse Layer (Fact + Dimension Tables)

In this phase, we build the warehouse layer for analytics and future ML work.  
It creates a star-schema structure, fills it with data and adds a set of analytical SQL queries.
The warehouse script sets up all dimension tables, fact tables and their relationships, then loads them with data from the operational layer.

### How to run
 
1. Run the warehouse schema and population script:
```bash
psql -U postgres -d transit_delay_optimization -f sql/fact_dim_tables.sql
```
2. (Optional) Run all analytical queries:
This queries are included to explore the warehouse tables and understand patterns in the data. They are helpful for analysis and model preparation but are not required for the main pipeline to run.
```bash
psql -U postgres -d transit_delay_optimization -f sql/analysis_queries.sql
```