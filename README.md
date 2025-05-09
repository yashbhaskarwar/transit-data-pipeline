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
