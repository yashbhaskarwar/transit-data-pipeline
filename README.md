# Automated Transit Data Pipeline with ML Integration

This project analyzes public transit delay patterns using GTFS data and a PostgreSQL data warehouse.

## How to run

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
