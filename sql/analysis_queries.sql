-- Route Performance Ranking with Trends

WITH weekly_performance AS (
    SELECT 
        dr.route_short_name,
        dr.route_long_name,
        dd.week_of_year,
        dd.year,
        COUNT(*) as total_delays,
        AVG(fde.delay_minutes) as avg_delay_minutes,
        SUM(CASE WHEN fde.is_severe_delay THEN 1 ELSE 0 END) as severe_delays
    FROM warehouse.fact_delay_events fde
    INNER JOIN warehouse.dim_route dr ON fde.route_key = dr.route_key
    INNER JOIN warehouse.dim_date dd ON fde.date_key = dd.date_key
    GROUP BY dr.route_short_name, dr.route_long_name, dd.week_of_year, dd.year
),
ranked_routes AS (
    SELECT 
        route_short_name,
        route_long_name,
        week_of_year,
        year,
        total_delays,
        avg_delay_minutes,
        severe_delays,
        LAG(avg_delay_minutes) OVER (
            PARTITION BY route_short_name 
            ORDER BY year, week_of_year
        ) as prev_week_avg_delay,
        ROW_NUMBER() OVER (
            PARTITION BY year, week_of_year 
            ORDER BY avg_delay_minutes DESC
        ) as performance_rank
    FROM weekly_performance
)
SELECT 
    route_short_name as "Route",
    route_long_name as "Route Name",
    week_of_year as "Week",
    year as "Year",
    total_delays as "Total Delays",
    ROUND(avg_delay_minutes::numeric, 2) as "Avg Delay (min)",
    severe_delays as "Severe Delays",
    ROUND(COALESCE(prev_week_avg_delay, 0)::numeric, 2) as "Prev Week Avg",
    ROUND((avg_delay_minutes - COALESCE(prev_week_avg_delay, avg_delay_minutes))::numeric, 2) as "Change",
    performance_rank as "Rank (Worst→Best)"
FROM ranked_routes
WHERE performance_rank <= 5  
ORDER BY year DESC, week_of_year DESC, performance_rank
LIMIT 20;

-- Rush Hour vs Off-Peak Analysis

SELECT 
    dt.rush_hour_period as "Time Period",
    COUNT(*) as "Total Delays",
    ROUND(AVG(fde.delay_minutes)::numeric, 2) as "Avg Delay (min)",
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY fde.delay_minutes)::numeric, 2) as "Median Delay",
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY fde.delay_minutes)::numeric, 2) as "95th Percentile",
    MAX(fde.delay_minutes) as "Max Delay (min)",
    SUM(CASE WHEN fde.is_significant_delay THEN 1 ELSE 0 END) as "Significant Delays (>10min)",
    ROUND(100.0 * SUM(CASE WHEN fde.is_significant_delay THEN 1 ELSE 0 END) / COUNT(*)::numeric, 2) as "Significant %",
    -- Calculate delay cost (assume $0.50 per passenger per minute)
    ROUND((AVG(fde.delay_minutes) * 50 * 0.50)::numeric, 2) as "Est. Cost per Delay ($)"
FROM warehouse.fact_delay_events fde
INNER JOIN warehouse.dim_time dt ON fde.time_key = dt.time_key
GROUP BY dt.rush_hour_period
ORDER BY AVG(fde.delay_minutes) DESC;

-- Weather Impact Analysis with Moving Average

WITH daily_weather_delays AS (
    SELECT 
        dd.full_date,
        dw.weather_condition,
        dw.severity_level,
        COUNT(*) as daily_delays,
        AVG(fde.delay_minutes) as avg_delay
    FROM warehouse.fact_delay_events fde
    INNER JOIN warehouse.dim_date dd ON fde.date_key = dd.date_key
    INNER JOIN warehouse.dim_weather dw ON fde.weather_key = dw.weather_key
    GROUP BY dd.full_date, dw.weather_condition, dw.severity_level
)
SELECT 
    full_date as "Date",
    weather_condition as "Weather",
    severity_level as "Severity",
    daily_delays as "Daily Delays",
    ROUND(avg_delay::numeric, 2) as "Avg Delay (min)",
    ROUND(AVG(avg_delay) OVER (
        ORDER BY full_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    )::numeric, 2) as "7-Day Moving Avg",
    ROUND(AVG(daily_delays) OVER (
        ORDER BY full_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    )::numeric, 2) as "7-Day Avg Delays"
FROM daily_weather_delays
ORDER BY full_date DESC
LIMIT 30;

-- Stop Performance with Cumulative Analysis

WITH stop_performance AS (
    SELECT 
        ds.stop_name,
        ds.stop_area,
        ds.is_major_hub,
        COUNT(*) as total_delays,
        SUM(fde.delay_minutes) as total_delay_minutes,
        AVG(fde.delay_minutes) as avg_delay_minutes,
        MAX(fde.delay_minutes) as max_delay
    FROM warehouse.fact_delay_events fde
    INNER JOIN warehouse.dim_stop ds ON fde.stop_key = ds.stop_key
    GROUP BY ds.stop_name, ds.stop_area, ds.is_major_hub
)
SELECT 
    stop_name as "Stop Name",
    stop_area as "Area",
    CASE WHEN is_major_hub THEN 'Yes' ELSE 'No' END as "Major Hub?",
    total_delays as "Total Delays",
    ROUND(avg_delay_minutes::numeric, 2) as "Avg Delay (min)",
    total_delay_minutes as "Total Minutes Lost",
    ROUND(100.0 * SUM(total_delay_minutes) OVER (
        ORDER BY total_delay_minutes DESC
    ) / SUM(total_delay_minutes) OVER ()::numeric, 2) as "Cumulative %",
    ROUND(PERCENT_RANK() OVER (ORDER BY total_delay_minutes DESC)::numeric * 100, 1) as "Percentile"
FROM stop_performance
ORDER BY total_delay_minutes DESC
LIMIT 15;

-- Day-of-Week Pattern Analysis

WITH daily_patterns AS (
    SELECT 
        dd.day_name,
        dd.day_of_week,
        dd.is_weekend,
        COUNT(*) as total_delays,
        AVG(fde.delay_minutes) as avg_delay,
        STDDEV(fde.delay_minutes) as stddev_delay
    FROM warehouse.fact_delay_events fde
    INNER JOIN warehouse.dim_date dd ON fde.date_key = dd.date_key
    GROUP BY dd.day_name, dd.day_of_week, dd.is_weekend
)
SELECT 
    day_name as "Day",
    CASE WHEN is_weekend THEN 'Weekend' ELSE 'Weekday' END as "Type",
    total_delays as "Total Delays",
    ROUND(avg_delay::numeric, 2) as "Avg Delay (min)",
    ROUND(stddev_delay::numeric, 2) as "Std Dev",
    ROUND(LAG(avg_delay) OVER (ORDER BY day_of_week)::numeric, 2) as "Prev Day Avg",
    ROUND((avg_delay - LAG(avg_delay) OVER (ORDER BY day_of_week))::numeric, 2) as "Change from Prev",
    ROUND((avg_delay - AVG(avg_delay) OVER ())::numeric, 2) as "vs Week Avg"
FROM daily_patterns
ORDER BY day_of_week;

-- Time Series - Delay Trend Analysis

WITH monthly_metrics AS (
    SELECT 
        dd.year,
        dd.month,
        dd.month_name,
        COUNT(*) as total_delays,
        AVG(fde.delay_minutes) as avg_delay,
        SUM(fde.delay_minutes) as total_delay_minutes,
        COUNT(DISTINCT fde.date_key) as active_days
    FROM warehouse.fact_delay_events fde
    INNER JOIN warehouse.dim_date dd ON fde.date_key = dd.date_key
    GROUP BY dd.year, dd.month, dd.month_name
)
SELECT 
    year as "Year",
    month as "Month #",
    TRIM(month_name) as "Month",
    total_delays as "Total Delays",
    ROUND(avg_delay::numeric, 2) as "Avg Delay (min)",
    ROUND((total_delays::numeric / active_days), 1) as "Delays per Day",
    -- Trend analysis
    ROUND(LAG(avg_delay, 1) OVER (ORDER BY year, month)::numeric, 2) as "Prev Month",
    ROUND((avg_delay - LAG(avg_delay, 1) OVER (ORDER BY year, month))::numeric, 2) as "MoM Change",
    -- Moving average
    ROUND(AVG(avg_delay) OVER (
        ORDER BY year, month 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    )::numeric, 2) as "3-Month MA",
    -- Quartile classification
    NTILE(4) OVER (ORDER BY avg_delay) as "Quartile (1=Best)"
FROM monthly_metrics
ORDER BY year, month;

-- Route-Stop Delay Hotspot Matrix

WITH route_stop_delays AS (
    SELECT 
        dr.route_short_name,
        ds.stop_name,
        ds.stop_area,
        COUNT(*) as delay_count,
        AVG(fde.delay_minutes) as avg_delay,
        MAX(fde.delay_minutes) as max_delay,
        SUM(CASE WHEN fde.is_severe_delay THEN 1 ELSE 0 END) as severe_count
    FROM warehouse.fact_delay_events fde
    INNER JOIN warehouse.dim_route dr ON fde.route_key = dr.route_key
    INNER JOIN warehouse.dim_stop ds ON fde.stop_key = ds.stop_key
    GROUP BY dr.route_short_name, ds.stop_name, ds.stop_area
    HAVING COUNT(*) >= 10  
),
ranked_delays AS (
    SELECT 
        route_short_name,
        stop_name,
        stop_area,
        delay_count,
        avg_delay,
        max_delay,
        severe_count,
        -- Rank within route
        DENSE_RANK() OVER (
            PARTITION BY route_short_name 
            ORDER BY avg_delay DESC
        ) as route_rank,
        -- Overall system rank
        DENSE_RANK() OVER (ORDER BY avg_delay DESC) as system_rank
    FROM route_stop_delays
)
SELECT 
    route_short_name as "Route",
    stop_name as "Stop",
    stop_area as "Area",
    delay_count as "# Delays",
    ROUND(avg_delay::numeric, 2) as "Avg Delay (min)",
    max_delay as "Max Delay",
    severe_count as "Severe Delays",
    route_rank as "Route Rank",
    system_rank as "System Rank"
FROM ranked_delays
WHERE system_rank <= 20
ORDER BY avg_delay DESC;
