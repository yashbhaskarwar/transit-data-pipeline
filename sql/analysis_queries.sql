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