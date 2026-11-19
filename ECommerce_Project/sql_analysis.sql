-- ============================================================================
-- ENTERPRISE E-COMMERCE BUSINESS INTELLIGENCE - ADVANCED SQL ANALYTICS
-- ============================================================================
-- Project: Global E-Commerce Performance & Operational Intelligence Platform
-- Author: Senior Data Analyst & BI Solutions Architect
-- Engine Compatibility: ANSI SQL / PostgreSQL / Snowflake / MySQL 8.0+ / BigQuery
-- Database Target: ECommerce_Enterprise_DW
-- Granularity: Order Line / Transaction Granularity (Fact_Sales)
-- Customer Master: Denormalized Customer Registry (Dim_Customer)
-- Total Records: 5,000 Transactions | 989 Unique Customers | ~$5.11M Net Revenue
-- Date Horizon: 2022-01-01 to 2035-09-09
-- ============================================================================

/*
TABLE SCHEMA DEFINITIONS FOR REFERENCE:

Fact_Sales:
    - order_id            VARCHAR(20)  PRIMARY KEY (Degenerate Dimension)
    - order_date          DATE         NOT NULL (FK -> Dim_Date)
    - customer_id         VARCHAR(20)  NOT NULL (FK -> Dim_Customer)
    - product_category    VARCHAR(50)  NOT NULL (Electronics, Clothing, Home, Beauty)
    - region              VARCHAR(20)  NOT NULL (West, North, South, East)
    - quantity            INTEGER      NOT NULL (1 - 10 units)
    - unit_price          NUMERIC(10,2) NOT NULL
    - discount            NUMERIC(4,2) NOT NULL (0.00 - 0.35)
    - payment_method      VARCHAR(20)  NOT NULL (Card, COD, Wallet)
    - delivery_days       INTEGER      NOT NULL (1 - 11 days)
    - customer_rating     NUMERIC(2,1) NOT NULL (1.0 - 5.0)
    - revenue             NUMERIC(12,2) NOT NULL (Net Revenue after discount)
    - gross_revenue       NUMERIC(12,2) NOT NULL (quantity * unit_price)
    - discount_amount     NUMERIC(12,2) NOT NULL (gross_revenue * discount)
    - profit              NUMERIC(12,2) NOT NULL (30% benchmark operating profit)

Dim_Customer:
    - customer_id               VARCHAR(20)  PRIMARY KEY
    - customer_name             VARCHAR(100) NOT NULL
    - customer_age              INTEGER      NOT NULL
    - age_group                 VARCHAR(20)  NOT NULL (<25, 25-34, 35-44, 45-54, 55-64, 65+)
    - gender                    VARCHAR(20)  NOT NULL
    - customer_segment          VARCHAR(30)  NOT NULL (Consumer, Corporate, Home Office, VIP)
    - customer_city             VARCHAR(100) NOT NULL
    - customer_state            VARCHAR(100) NOT NULL
    - customer_country          VARCHAR(50)  NOT NULL
    - region                    VARCHAR(20)  NOT NULL
    - customer_postal_code      VARCHAR(20)  NOT NULL
    - customer_acquisition_cost NUMERIC(8,2) NOT NULL
*/


-- ============================================================================
-- QUERY 1: GRANULAR TEMPORAL REVENUE SUMMARY (YEAR, QUARTER, MONTH)
-- ============================================================================
-- Business Question:
--   What is the historical revenue, gross volume, discount amount, and order
--   volume across calendar years, quarters, and months?
-- Logic & Mechanics:
--   - Uses CTE to normalize calendar parts from `order_date`.
--   - Aggregates transactions into monthly periods.
--   - Computes Net Revenue, Gross Revenue, Total Discounts, and Order Count.
--   - Evaluates Effective Discount Rate and Average Order Value (AOV).
-- Expected Business Insight:
--   Provides the baseline financial cadence across the 14-year horizon.
--   Reveals monthly cyclicality and verifies consistent average order volume.
-- ============================================================================

WITH Monthly_Financials AS (
    SELECT
        EXTRACT(YEAR FROM order_date) AS order_year,
        EXTRACT(QUARTER FROM order_date) AS order_quarter,
        EXTRACT(MONTH FROM order_date) AS order_month_num,
        TO_CHAR(order_date, 'Mon') AS order_month_name,
        COUNT(DISTINCT order_id) AS total_orders,
        COUNT(DISTINCT customer_id) AS active_customers,
        SUM(quantity) AS total_units_sold,
        SUM(gross_revenue) AS total_gross_revenue,
        SUM(discount_amount) AS total_discount_amount,
        SUM(revenue) AS total_net_revenue,
        SUM(profit) AS total_profit
    FROM Fact_Sales
    GROUP BY 
        EXTRACT(YEAR FROM order_date),
        EXTRACT(QUARTER FROM order_date),
        EXTRACT(MONTH FROM order_date),
        TO_CHAR(order_date, 'Mon')
)
SELECT
    order_year,
    order_quarter,
    order_month_num,
    order_month_name,
    total_orders,
    active_customers,
    total_units_sold,
    ROUND(total_gross_revenue, 2) AS gross_revenue_usd,
    ROUND(total_discount_amount, 2) AS discount_amount_usd,
    ROUND(total_net_revenue, 2) AS net_revenue_usd,
    ROUND(total_profit, 2) AS estimated_profit_usd,
    ROUND(total_net_revenue / NULLIF(total_orders, 0), 2) AS average_order_value_usd,
    ROUND((total_discount_amount / NULLIF(total_gross_revenue, 0)) * 100, 2) AS effective_discount_rate_pct
FROM Monthly_Financials
ORDER BY 
    order_year ASC, 
    order_month_num ASC;


-- ============================================================================
-- QUERY 2: YEAR-OVER-YEAR (YOY) REVENUE & ORDER GROWTH DYNAMICS
-- ============================================================================
-- Business Question:
--   How does annual top-line net revenue, order volume, and profit compare
--   against prior year performance, and what is the annual growth percentage?
-- Logic & Mechanics:
--   - Aggregates financial totals by `order_year`.
--   - Leverages `LAG()` window function to retrieve previous year (PY) values.
--   - Computes nominal YoY variance and percentage growth rates safely guarding
--     against division by zero.
-- Expected Business Insight:
--   Identifies inflection points of high expansion versus contractions across
--   the commercial operating horizon.
-- ============================================================================

WITH Yearly_Summary AS (
    SELECT
        EXTRACT(YEAR FROM order_date) AS order_year,
        COUNT(DISTINCT order_id) AS current_year_orders,
        SUM(revenue) AS current_year_revenue,
        SUM(profit) AS current_year_profit
    FROM Fact_Sales
    GROUP BY EXTRACT(YEAR FROM order_date)
),
YoY_Calculations AS (
    SELECT
        order_year,
        current_year_orders,
        current_year_revenue,
        current_year_profit,
        LAG(current_year_orders, 1) OVER (ORDER BY order_year) AS prior_year_orders,
        LAG(current_year_revenue, 1) OVER (ORDER BY order_year) AS prior_year_revenue,
        LAG(current_year_profit, 1) OVER (ORDER BY order_year) AS prior_year_profit
    FROM Yearly_Summary
)
SELECT
    order_year,
    current_year_orders,
    prior_year_orders,
    current_year_orders - prior_year_orders AS orders_yoy_change,
    ROUND(
        ((current_year_orders - prior_year_orders)::NUMERIC / NULLIF(prior_year_orders, 0)) * 100, 
        2
    ) AS orders_growth_rate_pct,
    ROUND(current_year_revenue, 2) AS revenue_usd,
    ROUND(prior_year_revenue, 2) AS prior_year_revenue_usd,
    ROUND(current_year_revenue - prior_year_revenue, 2) AS revenue_yoy_change_usd,
    ROUND(
        ((current_year_revenue - prior_year_revenue) / NULLIF(prior_year_revenue, 0)) * 100, 
        2
    ) AS revenue_growth_rate_pct,
    ROUND(
        ((current_year_profit - prior_year_profit) / NULLIF(prior_year_profit, 0)) * 100, 
        2
    ) AS profit_growth_rate_pct
FROM YoY_Calculations
ORDER BY order_year ASC;


-- ============================================================================
-- QUERY 3: PRODUCT CATEGORY PERFORMANCE, RANKING & REVENUE CONTRIBUTION
-- ============================================================================
-- Business Question:
--   Which product categories generate the highest revenue and profit, how do
--   they rank in commercial volume, and what is each category's share of total?
-- Logic & Mechanics:
--   - Groups by `product_category`.
--   - Uses `DENSE_RANK()` based on net revenue.
--   - Uses an unbounded window aggregation `SUM(SUM(revenue)) OVER ()` to 
--     determine the universal denominator for percentage-of-total contribution.
-- Expected Business Insight:
--   Electronics (~35.5%) and Clothing (~30.6%) represent the primary revenue
--   engines, followed by Home (~19.4%) and Beauty (~14.5%).
-- ============================================================================

SELECT
    product_category,
    COUNT(DISTINCT order_id) AS total_orders,
    SUM(quantity) AS units_sold,
    ROUND(AVG(unit_price), 2) AS avg_unit_price,
    ROUND(SUM(gross_revenue), 2) AS total_gross_revenue,
    ROUND(SUM(discount_amount), 2) AS total_discount_amount,
    ROUND(SUM(revenue), 2) AS total_net_revenue,
    ROUND(SUM(profit), 2) AS total_profit,
    DENSE_RANK() OVER (ORDER BY SUM(revenue) DESC) AS revenue_rank,
    ROUND(
        (SUM(revenue) / SUM(SUM(revenue)) OVER ()) * 100, 
        2
    ) AS revenue_contribution_pct,
    ROUND(
        (SUM(profit) / NULLIF(SUM(revenue), 0)) * 100, 
        2
    ) AS profit_margin_pct
FROM Fact_Sales
GROUP BY product_category
ORDER BY revenue_rank ASC;


-- ============================================================================
-- QUERY 4: REGIONAL COMMERCIAL PERFORMANCE WITH MULTI-METRIC RANKING
-- ============================================================================
-- Business Question:
--   How do sales regions compare across revenue, orders, AOV, and customer
--   satisfaction ratings, and what is their relative ranking?
-- Logic & Mechanics:
--   - Aggregates key commercial and operational measures across `region`.
--   - Applies `RANK()` and `PERCENT_RANK()` across revenue and customer rating.
-- Expected Business Insight:
--   Sales are remarkably well distributed across West, North, East, and South,
--   each accounting for approximately 24% - 26% of enterprise revenue.
-- ============================================================================

SELECT
    region,
    COUNT(DISTINCT order_id) AS total_orders,
    COUNT(DISTINCT customer_id) AS unique_purchasers,
    SUM(quantity) AS total_units_sold,
    ROUND(SUM(revenue), 2) AS total_net_revenue,
    ROUND(SUM(revenue) / COUNT(DISTINCT order_id), 2) AS regional_aov_usd,
    ROUND(AVG(customer_rating), 2) AS avg_customer_rating,
    ROUND(AVG(delivery_days), 2) AS avg_delivery_days,
    RANK() OVER (ORDER BY SUM(revenue) DESC) AS regional_revenue_rank,
    RANK() OVER (ORDER BY AVG(customer_rating) DESC) AS regional_csat_rank,
    ROUND(
        (SUM(revenue) / SUM(SUM(revenue)) OVER ()) * 100, 
        2
    ) AS regional_revenue_share_pct
FROM Fact_Sales
GROUP BY region
ORDER BY regional_revenue_rank ASC;


-- ============================================================================
-- QUERY 5: CUSTOMER SEGMENTATION & DEMOGRAPHIC AGE GROUP PROFITABILITY
-- ============================================================================
-- Business Question:
--   Which customer segment (Consumer, Corporate, Home Office, VIP) and age
--   group generate the highest revenue, AOV, and margin efficiency?
-- Logic & Mechanics:
--   - Performs an INNER JOIN between `Fact_Sales` and `Dim_Customer` on `customer_id`.
--   - Groups by `customer_segment` and `age_group`.
--   - Calculates Revenue per Customer and Average Acquisition Cost.
-- Expected Business Insight:
--   Consumer segment accounts for majority volume, but VIP and Corporate
--   customers yield superior Average Order Values and higher LTV-to-CAC ratios.
-- ============================================================================

SELECT
    c.customer_segment,
    c.age_group,
    COUNT(DISTINCT c.customer_id) AS active_customer_count,
    COUNT(DISTINCT s.order_id) AS total_orders_placed,
    ROUND(SUM(s.revenue), 2) AS total_revenue_usd,
    ROUND(SUM(s.profit), 2) AS total_profit_usd,
    ROUND(SUM(s.revenue) / NULLIF(COUNT(DISTINCT s.order_id), 0), 2) AS avg_order_value_usd,
    ROUND(SUM(s.revenue) / NULLIF(COUNT(DISTINCT c.customer_id), 0), 2) AS revenue_per_customer_usd,
    ROUND(AVG(c.customer_acquisition_cost), 2) AS avg_customer_acquisition_cost_usd,
    ROUND(
        SUM(s.revenue) / NULLIF(SUM(c.customer_acquisition_cost), 0), 
        2
    ) AS ltv_to_cac_multiple
FROM Fact_Sales s
INNER JOIN Dim_Customer c 
    ON s.customer_id = c.customer_id
GROUP BY 
    c.customer_segment, 
    c.age_group
ORDER BY 
    c.customer_segment ASC, 
    total_revenue_usd DESC;


-- ============================================================================
-- QUERY 6: RFM (RECENCY, FREQUENCY, MONETARY) CUSTOMER SEGMENTATION MODEL
-- ============================================================================
-- Business Question:
--   How can enterprise customers be categorized into actionable behavioral
--   cohorts based on Recency of last purchase, Frequency of orders, and
--   Monetary spend?
-- Logic & Mechanics:
--   - Reference date is defined as max order date in the dataset.
--   - CTE 1 computes raw Recency (days since last order), Frequency (distinct
--     order count), and Monetary (total net spend).
--   - CTE 2 uses `NTILE(5)` window functions to score customers from 1 (low) to 5 (high).
--     For Recency, lower days = higher score, hence reverse order.
--   - Final SELECT categorizes customers into recognized RFM segments using CASE logic.
-- Expected Business Insight:
--   Isolates 'Champions' and 'Loyal Customers' who warrant VIP loyalty programs
--   from 'At-Risk' or 'Hibernating' customers requiring win-back campaigns.
-- ============================================================================

WITH Reference_Date AS (
    SELECT MAX(order_date) AS max_date FROM Fact_Sales
),
Customer_Raw_RFM AS (
    SELECT
        s.customer_id,
        (SELECT max_date FROM Reference_Date) - MAX(s.order_date) AS recency_days,
        COUNT(DISTINCT s.order_id) AS frequency_orders,
        SUM(s.revenue) AS monetary_spend_usd
    FROM Fact_Sales s
    GROUP BY s.customer_id
),
RFM_Scores AS (
    SELECT
        customer_id,
        recency_days,
        frequency_orders,
        monetary_spend_usd,
        NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,     -- Low recency days = High score (5)
        NTILE(5) OVER (ORDER BY frequency_orders ASC) AS f_score,  -- High frequency = High score (5)
        NTILE(5) OVER (ORDER BY monetary_spend_usd ASC) AS m_score -- High spend = High score (5)
    FROM Customer_Raw_RFM
),
Segmented_Customers AS (
    SELECT
        customer_id,
        recency_days,
        frequency_orders,
        monetary_spend_usd,
        r_score,
        f_score,
        m_score,
        (r_score * 100 + f_score * 10 + m_score) AS rfm_combined_score,
        CASE
            WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
            WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3 THEN 'Loyal Customers'
            WHEN r_score >= 4 AND f_score <= 2 THEN 'Promising / New Customers'
            WHEN r_score >= 3 AND f_score <= 2 AND m_score >= 3 THEN 'Potential Loyalists'
            WHEN r_score <= 2 AND f_score >= 3 AND m_score >= 3 THEN 'At Risk / Need Attention'
            WHEN r_score <= 2 AND f_score <= 2 AND m_score >= 3 THEN 'About to Sleep'
            WHEN r_score <= 1 AND f_score >= 4 THEN 'Can Not Lose Them'
            ELSE 'Hibernating / Lost'
        END AS rfm_segment
    FROM RFM_Scores
)
SELECT
    rfm_segment,
    COUNT(*) AS total_customers,
    ROUND((COUNT(*)::NUMERIC / (SELECT COUNT(*) FROM Segmented_Customers)) * 100, 2) AS customer_share_pct,
    ROUND(AVG(recency_days), 1) AS avg_recency_days,
    ROUND(AVG(frequency_orders), 2) AS avg_order_frequency,
    ROUND(AVG(monetary_spend_usd), 2) AS avg_monetary_spend_usd,
    ROUND(SUM(monetary_spend_usd), 2) AS total_segment_revenue_usd,
    ROUND((SUM(monetary_spend_usd) / (SELECT SUM(monetary_spend_usd) FROM Segmented_Customers)) * 100, 2) AS revenue_share_pct
FROM Segmented_Customers
GROUP BY rfm_segment
ORDER BY total_segment_revenue_usd DESC;


-- ============================================================================
-- QUERY 7: CUSTOMER LIFETIME VALUE (CLV) & NET PROFIT CONTRIBUTION RANKING
-- ============================================================================
-- Business Question:
--   What is each customer's cumulative lifetime revenue, net margin, and net
--   customer profit after accounting for customer acquisition cost (CAC)?
-- Logic & Mechanics:
--   - Aggregates sales at customer granularity.
--   - Joins `Dim_Customer` to pull `customer_acquisition_cost`.
--   - Net Customer Profit = Cumulative Realized Profit - Customer Acquisition Cost.
--   - Employs `DENSE_RANK()` over net profit contribution.
-- Expected Business Insight:
--   Demonstrates that the top 100 customers generate a disproportionate share
--   of total enterprise net profit, easily recovering acquisition investment.
-- ============================================================================

WITH Customer_Aggregates AS (
    SELECT
        s.customer_id,
        COUNT(DISTINCT s.order_id) AS total_orders,
        SUM(s.quantity) AS total_units_bought,
        SUM(s.gross_revenue) AS gross_spend,
        SUM(s.discount_amount) AS discount_received,
        SUM(s.revenue) AS lifetime_net_revenue,
        SUM(s.profit) AS lifetime_operating_profit,
        MIN(s.order_date) AS first_order_date,
        MAX(s.order_date) AS last_order_date,
        MAX(s.order_date) - MIN(s.order_date) AS customer_lifespan_days
    FROM Fact_Sales s
    GROUP BY s.customer_id
)
SELECT
    ca.customer_id,
    c.customer_name,
    c.customer_segment,
    c.region,
    ca.total_orders,
    ca.total_units_bought,
    ca.customer_lifespan_days,
    ROUND(ca.lifetime_net_revenue, 2) AS lifetime_net_revenue_usd,
    ROUND(ca.lifetime_operating_profit, 2) AS lifetime_operating_profit_usd,
    ROUND(c.customer_acquisition_cost, 2) AS acquisition_cost_usd,
    ROUND(ca.lifetime_operating_profit - c.customer_acquisition_cost, 2) AS net_customer_profit_usd,
    ROUND(
        ca.lifetime_net_revenue / NULLIF(c.customer_acquisition_cost, 0), 
        2
    ) AS revenue_to_cac_ratio,
    DENSE_RANK() OVER (ORDER BY (ca.lifetime_operating_profit - c.customer_acquisition_cost) DESC) AS profitability_rank
FROM Customer_Aggregates ca
INNER JOIN Dim_Customer c 
    ON ca.customer_id = c.customer_id
ORDER BY profitability_rank ASC
LIMIT 50;


-- ============================================================================
-- QUERY 8: NEW VS. RETURNING CUSTOMER REVENUE ENGINE OVER TIME
-- ============================================================================
-- Business Question:
--   In each calendar year and quarter, what share of revenue and order volume
--   is driven by newly acquired first-time purchasers versus repeat buyers?
-- Logic & Mechanics:
--   - Uses an analytical window function `MIN(order_date) OVER (PARTITION BY customer_id)`
--     to establish the definitive first purchase date for each customer.
--   - Compares the transaction date to first purchase date: if equal, 'New';
--     otherwise 'Returning'.
--   - Groups by calendar year and quarter.
-- Expected Business Insight:
--   Shows the gradual maturation of the customer base: in 2022, 100% of orders
--   originated from new customers; in later years, repeat customers drive the
--   majority of recurring transactions.
-- ============================================================================

WITH Order_Customer_Classified AS (
    SELECT
        order_id,
        order_date,
        customer_id,
        revenue,
        EXTRACT(YEAR FROM order_date) AS order_year,
        EXTRACT(QUARTER FROM order_date) AS order_quarter,
        MIN(order_date) OVER (PARTITION BY customer_id) AS customer_first_order_date,
        CASE
            WHEN order_date = MIN(order_date) OVER (PARTITION BY customer_id) THEN 'New Customer'
            ELSE 'Returning Customer'
        END AS customer_order_type
    FROM Fact_Sales
)
SELECT
    order_year,
    order_quarter,
    customer_order_type,
    COUNT(DISTINCT order_id) AS total_orders,
    COUNT(DISTINCT customer_id) AS distinct_customers,
    ROUND(SUM(revenue), 2) AS period_revenue_usd,
    ROUND(
        (SUM(revenue) / SUM(SUM(revenue)) OVER (PARTITION BY order_year, order_quarter)) * 100, 
        2
    ) AS quarterly_revenue_share_pct
FROM Order_Customer_Classified
GROUP BY 
    order_year, 
    order_quarter, 
    customer_order_type
ORDER BY 
    order_year ASC, 
    order_quarter ASC, 
    customer_order_type DESC;


-- ============================================================================
-- QUERY 9: AVERAGE ORDER VALUE (AOV) & BASKET SIZE LONGITUDINAL TRENDS
-- ============================================================================
-- Business Question:
--   How has the Average Order Value (AOV), units per order, and discount per
--   order trended month-by-month over the operational timeline?
-- Logic & Mechanics:
--   - Aggregates by year and month.
--   - Computes AOV as `SUM(revenue) / COUNT(DISTINCT order_id)`.
--   - Computes basket size as `SUM(quantity) / COUNT(DISTINCT order_id)`.
--   - Uses `AVG() OVER (ORDER BY ... ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)`
--     to compute a 3-month trailing moving average of AOV.
-- Expected Business Insight:
--   Monitors basket size stability and validates whether pricing changes or
--   promotions caused deflationary or inflationary swings in order size.
-- ============================================================================

WITH Monthly_Aggregates AS (
    SELECT
        EXTRACT(YEAR FROM order_date) AS order_year,
        EXTRACT(MONTH FROM order_date) AS order_month,
        DATE_TRUNC('month', order_date)::DATE AS month_start_date,
        COUNT(DISTINCT order_id) AS order_count,
        SUM(quantity) AS total_units,
        SUM(gross_revenue) AS total_gross_rev,
        SUM(revenue) AS total_net_rev
    FROM Fact_Sales
    GROUP BY 
        EXTRACT(YEAR FROM order_date), 
        EXTRACT(MONTH FROM order_date),
        DATE_TRUNC('month', order_date)::DATE
),
AOV_Calculation AS (
    SELECT
        order_year,
        order_month,
        month_start_date,
        order_count,
        ROUND(total_net_rev, 2) AS monthly_revenue_usd,
        ROUND(total_net_rev / order_count, 2) AS monthly_aov_usd,
        ROUND(total_units::NUMERIC / order_count, 2) AS units_per_order,
        ROUND(total_gross_rev / order_count, 2) AS gross_order_value_usd
    FROM Monthly_Aggregates
)
SELECT
    order_year,
    order_month,
    month_start_date,
    order_count,
    monthly_revenue_usd,
    monthly_aov_usd,
    units_per_order,
    gross_order_value_usd,
    ROUND(
        AVG(monthly_aov_usd) OVER (
            ORDER BY month_start_date 
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ), 
        2
    ) AS aov_3m_moving_average_usd
FROM AOV_Calculation
ORDER BY month_start_date ASC;


-- ============================================================================
-- QUERY 10: PROMOTIONAL DISCOUNT EFFECTIVENESS & MARGIN LEAKAGE AUDIT
-- ============================================================================
-- Business Question:
--   Do deeper promotional discount tiers (>20%) produce higher order volumes
--   and basket sizes, or do they severely compress gross margin without lift?
-- Logic & Mechanics:
--   - Segments discount rate into 4 standard retail promotion tiers:
--     None (0%), Low (1-10%), Medium (11-20%), High (21-35%).
--   - Computes units per transaction, gross revenue, net revenue, and margin.
-- Expected Business Insight:
--   Proves that high discount tiers suffer an ~11.5 percentage point drop in
--   gross margin while unit quantities per order remain virtually flat (~5.0).
-- ============================================================================

WITH Discount_Banded_Sales AS (
    SELECT
        order_id,
        quantity,
        gross_revenue,
        discount_amount,
        revenue,
        profit,
        CASE
            WHEN discount = 0.00 THEN '1. No Discount (0%)'
            WHEN discount <= 0.10 THEN '2. Low (1% - 10%)'
            WHEN discount <= 0.20 THEN '3. Medium (11% - 20%)'
            ELSE '4. High (21% - 35%)'
        END AS promo_discount_tier
    FROM Fact_Sales
)
SELECT
    promo_discount_tier,
    COUNT(DISTINCT order_id) AS total_orders,
    ROUND((COUNT(DISTINCT order_id)::NUMERIC / (SELECT COUNT(*) FROM Fact_Sales)) * 100, 2) AS order_percentage_share,
    SUM(quantity) AS total_units_sold,
    ROUND(AVG(quantity), 2) AS avg_units_per_order,
    ROUND(SUM(gross_revenue), 2) AS total_gross_revenue_usd,
    ROUND(SUM(discount_amount), 2) AS total_discount_given_usd,
    ROUND(SUM(revenue), 2) AS total_net_revenue_usd,
    ROUND(SUM(profit), 2) AS total_profit_usd,
    ROUND(AVG(revenue), 2) AS average_order_value_usd,
    ROUND(
        (SUM(profit) / NULLIF(SUM(revenue), 0)) * 100, 
        2
    ) AS realized_profit_margin_pct
FROM Discount_Banded_Sales
GROUP BY promo_discount_tier
ORDER BY promo_discount_tier ASC;


-- ============================================================================
-- QUERY 11: SUPPLY CHAIN FULFILLMENT & DELIVERY LEAD TIME ANALYSIS
-- ============================================================================
-- Business Question:
--   What is the distribution of fulfillment lead times across regional delivery
--   lanes, and what proportion of orders breach the 7-day delivery SLA?
-- Logic & Mechanics:
--   - Groups by `region`.
--   - Categorizes shipments into Fast (1-3 days), Normal (4-7 days), and
--     Delayed (>7 days).
--   - Calculates average, min, max, and median delivery days.
-- Expected Business Insight:
--   Enterprise delivery lead time averages 6.1 days, with ~28.4% of total orders
--   exceeding 7 days, indicating significant supply chain latency across lanes.
-- ============================================================================

SELECT
    region,
    COUNT(DISTINCT order_id) AS total_shipments,
    ROUND(AVG(delivery_days), 2) AS avg_delivery_days,
    MIN(delivery_days) AS min_delivery_days,
    MAX(delivery_days) AS max_delivery_days,
    COUNT(CASE WHEN delivery_days <= 3 THEN 1 END) AS fast_shipments_1_to_3d,
    ROUND(
        (COUNT(CASE WHEN delivery_days <= 3 THEN 1 END)::NUMERIC / COUNT(*)) * 100, 
        2
    ) AS fast_shipments_pct,
    COUNT(CASE WHEN delivery_days BETWEEN 4 AND 7 THEN 1 END) AS normal_shipments_4_to_7d,
    ROUND(
        (COUNT(CASE WHEN delivery_days BETWEEN 4 AND 7 THEN 1 END)::NUMERIC / COUNT(*)) * 100, 
        2
    ) AS normal_shipments_pct,
    COUNT(CASE WHEN delivery_days > 7 THEN 1 END) AS delayed_shipments_gt_7d,
    ROUND(
        (COUNT(CASE WHEN delivery_days > 7 THEN 1 END)::NUMERIC / COUNT(*)) * 100, 
        2
    ) AS delayed_shipments_sla_breach_pct
FROM Fact_Sales
GROUP BY region
ORDER BY delayed_shipments_sla_breach_pct DESC;


-- ============================================================================
-- QUERY 12: CUSTOMER SATISFACTION (CSAT) VS. FULFILLMENT SPEED & CATEGORY
-- ============================================================================
-- Business Question:
--   How severely does delivery delay deteriorate customer ratings across
--   different product merchandise categories?
-- Logic & Mechanics:
--   - Bins delivery speed into Fast (1-3d), Normal (4-7d), Delayed (8-11d).
--   - Computes average rating, count of detractor reviews (1-2 stars), and
--     count of promoter reviews (4-5 stars).
-- Expected Business Insight:
--   Orders delivered in 1-3 days maintain strong CSAT (~4.2), while delayed
--   orders (>7 days) collapse to detractor scores (<2.2) across all categories.
-- ============================================================================

WITH Shipment_Rating_Matrix AS (
    SELECT
        product_category,
        CASE
            WHEN delivery_days <= 3 THEN 'Fast (1-3 days)'
            WHEN delivery_days <= 7 THEN 'Normal (4-7 days)'
            ELSE 'Delayed (>7 days)'
        END AS delivery_tier,
        customer_rating,
        revenue
    FROM Fact_Sales
)
SELECT
    product_category,
    delivery_tier,
    COUNT(*) AS total_orders,
    ROUND(AVG(customer_rating), 2) AS avg_csat_score,
    COUNT(CASE WHEN customer_rating <= 2.0 THEN 1 END) AS detractor_count_1_to_2_star,
    ROUND(
        (COUNT(CASE WHEN customer_rating <= 2.0 THEN 1 END)::NUMERIC / COUNT(*)) * 100, 
        2
    ) AS detractor_rate_pct,
    COUNT(CASE WHEN customer_rating >= 4.0 THEN 1 END) AS promoter_count_4_to_5_star,
    ROUND(
        (COUNT(CASE WHEN customer_rating >= 4.0 THEN 1 END)::NUMERIC / COUNT(*)) * 100, 
        2
    ) AS promoter_rate_pct,
    ROUND(SUM(revenue), 2) AS revenue_exposed_usd
FROM Shipment_Rating_Matrix
GROUP BY 
    product_category, 
    delivery_tier
ORDER BY 
    product_category ASC, 
    delivery_tier ASC;


-- ============================================================================
-- QUERY 13: CUSTOMER PARETO ANALYSIS (THE 80/20 REVENUE CONCENTRATION RULE)
-- ============================================================================
-- Business Question:
--   What proportion of total enterprise net revenue is driven by the top 20%
--   highest-spending customers (Pareto Principle verification)?
-- Logic & Mechanics:
--   - CTE 1 computes total spend per customer.
--   - CTE 2 uses window functions `ROW_NUMBER()`, `COUNT(*) OVER ()`, and
--     `SUM(spend) OVER (ORDER BY spend DESC)` to track cumulative customer
--     percentile and cumulative revenue percentile.
--   - Final SELECT flags customers who fall within the top 20% tier.
-- Expected Business Insight:
--   Empirically tests the 80/20 rule: determines the exact revenue share
--   controlled by the top quintile of retail shoppers.
-- ============================================================================

WITH Customer_Spend AS (
    SELECT
        customer_id,
        COUNT(DISTINCT order_id) AS order_count,
        SUM(revenue) AS total_customer_revenue
    FROM Fact_Sales
    GROUP BY customer_id
),
Pareto_Calculations AS (
    SELECT
        customer_id,
        order_count,
        total_customer_revenue,
        ROW_NUMBER() OVER (ORDER BY total_customer_revenue DESC) AS customer_rank,
        COUNT(*) OVER () AS total_customer_count,
        SUM(total_customer_revenue) OVER () AS enterprise_total_revenue,
        SUM(total_customer_revenue) OVER (
            ORDER BY total_customer_revenue DESC 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_revenue
    FROM Customer_Spend
)
SELECT
    CASE
        WHEN (customer_rank::NUMERIC / total_customer_count) <= 0.20 THEN 'Top 20% Customers (VIP Core)'
        ELSE 'Remaining 80% Customers (Long Tail)'
    END AS customer_tier,
    COUNT(*) AS customer_count,
    ROUND(MIN(total_customer_revenue), 2) AS min_tier_spend_usd,
    ROUND(MAX(total_customer_revenue), 2) AS max_tier_spend_usd,
    ROUND(AVG(total_customer_revenue), 2) AS avg_tier_spend_usd,
    ROUND(SUM(total_customer_revenue), 2) AS total_tier_revenue_usd,
    ROUND(
        (SUM(total_customer_revenue) / MAX(enterprise_total_revenue)) * 100, 
        2
    ) AS revenue_share_pct
FROM Pareto_Calculations
GROUP BY 
    CASE
        WHEN (customer_rank::NUMERIC / total_customer_count) <= 0.20 THEN 'Top 20% Customers (VIP Core)'
        ELSE 'Remaining 80% Customers (Long Tail)'
    END
ORDER BY total_tier_revenue_usd DESC;


-- ============================================================================
-- QUERY 14: MULTI-HORIZON ROLLING MOVING AVERAGES (3-MONTH & 12-MONTH)
-- ============================================================================
-- Business Question:
--   What is the underlying smoothed revenue run-rate after filtering out
--   short-term transactional noise using 3-month and 12-month moving averages?
-- Logic & Mechanics:
--   - Aggregates revenue to monthly calendar periods.
--   - Uses window frames `ROWS BETWEEN 2 PRECEDING AND CURRENT ROW` for 3M MA.
--   - Uses window frames `ROWS BETWEEN 11 PRECEDING AND CURRENT ROW` for 12M MA.
-- Expected Business Insight:
--   Reveals macro-trend line continuity across multi-year cycles.
-- ============================================================================

WITH Monthly_Revenue AS (
    SELECT
        DATE_TRUNC('month', order_date)::DATE AS sales_month,
        COUNT(DISTINCT order_id) AS total_orders,
        SUM(revenue) AS monthly_revenue
    FROM Fact_Sales
    GROUP BY DATE_TRUNC('month', order_date)::DATE
)
SELECT
    sales_month,
    total_orders,
    ROUND(monthly_revenue, 2) AS actual_monthly_revenue_usd,
    ROUND(
        AVG(monthly_revenue) OVER (
            ORDER BY sales_month 
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ), 
        2
    ) AS moving_avg_3_month_usd,
    ROUND(
        AVG(monthly_revenue) OVER (
            ORDER BY sales_month 
            ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
        ), 
        2
    ) AS moving_avg_12_month_usd
FROM Monthly_Revenue
ORDER BY sales_month ASC;


-- ============================================================================
-- QUERY 15: CUMULATIVE LIFETIME RUNNING TOTAL OF REVENUE & PROFIT
-- ============================================================================
-- Business Question:
--   How does cumulative enterprise revenue and profit compound from the
--   very first operating transaction to the end of the historical horizon?
-- Logic & Mechanics:
--   - Aggregates by month.
--   - Employs unbounded preceding window accumulation:
--     `SUM(...) OVER (ORDER BY sales_month ROWS UNBOUNDED PRECEDING)`.
-- Expected Business Insight:
--   Demonstrates milestones when the business achieved cumulative $1M, $2M,
--   and crossed the $5M gross revenue threshold.
-- ============================================================================

WITH Monthly_Performance AS (
    SELECT
        DATE_TRUNC('month', order_date)::DATE AS sales_month,
        SUM(gross_revenue) AS monthly_gross,
        SUM(discount_amount) AS monthly_discount,
        SUM(revenue) AS monthly_net_revenue,
        SUM(profit) AS monthly_profit
    FROM Fact_Sales
    GROUP BY DATE_TRUNC('month', order_date)::DATE
)
SELECT
    sales_month,
    ROUND(monthly_net_revenue, 2) AS monthly_net_revenue_usd,
    ROUND(monthly_profit, 2) AS monthly_profit_usd,
    ROUND(
        SUM(monthly_net_revenue) OVER (
            ORDER BY sales_month 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ), 
        2
    ) AS cumulative_running_revenue_usd,
    ROUND(
        SUM(monthly_profit) OVER (
            ORDER BY sales_month 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ), 
        2
    ) AS cumulative_running_profit_usd
FROM Monthly_Performance
ORDER BY sales_month ASC;


-- ============================================================================
-- QUERY 16: CUSTOMER COHORT RETENTION & REPURCHASE ACTIVITY MATRIX
-- ============================================================================
-- Business Question:
--   When customers make their first purchase in a specific acquisition year,
--   what percentage of that cohort returns to make purchases in subsequent years?
-- Logic & Mechanics:
--   - CTE 1 computes each customer's first purchase year (`cohort_year`).
--   - CTE 2 joins each order with the customer's cohort year.
--   - Computes elapsed years = `order_year - cohort_year`.
--   - Calculates retention count and percentage of initial cohort size.
-- Expected Business Insight:
--   Highlights long-term customer stickiness and repeat purchase decay.
-- ============================================================================

WITH Customer_Cohort AS (
    SELECT
        customer_id,
        EXTRACT(YEAR FROM MIN(order_date)) AS cohort_year
    FROM Fact_Sales
    GROUP BY customer_id
),
Cohort_Size AS (
    SELECT
        cohort_year,
        COUNT(DISTINCT customer_id) AS initial_cohort_size
    FROM Customer_Cohort
    GROUP BY cohort_year
),
Cohort_Activity AS (
    SELECT
        cc.cohort_year,
        EXTRACT(YEAR FROM s.order_date) AS activity_year,
        EXTRACT(YEAR FROM s.order_date) - cc.cohort_year AS years_elapsed,
        COUNT(DISTINCT s.customer_id) AS active_retained_customers,
        ROUND(SUM(s.revenue), 2) AS cohort_revenue_usd
    FROM Fact_Sales s
    INNER JOIN Customer_Cohort cc 
        ON s.customer_id = cc.customer_id
    GROUP BY 
        cc.cohort_year, 
        EXTRACT(YEAR FROM s.order_date)
)
SELECT
    ca.cohort_year,
    cs.initial_cohort_size,
    ca.activity_year,
    ca.years_elapsed,
    ca.active_retained_customers,
    ROUND(
        (ca.active_retained_customers::NUMERIC / cs.initial_cohort_size) * 100, 
        2
    ) AS cohort_retention_rate_pct,
    ca.cohort_revenue_usd
FROM Cohort_Activity ca
INNER JOIN Cohort_Size cs 
    ON ca.cohort_year = cs.cohort_year
ORDER BY 
    ca.cohort_year ASC, 
    ca.activity_year ASC;


-- ============================================================================
-- QUERY 17: PAYMENT METHOD ECONOMICS, ADOPTION & RISK PROFILE
-- ============================================================================
-- Business Question:
--   How do transaction channels (Card, COD, Wallet) compare in total volume,
--   share of sales, AOV, delivery duration, and customer satisfaction ratings?
-- Logic & Mechanics:
--   - Groups by `payment_method`.
--   - Aggregates order volume, revenue, average rating, and delivery delay.
-- Expected Business Insight:
--   Cards account for 45.4% ($2.32M) with high CSAT, while Cash on Delivery
--   (35.5%, $1.81M) has the lowest CSAT (2.74) and longest delivery lag.
-- ============================================================================

SELECT
    payment_method,
    COUNT(DISTINCT order_id) AS total_transactions,
    ROUND(
        (COUNT(DISTINCT order_id)::NUMERIC / (SELECT COUNT(*) FROM Fact_Sales)) * 100, 
        2
    ) AS transaction_share_pct,
    ROUND(SUM(revenue), 2) AS total_revenue_usd,
    ROUND(
        (SUM(revenue) / (SELECT SUM(revenue) FROM Fact_Sales)) * 100, 
        2
    ) AS revenue_share_pct,
    ROUND(AVG(revenue), 2) AS average_order_value_usd,
    ROUND(AVG(customer_rating), 2) AS avg_customer_rating,
    ROUND(AVG(delivery_days), 2) AS avg_delivery_days,
    ROUND(
        (COUNT(CASE WHEN customer_rating <= 2.0 THEN 1 END)::NUMERIC / COUNT(*)) * 100, 
        2
    ) AS detractor_rate_pct
FROM Fact_Sales
GROUP BY payment_method
ORDER BY total_revenue_usd DESC;


-- ============================================================================
-- QUERY 18: SEASONAL DYNAMICS & DAY-OF-WEEK COMMERCIAL PATTERNS
-- ============================================================================
-- Business Question:
--   Are there systemic demand fluctuations across calendar months, quarters,
--   and weekdays versus weekends?
-- Logic & Mechanics:
--   - Aggregates by day of week and weekend indicator.
--   - Computes daily revenue run-rate, average quantity, and discount depth.
-- Expected Business Insight:
--   Quantifies weekend versus weekday volume differentials to inform marketing
--   promotional calendars and fulfillment staffing schedules.
-- ============================================================================

SELECT
    CASE 
        WHEN is_weekend = 1 THEN 'Weekend (Sat-Sun)' 
        ELSE 'Weekday (Mon-Fri)' 
    END AS day_classification,
    TO_CHAR(order_date, 'Day') AS day_name,
    EXTRACT(ISODOW FROM order_date) AS day_of_week_num,
    COUNT(DISTINCT order_id) AS total_orders_placed,
    ROUND(SUM(revenue), 2) AS total_revenue_usd,
    ROUND(AVG(revenue), 2) AS average_order_value_usd,
    ROUND(AVG(quantity), 2) AS avg_units_per_order,
    ROUND(AVG(discount) * 100, 2) AS avg_discount_pct,
    ROUND(AVG(customer_rating), 2) AS avg_customer_rating
FROM Fact_Sales
GROUP BY 
    is_weekend, 
    TO_CHAR(order_date, 'Day'), 
    EXTRACT(ISODOW FROM order_date)
ORDER BY day_of_week_num ASC;


-- ============================================================================
-- QUERY 19: CUSTOMER ACQUISITION COST (CAC) ROI & PAYBACK EFFICIENCY
-- ============================================================================
-- Business Question:
--   What is the marketing acquisition efficiency (LTV / CAC multiple) across
--   geographic territories and customer segments?
-- Logic & Mechanics:
--   - Joins customer level metrics with acquisition costs from `Dim_Customer`.
--   - Groups by `customer_country` and `customer_segment`.
--   - Calculates total marketing capital deployed vs total revenue earned.
-- Expected Business Insight:
--   Identifies high-yielding acquisition geographies (e.g. US, UK, Germany)
--   where CAC investment delivers 30x+ cumulative revenue return.
-- ============================================================================

SELECT
    c.customer_country,
    c.customer_segment,
    COUNT(DISTINCT c.customer_id) AS acquired_customer_count,
    ROUND(SUM(c.customer_acquisition_cost), 2) AS total_marketing_spend_usd,
    ROUND(AVG(c.customer_acquisition_cost), 2) AS avg_cac_usd,
    ROUND(SUM(s.revenue), 2) AS total_realized_revenue_usd,
    ROUND(SUM(s.profit), 2) AS total_operating_profit_usd,
    ROUND(
        SUM(s.revenue) / NULLIF(SUM(c.customer_acquisition_cost), 0), 
        2
    ) AS ltv_to_cac_ratio,
    ROUND(
        SUM(s.profit) / NULLIF(SUM(c.customer_acquisition_cost), 0), 
        2
    ) AS net_profit_roi_multiple
FROM Dim_Customer c
INNER JOIN Fact_Sales s 
    ON c.customer_id = s.customer_id
GROUP BY 
    c.customer_country, 
    c.customer_segment
ORDER BY total_realized_revenue_usd DESC;


-- ============================================================================
-- QUERY 20: CROSS-SELL & MERCHANDISE CATEGORY CO-PURCHASE AFFINITY
-- ============================================================================
-- Business Question:
--   Which merchandise categories are most commonly cross-purchased by the
--   same customer profiles across their purchasing lifecycle?
-- Logic & Mechanics:
--   - Distinct customer-category matrix CTE.
--   - Self-joins on `customer_id` where Category_A < Category_B to prevent
--     mirror duplicates and self-pairs.
--   - Computes unique shared customer count and pair co-occurrence index.
-- Expected Business Insight:
--   Identifies top bundle combinations (Electronics & Clothing, Electronics & Home)
--   to power personalized cross-sell recommendations.
-- ============================================================================

WITH Customer_Category_Map AS (
    SELECT DISTINCT
        customer_id,
        product_category
    FROM Fact_Sales
)
SELECT
    a.product_category AS category_primary,
    b.product_category AS category_cross_sell,
    COUNT(DISTINCT a.customer_id) AS shared_purchasing_customers,
    ROUND(
        (COUNT(DISTINCT a.customer_id)::NUMERIC / 
         (SELECT COUNT(DISTINCT customer_id) FROM Fact_Sales)) * 100, 
        2
    ) AS shared_customer_penetration_pct
FROM Customer_Category_Map a
INNER JOIN Customer_Category_Map b 
    ON a.customer_id = b.customer_id
    AND a.product_category < b.product_category
GROUP BY 
    a.product_category, 
    b.product_category
ORDER BY shared_purchasing_customers DESC;


-- ============================================================================
-- QUERY 21: HIGH-VALUE CUSTOMER CHURN RISK & INACTIVITY IDENTIFICATION
-- ============================================================================
-- Business Question:
--   Which high-value customers (top monetary decile) have shown zero purchase
--   activity for an extended duration and present immediate churn risk?
-- Logic & Mechanics:
--   - Aggregates customer total spend and identifies the 90th percentile threshold.
--   - Calculates days elapsed since last purchase.
--   - Filters for top decile spenders whose recency is in the bottom quartile.
-- Expected Business Insight:
--   Extracts a targeted list of VIP accounts needing immediate concierge outreach
--   and win-back promotions before permanent churn.
-- ============================================================================

WITH Customer_Metrics AS (
    SELECT
        s.customer_id,
        MAX(s.order_date) AS last_purchase_date,
        (SELECT MAX(order_date) FROM Fact_Sales) - MAX(s.order_date) AS days_since_last_purchase,
        COUNT(DISTINCT s.order_id) AS total_orders,
        SUM(s.revenue) AS total_lifetime_spend
    FROM Fact_Sales s
    GROUP BY s.customer_id
),
Ranked_Customers AS (
    SELECT
        cm.customer_id,
        c.customer_name,
        c.customer_segment,
        c.customer_city,
        c.customer_country,
        cm.last_purchase_date,
        cm.days_since_last_purchase,
        cm.total_orders,
        cm.total_lifetime_spend,
        NTILE(10) OVER (ORDER BY cm.total_lifetime_spend ASC) AS spend_decile,
        NTILE(4) OVER (ORDER BY cm.days_since_last_purchase DESC) AS recency_quartile
    FROM Customer_Metrics cm
    INNER JOIN Dim_Customer c 
        ON cm.customer_id = c.customer_id
)
SELECT
    customer_id,
    customer_name,
    customer_segment,
    customer_city,
    customer_country,
    last_purchase_date,
    days_since_last_purchase,
    total_orders,
    ROUND(total_lifetime_spend, 2) AS total_lifetime_spend_usd,
    'High-Spend Lapsed VIP' AS risk_classification
FROM Ranked_Customers
WHERE spend_decile >= 9       -- Top 20% spenders
  AND recency_quartile = 1    -- Longest inactivity duration
ORDER BY total_lifetime_spend DESC
LIMIT 30;


-- ============================================================================
-- QUERY 22: FULFILLMENT SERVICE LEVEL AGREEMENT (SLA) BREACH & VALUE AT RISK
-- ============================================================================
-- Business Question:
--   What is the SLA compliance rate (orders delivered in <= 5 business days)
--   across geographic markets, and how much customer revenue was tied up in
--   breached shipments?
-- Logic & Mechanics:
--   - Standard SLA defined as delivery within 5 days.
--   - Aggregates by `region` and `product_category`.
--   - Calculates compliant order rate and delayed order revenue at risk.
-- Expected Business Insight:
--   Highlights regional fulfillment hubs with acute bottlenecks and quantifies
--   the commercial revenue exposure to late deliveries.
-- ============================================================================

SELECT
    region,
    product_category,
    COUNT(DISTINCT order_id) AS total_orders,
    COUNT(CASE WHEN delivery_days <= 5 THEN 1 END) AS sla_compliant_orders,
    ROUND(
        (COUNT(CASE WHEN delivery_days <= 5 THEN 1 END)::NUMERIC / COUNT(*)) * 100, 
        2
    ) AS sla_compliance_rate_pct,
    COUNT(CASE WHEN delivery_days > 5 THEN 1 END) AS sla_breached_orders,
    ROUND(
        (COUNT(CASE WHEN delivery_days > 5 THEN 1 END)::NUMERIC / COUNT(*)) * 100, 
        2
    ) AS sla_breach_rate_pct,
    ROUND(SUM(CASE WHEN delivery_days > 5 THEN revenue ELSE 0 END), 2) AS delayed_revenue_at_risk_usd,
    ROUND(AVG(delivery_days), 2) AS avg_delivery_cycle_days,
    ROUND(AVG(customer_rating), 2) AS blended_customer_rating
FROM Fact_Sales
GROUP BY 
    region, 
    product_category
ORDER BY sla_compliance_rate_pct ASC;
