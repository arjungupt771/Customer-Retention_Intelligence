-- ============================================================
-- TELECOM CUSTOMER CHURN ANALYSIS — COMPLETE SQL SCRIPT
-- Dataset: telco_churn (customer snapshot; verify the row count against the loaded file)
-- Author: [Your Name] | Portfolio Project
-- Tools: SQLite / PostgreSQL / MySQL compatible
-- ============================================================

-- ────────────────────────────────────────────────────────────
-- SECTION 1: KPI SUMMARY — Core Business Metrics
-- ────────────────────────────────────────────────────────────

SELECT
    COUNT(*)                                                          AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)                   AS churned_customers,
    COUNT(*) - SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)        AS active_customers,
    COALESCE(ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / NULLIF(COUNT(*), 0), 2), 0)                                 AS churn_rate_pct,
    COALESCE(ROUND(
        (COUNT(*) - SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)) * 100.0
        / NULLIF(COUNT(*), 0), 2), 0)                                 AS retention_rate_pct,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2)
                                                                      AS monthly_revenue_lost,
    ROUND(SUM(MonthlyCharges), 2)                                     AS total_monthly_revenue,
    COALESCE(ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END) * 100.0
        / NULLIF(SUM(MonthlyCharges), 0), 2), 0)                      AS revenue_loss_pct
FROM telco_churn;


-- ────────────────────────────────────────────────────────────
-- SECTION 2: CONTRACT TYPE ANALYSIS
-- CTE + RANK window function
-- ────────────────────────────────────────────────────────────

WITH contract_metrics AS (
    SELECT
        Contract,
        COUNT(*)                                                      AS total_customers,
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)               AS churned,
        ROUND(
            SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0
            / COUNT(*), 2)                                            AS churn_rate,
        ROUND(AVG(MonthlyCharges), 2)                                 AS avg_monthly_charge,
        ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2)
                                                                      AS historical_churned_monthly_revenue
    FROM telco_churn
    GROUP BY Contract
)
SELECT
    Contract,
    total_customers,
    churned,
    churn_rate,
    avg_monthly_charge,
    historical_churned_monthly_revenue,
    RANK() OVER (ORDER BY churn_rate DESC)                            AS risk_rank
FROM contract_metrics
ORDER BY churn_rate DESC;


-- ────────────────────────────────────────────────────────────
-- SECTION 3: PAYMENT METHOD RETENTION ANALYSIS
-- DENSE_RANK + CASE for risk labeling
-- ────────────────────────────────────────────────────────────

SELECT
    PaymentMethod,
    COUNT(*)                                                          AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)                   AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 2)                                                AS churn_rate_pct,
    ROUND(
        SUM(CASE WHEN Churn = 'No' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 2)                                                AS retention_rate_pct,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2)
                                                                      AS monthly_revenue_lost,
    DENSE_RANK() OVER (
        ORDER BY
            SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0
            / COUNT(*) DESC
    )                                                                 AS churn_rank,
    CASE
        WHEN SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0
             / COUNT(*) > 35 THEN 'HIGH RISK'
        WHEN SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0
             / COUNT(*) > 20 THEN 'MEDIUM RISK'
        ELSE 'LOW RISK'
    END                                                               AS risk_tier
FROM telco_churn
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;


-- ────────────────────────────────────────────────────────────
-- SECTION 4: TENURE COHORT ANALYSIS
-- CTE + CASE bucketing + Running total window function
-- ────────────────────────────────────────────────────────────

WITH tenure_segmented AS (
    SELECT
        CASE
            WHEN tenure BETWEEN 0  AND 12 THEN '01. 0–12 months'
            WHEN tenure BETWEEN 13 AND 24 THEN '02. 13–24 months'
            WHEN tenure BETWEEN 25 AND 48 THEN '03. 25–48 months'
            ELSE                               '04. 49+ months'
        END                                                           AS tenure_cohort,
        MonthlyCharges,
        TotalCharges,
        Churn
    FROM telco_churn
),
cohort_summary AS (
    SELECT
        tenure_cohort,
        COUNT(*)                                                      AS customers,
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)               AS churned,
        ROUND(
            SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0
            / COUNT(*), 2)                                            AS churn_rate,
        ROUND(AVG(MonthlyCharges), 2)                                 AS avg_monthly_charge,
        ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2)
                                                                      AS revenue_lost
    FROM tenure_segmented
    GROUP BY tenure_cohort
)
SELECT
    tenure_cohort,
    customers,
    churned,
    churn_rate,
    avg_monthly_charge,
    revenue_lost,
    SUM(customers) OVER (
        ORDER BY tenure_cohort
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )                                                                 AS cumulative_customers,
    ROUND(
        SUM(revenue_lost) OVER (
            ORDER BY churn_rate DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ), 2)                                                         AS cumulative_revenue_lost
FROM cohort_summary
ORDER BY tenure_cohort;


-- ────────────────────────────────────────────────────────────
-- SECTION 5: INTERNET SERVICE + TECH SUPPORT CROSS ANALYSIS
-- JOIN-style aggregation using CASE
-- ────────────────────────────────────────────────────────────

SELECT
    InternetService,
    TechSupport,
    COUNT(*)                                                          AS customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)                   AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 2)                                                AS churn_rate,
    ROUND(AVG(MonthlyCharges), 2)                                     AS avg_charge,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2)
                                                                      AS revenue_at_risk,
    RANK() OVER (ORDER BY
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*) DESC)                                              AS risk_rank
FROM telco_churn
WHERE InternetService != 'No'
GROUP BY InternetService, TechSupport
ORDER BY churn_rate DESC;


-- ────────────────────────────────────────────────────────────
-- SECTION 6: HEURISTIC RISK SCORE (observational association check)
-- Hand-assigned weights, not probabilities or causal effects:
-- month-to-month 30, fiber 15, electronic check 12, no support 10,
-- senior 8, tenure under 12 months 10, monthly charges above $75 8.
-- ────────────────────────────────────────────────────────────

WITH risk_scored AS (
    SELECT
        customerID,
        Contract,
        InternetService,
        TechSupport,
        PaymentMethod,
        SeniorCitizen,
        tenure,
        MonthlyCharges,
        Churn,
        -- Risk scoring model (higher = more at risk)
        (CASE WHEN Contract = 'Month-to-month'      THEN 30 ELSE 0 END) +
        (CASE WHEN InternetService = 'Fiber optic'  THEN 15 ELSE 0 END) +
        (CASE WHEN PaymentMethod = 'Electronic check' THEN 12 ELSE 0 END) +
        (CASE WHEN TechSupport = 'No'               THEN 10 ELSE 0 END) +
        (CASE WHEN SeniorCitizen = 1                THEN 8  ELSE 0 END) +
        (CASE WHEN tenure < 12                      THEN 10 ELSE 0 END) +
        (CASE WHEN MonthlyCharges > 75              THEN 8  ELSE 0 END)  AS risk_score
    FROM telco_churn
)
SELECT
    risk_score,
    CASE
        WHEN risk_score >= 55 THEN 'CRITICAL — Intervene immediately'
        WHEN risk_score >= 40 THEN 'HIGH — Priority outreach'
        WHEN risk_score >= 25 THEN 'MEDIUM — Monitor closely'
        ELSE 'LOW — Standard retention'
    END                                                               AS risk_label,
    COUNT(*)                                                          AS customers,
    ROUND(AVG(MonthlyCharges), 2)                                     AS avg_monthly_value,
    ROUND(SUM(CASE WHEN Churn = 'No' THEN MonthlyCharges ELSE 0 END), 2)
                                                                      AS active_monthly_revenue_at_risk,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2)
                                                                      AS observed_churn_rate_pct
FROM risk_scored
GROUP BY risk_score, risk_label
ORDER BY risk_score DESC
LIMIT 15;


-- ────────────────────────────────────────────────────────────
-- SECTION 7: SENIOR CITIZEN DEEP DIVE
-- ────────────────────────────────────────────────────────────

SELECT
    CASE WHEN SeniorCitizen = 1 THEN 'Senior' ELSE 'Non-Senior' END  AS customer_type,
    Contract,
    COUNT(*)                                                          AS customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)                   AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 2)                                                AS churn_rate,
    ROUND(AVG(MonthlyCharges), 2)                                     AS avg_charge,
    ROUND(AVG(tenure), 1)                                             AS avg_tenure_months
FROM telco_churn
GROUP BY SeniorCitizen, Contract
ORDER BY SeniorCitizen DESC, churn_rate DESC;


-- ────────────────────────────────────────────────────────────
-- SECTION 8: HISTORICAL LOSS AND ACTIVE EXPOSURE (not recoverable revenue)
-- Segment rows overlap. Do not sum them. Active exposure is descriptive and
-- does not estimate campaign effect, churn probability, or recoverable revenue.
-- ────────────────────────────────────────────────────────────

SELECT 'All customers' AS segment, COUNT(*) AS customers,
       SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
       ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2)
         AS historical_churned_monthly_revenue,
       SUM(CASE WHEN Churn = 'No' THEN 1 ELSE 0 END) AS active_customers,
       ROUND(SUM(CASE WHEN Churn = 'No' THEN MonthlyCharges ELSE 0 END), 2)
         AS active_monthly_revenue_at_risk
FROM telco_churn
UNION ALL
SELECT 'Active month-to-month', COUNT(*), 0, 0, COUNT(*), ROUND(COALESCE(SUM(MonthlyCharges), 0), 2)
FROM telco_churn WHERE Churn = 'No' AND Contract = 'Month-to-month'
UNION ALL
SELECT 'Active electronic check', COUNT(*), 0, 0, COUNT(*), ROUND(COALESCE(SUM(MonthlyCharges), 0), 2)
FROM telco_churn WHERE Churn = 'No' AND PaymentMethod = 'Electronic check'
UNION ALL
SELECT 'Active fiber without tech support', COUNT(*), 0, 0, COUNT(*), ROUND(COALESCE(SUM(MonthlyCharges), 0), 2)
FROM telco_churn
WHERE Churn = 'No' AND InternetService = 'Fiber optic' AND TechSupport = 'No';

