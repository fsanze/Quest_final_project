USE Company_analysis;

SHOW TABLES;

SELECT *
FROM 1_overview;

SELECT COUNT(*) AS total_rows 
FROM 1_overview;

SELECT *
FROM 2_balance;

SELECT COUNT(*) AS total_rows 
FROM 2_balance;

SELECT *
FROM 3_cash_flow;

SELECT COUNT(*) AS total_rows 
FROM 3_cash_flow;

SELECT *
FROM 4_income;

SELECT COUNT(*) AS total_rows 
FROM 4_income;

SELECT *
FROM 5_stocks;

SELECT COUNT(*) AS total_rows 
FROM 5_stocks;

SELECT *
FROM 6_ratios;

SELECT COUNT(*) AS total_rows 
FROM 6_ratios;

SELECT company_id, COUNT(*) AS total_rows 
FROM 6_ratios
group by company_id;

--------------------------------

-- Top 5 Compañías con mayor valor en bolsa el 29/12/2023
SELECT o.name, s.market_cap
FROM 5_stocks AS s
JOIN 1_overview AS o 
USING (company_id)
WHERE date = '29/12/2023'
ORDER BY s.market_cap desc
LIMIT 5;

-- Top 3 sectores con mayores ingresos en 2020
SELECT o.sector, SUM(i.total_revenue) as total_revenue
FROM 4_income AS i
JOIN 1_overview AS o 
USING (company_id)
WHERE i.year = '2020'
GROUP BY o.sector
ORDER BY total_revenue DESC
LIMIT 3;

-- Top 10 compañias con mayores dividendos por accion y el año
SELECT o.name, r.year, r.dividend_per_share
FROM 6_ratios AS r
JOIN 1_overview AS o 
USING (company_id)
ORDER BY r.dividend_per_share DESC
LIMIT 10;

-- Sector con la menor deuda a corto plazo media
SELECT o.sector, AVG(b.short_term_debt) as short_term_debt
FROM 2_balance AS b
JOIN 1_overview AS o 
USING (company_id)
GROUP BY o.sector
ORDER BY short_term_debt ASC
LIMIT 1;

-- Compañía con menor operating_cashflow
SELECT o.name, c.operating_cashflow
FROM 3_cash_flow AS c
JOIN 1_overview AS o 
USING (company_id)
ORDER BY c.operating_cashflow ASC
LIMIT 1;
