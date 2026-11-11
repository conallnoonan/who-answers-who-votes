-- 03_poststrat_frame.sql
-- Adult citizens per state x age x sex x race x education cell (ACS PUMS).
CREATE OR REPLACE TABLE mart.poststrat AS
SELECT
    st AS state_fips,
    CASE WHEN agep BETWEEN 18 AND 29 THEN '18-29'
         WHEN agep BETWEEN 30 AND 44 THEN '30-44'
         WHEN agep BETWEEN 45 AND 64 THEN '45-64'
         ELSE '65+' END                         AS age_grp,
    CASE WHEN sex = 1 THEN 'male' ELSE 'female' END AS sex,
    CASE WHEN hisp <> 1 THEN 'hispanic'
         WHEN rac1p = 1 THEN 'white'
         WHEN rac1p = 2 THEN 'black'
         WHEN rac1p = 6 THEN 'asian'
         ELSE 'other' END                       AS race,
    CASE WHEN schl <= 15 THEN 'lt_hs'
         WHEN schl IN (16, 17) THEN 'hs'
         WHEN schl BETWEEN 18 AND 20 THEN 'some_college'
         WHEN schl = 21 THEN 'ba'
         ELSE 'postgrad' END                    AS educ,
    SUM(pwgtp)                                  AS n
FROM raw.acs_pums_person
WHERE agep >= 18
  AND cit <> 5          -- 5 = not a U.S. citizen
  AND st <= 56          -- 50 states + DC (drops Puerto Rico)
GROUP BY ALL;

-- Margin tables used by raking (state-specific population shares).
CREATE OR REPLACE VIEW mart.ps_margins AS
SELECT state_fips, 'age_grp' AS var, age_grp AS level, SUM(n) AS n FROM mart.poststrat GROUP BY ALL
UNION ALL
SELECT state_fips, 'sex', sex, SUM(n) FROM mart.poststrat GROUP BY ALL
UNION ALL
SELECT state_fips, 'race', race, SUM(n) FROM mart.poststrat GROUP BY ALL
UNION ALL
SELECT state_fips, 'educ', educ, SUM(n) FROM mart.poststrat GROUP BY ALL;
