-- 04_truth_and_covariates.sql
-- Dem share of the two-party vote by state and year, from MIT Election Lab.
-- party_simplified, candidatevotes are the names used in recent releases).
CREATE OR REPLACE TABLE mart.truth AS
SELECT
    year,
    CAST(state_fips AS INTEGER) AS state_fips,
    SUM(candidatevotes) FILTER (WHERE party_simplified = 'DEMOCRAT')::DOUBLE
      / SUM(candidatevotes) FILTER (WHERE party_simplified IN ('DEMOCRAT', 'REPUBLICAN'))
      AS dem_2p
FROM raw.mit_president_state
WHERE year IN (2020, 2024)
GROUP BY ALL;

-- State-level predictor for MRP / ML: centered logit of the 2020 Dem share.
CREATE OR REPLACE TABLE mart.state_covs AS
WITH t AS (
    SELECT state_fips, ln(dem_2p / (1 - dem_2p)) AS logit_dem_2020
    FROM mart.truth WHERE year = 2020
)
SELECT state_fips, logit_dem_2020 - AVG(logit_dem_2020) OVER () AS dem_2020_c
FROM t;
-- State vote totals by year, used by the recalled-2020-vote methods:
-- 2020 totals set each state's 2020 vote-group targets; 2024 totals calibrate turnout.
-- `total` includes minor candidates (and, in a few states, blank/void ballots).
CREATE OR REPLACE TABLE mart.state_votes AS
SELECT
    year,
    CAST(state_fips AS INTEGER) AS state_fips,
    SUM(candidatevotes) FILTER (WHERE party_simplified = 'DEMOCRAT')   AS dem,
    SUM(candidatevotes) FILTER (WHERE party_simplified = 'REPUBLICAN') AS rep,
    MAX(totalvotes)                                                    AS total
FROM raw.mit_president_state
WHERE year IN (2020, 2024)
GROUP BY ALL;
