-- 01_load_raw.sql
-- Raw ingestion. DuckDB reads the CSVs directly; no separate import step.
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS clean;
CREATE SCHEMA IF NOT EXISTS mart;

-- ACS PUMS person files (5-year). Keep only the columns we need.
CREATE OR REPLACE TABLE raw.acs_pums_person AS
SELECT
    CAST(ST    AS INTEGER) AS st,
    CAST(AGEP  AS INTEGER) AS agep,
    CAST(SEX   AS INTEGER) AS sex,
    CAST(RAC1P AS INTEGER) AS rac1p,
    CAST(HISP  AS INTEGER) AS hisp,
    CAST(SCHL  AS INTEGER) AS schl,
    CAST(CIT   AS INTEGER) AS cit,
    CAST(PWGTP AS DOUBLE)  AS pwgtp
FROM read_csv('$PUMS_GLOB', union_by_name = true, all_varchar = true);

-- MIT Election Lab state presidential returns.
CREATE OR REPLACE TABLE raw.mit_president_state AS
SELECT * FROM read_csv_auto('$MIT_FILE');
