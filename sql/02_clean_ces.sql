-- 02_clean_ces.sql
-- Recode CES respondents into the same cells as the ACS frame (03).
-- TODO: verify every code against the CES 2024 codebook.
CREATE OR REPLACE TABLE clean.ces AS
SELECT
    caseid,
    CAST(inputstate AS INTEGER)                         AS state_fips,
    CASE WHEN 2024 - birthyr BETWEEN 18 AND 29 THEN '18-29'
         WHEN 2024 - birthyr BETWEEN 30 AND 44 THEN '30-44'
         WHEN 2024 - birthyr BETWEEN 45 AND 64 THEN '45-64'
         ELSE '65+' END                                 AS age_grp,
    -- ACS records binary sex only; nonbinary/other respondents get NULL here
    -- and are dropped from the poststratified models (note this in limitations).
    CASE WHEN gender4 = 1 THEN 'male'
         WHEN gender4 = 2 THEN 'female' END             AS sex,
    CASE WHEN race = 3 OR hispanic = 1 THEN 'hispanic'
         WHEN race = 1 THEN 'white'
         WHEN race = 2 THEN 'black'
         WHEN race = 4 THEN 'asian'
         ELSE 'other' END                               AS race,
    CASE WHEN educ = 1 THEN 'lt_hs'
         WHEN educ = 2 THEN 'hs'
         WHEN educ IN (3, 4) THEN 'some_college'
         WHEN educ = 5 THEN 'ba'
         ELSE 'postgrad' END                            AS educ,
    CAST(cc24_401 = $SELF_VOTED_CODE AS INTEGER)        AS self_voted,
    -- Validated vote: any TargetSmart vote-method code = voted; NULL = no record.
    CAST(ts_g2024 IS NOT NULL AS INTEGER)               AS validated_voted,
    CASE WHEN cc24_410 = $DEM_CODE THEN 1
         WHEN cc24_410 = $REP_CODE THEN 0 END           AS vote_dem,   -- two-party only
    pid7, ideo5, newsint,
    commonpostweight                                    AS ces_wt
FROM raw.ces_2024
WHERE inputstate IS NOT NULL AND birthyr IS NOT NULL;
