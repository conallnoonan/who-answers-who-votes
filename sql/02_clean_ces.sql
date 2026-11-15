-- 02_clean_ces.sql
-- Recode CES respondents into the same cells as the ACS frame (03).
-- Codes verified against the CES 2024 codebook.
CREATE OR REPLACE TABLE clean.ces AS
SELECT
    caseid,
    CAST(inputstate AS INTEGER)                         AS state_fips,
    CASE WHEN 2024 - birthyr BETWEEN 18 AND 29 THEN '18-29'
         WHEN 2024 - birthyr BETWEEN 30 AND 44 THEN '30-44'
         WHEN 2024 - birthyr BETWEEN 45 AND 64 THEN '45-64'
         ELSE '65+' END                                 AS age_grp,
    -- ACS records binary sex only; nonbinary/other respondents get NULL here
    -- and are dropped from the poststratified models (noted in limitations).
    CASE WHEN gender4 = 1 THEN 'male'
         WHEN gender4 = 2 THEN 'female' END             AS sex,
    CASE WHEN race = 3 OR hispanic = 1 THEN 'hispanic'
         WHEN race IN (1, 8) THEN 'white'   -- 8 = Middle Eastern; ACS codes most as White
         WHEN race = 2 THEN 'black'
         WHEN race = 4 THEN 'asian'
         ELSE 'other' END                               AS race,
    CASE WHEN educ = 1 THEN 'lt_hs'
         WHEN educ = 2 THEN 'hs'
         WHEN educ IN (3, 4) THEN 'some_college'
         WHEN educ = 5 THEN 'ba'
         WHEN educ = 6 THEN 'postgrad' END              AS educ,
    -- NULL = skipped / not asked / not in the post-election wave
    CASE WHEN cc24_401 = $SELF_VOTED_CODE THEN 1
         WHEN cc24_401 BETWEEN 1 AND 4 THEN 0 END       AS self_voted,
    -- Guide spec. 1: codes 1-6 = voted; 7 = matched non-voter; NULL = unmatched (non-voter)
    CASE WHEN ts_g2024 < 7 THEN 1 ELSE 0 END            AS validated_voted,
    CAST(ts_g2024 IS NOT NULL AS INTEGER)               AS matched,
    CASE WHEN cc24_410 = $DEM_CODE THEN 1
         WHEN cc24_410 = $REP_CODE THEN 0 END           AS vote_dem,   -- two-party only
             -- Recalled 2020 presidential vote. Respondents too young to vote in 2020
    -- usually answer "did not vote", which matches how the targets treat them.
    -- NULL = skipped / not asked; these respondents are left out of the *_pv methods.
    CASE WHEN presvote20post = $PV_DEM_CODE THEN 'biden'
         WHEN presvote20post = $PV_REP_CODE THEN 'trump'
         WHEN presvote20post IN ($PV_OTHER_CODES) THEN 'other'
         WHEN presvote20post IN ($PV_NOVOTE_CODES) THEN 'nonvoter' END AS pv20,
    pid7, ideo5, newsint,
    commonpostweight                                    AS ces_wt
FROM raw.ces_2024
WHERE inputstate BETWEEN 1 AND 56      -- drops skipped/not asked (98, 99)
  AND birthyr IS NOT NULL
  AND educ BETWEEN 1 AND 6;            -- drops skipped/not asked (8, 9)