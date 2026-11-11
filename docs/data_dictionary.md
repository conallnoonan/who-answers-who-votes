# Data dictionary

All tables live in one DuckDB file (`data/survey_bias.duckdb`). Explore it with
`duckdb data/survey_bias.duckdb` or from Python with `wawv.db.query("SELECT ...")`.

## Sources

| Source | Version | Terms |
|---|---|---|
| Cooperative Election Study, Common Content | 2024 (Harvard Dataverse) | Free for research; cite per the Dataverse page |
| ACS Public Use Microdata Sample | Most recent 5-year release | Public domain (U.S. Census Bureau) |
| MIT Election Data + Science Lab, U.S. President 1976–2024 | Latest (Harvard Dataverse) | CC0; cite per the Dataverse page |

## `raw` schema: files as delivered (only needed columns kept)

| Table | Grain | Built in |
|---|---|---|
| `raw.ces_2024` | respondent | `wawv.load` (pyreadstat) |
| `raw.acs_pums_person` | ACS person record | `sql/01_load_raw.sql` |
| `raw.mit_president_state` | state × year × candidate | `sql/01_load_raw.sql` |

## `clean.ces`: one row per CES respondent (`sql/02_clean_ces.sql`)

| Column | Type | Definition |
|---|---|---|
| `caseid` | id | CES respondent ID |
| `state_fips` | int | State FIPS code (`inputstate`) |
| `age_grp` | text | 18-29 / 30-44 / 45-64 / 65+, from `birthyr` |
| `sex` | text | male / female from `gender4`; NULL for nonbinary/other (no ACS equivalent) |
| `race` | text | hispanic (any race) / white / black / asian / other |
| `educ` | text | lt_hs / hs / some_college (incl. 2-year) / ba / postgrad |
| `self_voted` | 0/1 | Says they definitely voted in the 2024 general election |
| `validated_voted` | 0/1 | Matched to a 2024 general-election vote record (TargetSmart) |
| `vote_dem` | 0/1/NULL | 1 = Democratic, 0 = Republican presidential vote; NULL otherwise |
| `pid7`, `ideo5`, `newsint` | int | Party ID (7-pt), ideology (5-pt), news interest (1 = most of the time … 4 = hardly at all) |
| `ces_wt` | float | `commonpostweight` |

## `mart` schema: analysis-ready

| Table | Grain | Columns |
|---|---|---|
| `mart.poststrat` | state × age × sex × race × educ cell | 5 cell keys + `n` (adult citizens, PUMS-weighted) |
| `mart.ps_margins` (view) | state × variable × level | one-way margins used for raking |
| `mart.truth` | state × year (2020, 2024) | `dem_2p` = Dem / (Dem + Rep) votes |
| `mart.state_covs` | state | `dem_2020_c` = centered logit of 2020 Dem two-party share |

Cell definitions are identical in `clean.ces` and `mart.poststrat`; that equivalence is
what makes poststratification valid, so change both files together.

## Outputs (`reports/tables/`)

| File | Contents |
|---|---|
| `rq1_overreport_mcnemar.csv` | Self-report vs validated rates, McNemar test |
| `rq2_overreport_ame.csv` | Average marginal effects on overreporting |
| `rq3_turnout_metrics.csv` | AUC / Brier by model and feature set, bootstrap AUC-difference CI |
| `rq3_test_predictions.csv` | Held-out predictions (for calibration plots) |
| `rq3_importance_full_model.csv` | Permutation importance, full turnout model |
| `rq4_state_estimates.csv` | Every estimator's estimate per state (MRP adds 95% intervals) |
| `rq4_state_errors.csv` | Estimate, certified result and error, per method × state |
| `rq4_accuracy_summary.csv` | Bias / MAE / RMSE by method |
| `rq4_paired_tests.csv` | Wilcoxon tests and bootstrap RMSE-difference CIs vs the best method |
| `rq4_raking_deff.csv` | Kish design effect of the raked weights, by state |
| `rq4_mrp_diagnostics.csv` | Max R̂, min ESS, divergences for the MRP model |
