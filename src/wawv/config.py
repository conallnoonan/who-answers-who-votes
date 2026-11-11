"""Project-wide paths and constants.

Set the environment variable WAWV_DATA_RAW to point the pipeline at a different
raw-data folder (the synthetic demo uses this).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = Path(os.environ.get("WAWV_DATA_RAW", ROOT / "data" / "raw"))
DB_PATH = Path(os.environ.get("WAWV_DB", ROOT / "data" / "survey_bias.duckdb"))
SQL_DIR = ROOT / "sql"

# Demo runs (synthetic data) write to their own folders so they never mix
# with real results.
IS_DEMO = "synthetic" in DATA_RAW.as_posix()
_suffix = "demo" if IS_DEMO else ""
REPORT_DIR = ROOT / "reports" / _suffix if IS_DEMO else ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
TAB_DIR = REPORT_DIR / "tables"
MODEL_DIR = ROOT / "models" / _suffix if IS_DEMO else ROOT / "models"

# ---- Raw input files (download instructions in README) --------------------
CES_FILE = DATA_RAW / "CES24_Common.dta"
PUMS_GLOB = DATA_RAW / "pums" / "psam_pus*.csv"
MIT_FILE = DATA_RAW / "1976-2024-president.csv"

# ---- CES variable names and codes -----------------------------------------
# TODO: confirm every name/code below against the CES 2024 codebook.
CES_COLS = [
    "caseid", "inputstate", "birthyr", "gender4", "race", "hispanic", "educ",
    "pid7", "ideo5", "newsint", "commonpostweight",
    "cc24_401",   # post-election self-reported turnout
    "cc24_410",   # post-election presidential vote choice
    "ts_g2024",   # TargetSmart validated 2024 general-election vote
]
SELF_VOTED_CODE = 5     # cc24_401: "I definitely voted"
DEM_CODE = 1            # cc24_410 code for the Democratic candidate
REP_CODE = 2            # cc24_410 code for the Republican candidate

# ---- Modeling ---------------------------------------------------------------
PS_VARS = ["state_fips", "age_grp", "sex", "race", "educ"]  # shared with ACS frame
SEED = 42

for d in (FIG_DIR, TAB_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
