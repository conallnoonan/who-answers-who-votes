"""Step 1: ingest raw files into DuckDB and build the clean/mart tables.

    python -m wawv.load
"""
import pyreadstat

from . import config
from .db import connect, run_sql_file

SQL_STEPS = [
    "01_load_raw.sql",
    "02_clean_ces.sql",
    "03_poststrat_frame.sql",
    "04_truth_and_covariates.sql",
]


def load_ces(con) -> None:
    """The CES ships as Stata .dta, which DuckDB can't read, so go through pandas."""
    df, _ = pyreadstat.read_dta(str(config.CES_FILE))
    df.columns = df.columns.str.lower()
    missing = sorted(set(config.CES_COLS) - set(df.columns))
    if missing:
        raise KeyError(f"CES columns not found (check codebook / config.CES_COLS): {missing}")
    df = df[config.CES_COLS]
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.register("ces_df", df)
    con.execute("CREATE OR REPLACE TABLE raw.ces_2024 AS SELECT * FROM ces_df")
    print(f"  loaded raw.ces_2024 ({len(df):,} rows)")


def main() -> None:
    with connect() as con:
        load_ces(con)
        for step in SQL_STEPS:
            run_sql_file(con, step)
        for t in ["clean.ces", "mart.poststrat", "mart.truth", "mart.state_covs"]:
            n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t}: {n:,} rows")


if __name__ == "__main__":
    main()
