"""DuckDB connection and SQL-file runner."""
from string import Template

import duckdb
import pandas as pd

from . import config


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(config.DB_PATH), read_only=read_only)


def sql_params() -> dict:
    """Values substituted into $PLACEHOLDERS inside the .sql files."""
    return {
        "PUMS_GLOB": config.PUMS_GLOB.as_posix(),
        "MIT_FILE": config.MIT_FILE.as_posix(),
        "SELF_VOTED_CODE": config.SELF_VOTED_CODE,
        "DEM_CODE": config.DEM_CODE,
        "REP_CODE": config.REP_CODE,
    }


def run_sql_file(con, name: str) -> None:
    sql = Template((config.SQL_DIR / name).read_text()).substitute(sql_params())
    con.execute(sql)
    print(f"  ran {name}")


def query(sql: str) -> pd.DataFrame:
    with connect(read_only=True) as con:
        return con.execute(sql).df()
