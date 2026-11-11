"""RQ1–RQ2: how much, and who, overreports turnout.

    python -m wawv.overreport
"""
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import mcnemar

from . import config
from .db import query


def rq1_mcnemar(df: pd.DataFrame) -> pd.DataFrame:
    """Paired test: each respondent's self-report vs. their validated record."""
    tab = (pd.crosstab(df.self_voted, df.validated_voted)
             .reindex(index=[0, 1], columns=[0, 1], fill_value=0))
    res = mcnemar(tab.values, exact=False, correction=True)
    w = df.ces_wt
    out = pd.DataFrame([{
        "n": len(df),
        "self_report_rate": df.self_voted.mean(),
        "validated_rate": df.validated_voted.mean(),
        "self_report_rate_wtd": (df.self_voted * w).sum() / w.sum(),
        "validated_rate_wtd": (df.validated_voted * w).sum() / w.sum(),
        "overreporters": int(tab.loc[1, 0]),   # said yes, no record
        "underreporters": int(tab.loc[0, 1]),  # said no, has record
        "mcnemar_chi2": res.statistic,
        "p_value": res.pvalue,
    }])
    print(tab, "\n", out.T)
    return out


def rq2_who_overreports(df: pd.DataFrame) -> pd.DataFrame:
    """Among respondents with no validated vote: who claims to have voted?"""
    nv = df[df.validated_voted == 0].dropna(subset=["pid7", "newsint"])
    m = smf.logit(
        "self_voted ~ C(educ, Treatment('hs')) + C(age_grp) "
        "+ C(race, Treatment('white')) + C(pid7) + C(newsint)",
        data=nv,
    ).fit(disp=False)
    print(m.summary())
    ame = m.get_margeff(at="overall").summary_frame()
    return ame.rename_axis("term").reset_index()


def main() -> None:
    df = query("SELECT * FROM clean.ces")
    rq1_mcnemar(df).to_csv(config.TAB_DIR / "rq1_overreport_mcnemar.csv", index=False)
    rq2_who_overreports(df).to_csv(config.TAB_DIR / "rq2_overreport_ame.csv", index=False)
    print(f"Tables written to {config.TAB_DIR}")


if __name__ == "__main__":
    main()
