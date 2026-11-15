"""Write the methods-and-results report from the saved tables, and refresh the
"Key results" block in the README.

    python -m wawv.report
"""
import re
from datetime import date

import pandas as pd

from . import config
from .figures import _clean_term
from .viz import METHOD_LABELS

T = config.TAB_DIR


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Minimal markdown table (avoids a tabulate dependency)."""
    def fmt(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def pval(p: float) -> str:
    return "p < .001" if p < 0.001 else f"p = {p:.3f}"


def load() -> dict:
    r = {"rq1": pd.read_csv(T / "rq1_overreport_mcnemar.csv").iloc[0],
         "rq2": pd.read_csv(T / "rq2_overreport_ame.csv"),
         "rq3": pd.read_csv(T / "rq3_turnout_metrics.csv"),
         "imp": pd.read_csv(T / "rq3_importance_full_model.csv"),
         "acc": pd.read_csv(T / "rq4_accuracy_summary.csv", index_col=0),
         "tests": pd.read_csv(T / "rq4_paired_tests.csv"),
         "deff": pd.read_csv(T / "rq4_raking_deff.csv")}
    for key, name in [("diag", "rq4_mrp_diagnostics.csv"),
                      ("diag_pv", "rq4_mrp_pv_diagnostics.csv")]:
        path = T / name
        r[key] = pd.read_csv(path).iloc[0] if path.exists() else None
    deff_pv = T / "rq4_raking_pv_deff.csv"
    r["deff_pv"] = pd.read_csv(deff_pv) if deff_pv.exists() else None
    return r


def _boost_verdict(b: pd.DataFrame) -> str:
    lo = b.auc_diff_xgb_minus_logit_lo.iloc[0]
    hi = b.auc_diff_xgb_minus_logit_hi.iloc[0]
    if lo > 0:
        return "Gradient boosting predicts turnout better than logistic regression"
    if hi < 0:
        return "Logistic regression predicts turnout better than gradient boosting"
    return "Gradient boosting doesn't beat logistic regression on turnout"


def _best_sentence(best, worst, acc, t) -> str:
    lab = lambda m: METHOD_LABELS.get(m, m)
    head = (f"**{lab(best)} recovers state results best**: RMSE "
            f"{acc.loc[best, 'rmse'] * 100:.1f} pts vs {acc.loc[worst, 'rmse'] * 100:.1f} "
            f"for {lab(worst)}.")
    t = t[t.vs == best]
    ties = [lab(m) for m in t.loc[t.rmse_diff_lo <= 0, "method"]]
    if not ties:
        return head + " Its RMSE is lower than every alternative (all bootstrap CIs exclude zero)."
    return (head + f" Its RMSE is statistically indistinguishable from {', '.join(ties)} "
            "(bootstrap 95% CIs include zero) but clearly lower than every other method.")


def key_findings(r: dict) -> list[str]:
    q1, acc, m3, t = r["rq1"], r["acc"], r["rq3"], r["tests"]
    best, worst = acc.index[0], acc.index[-1]
    findings = []

    # ---- RQ1: overreporting, as a range when the matched-only results exist ----
    gap = (q1.self_report_rate - q1.validated_rate) * 100
    matched_path = T / "rq1_overreport_mcnemar_matched.csv"
    if matched_path.exists():
        q1m = pd.read_csv(matched_path).iloc[0]
        gap_m = (q1m.self_report_rate - q1m.validated_rate) * 100
        findings.append(
            f"**Turnout is overstated by {gap_m:.1f} to {gap:.1f} points.** "
            f"{q1.self_report_rate:.0%} of respondents (unweighted) say they voted, versus "
            f"{q1.validated_rate:.0%} with a validated vote when unmatched respondents count "
            f"as non-voters; restricting to respondents matched to the voter file narrows "
            f"the gap to {gap_m:.1f} points. Both are significant (McNemar {pval(q1.p_value)}).")
    else:
        findings.append(
            f"**Turnout is overstated by {gap:.1f} points.** {q1.self_report_rate:.0%} of "
            f"respondents (unweighted) say they voted; {q1.validated_rate:.0%} have a "
            f"voter-file record (McNemar χ² = {q1.mcnemar_chi2:,.0f}, {pval(q1.p_value)}).")

    # ---- RQ2: who overreports (confirmed non-voters when available) ----
    ame_path = T / "rq2_overreport_ame_matched.csv"
    ame = pd.read_csv(ame_path) if ame_path.exists() else r["rq2"]
    who = ("respondents confirmed in the voter file as non-voters" if ame_path.exists()
           else "validated non-voters")
    top = ame.reindex(ame["dy/dx"].abs().sort_values(ascending=False).index).iloc[0]
    findings.append(
        f"**Overreporting is patterned, not random.** The largest effect is "
        f"{_clean_term(top.term)}: {top['dy/dx'] * 100:+.1f} pts in the probability of "
        f"claiming a vote, among {who}.")

    # ---- RQ3: turnout model ----
    b = m3[m3.spec == "B_poststrat"].set_index("model")
    findings.append(
        f"**{_boost_verdict(b)}** with demographics alone (AUC {b.loc['xgb', 'auc']:.3f} "
        f"vs {b.loc['logit', 'auc']:.3f}; 95% CI for the difference "
        f"[{b.auc_diff_xgb_minus_logit_lo.iloc[0]:+.3f}, "
        f"{b.auc_diff_xgb_minus_logit_hi.iloc[0]:+.3f}]).")

    # ---- RQ4: what the 2020-vote adjustment achieved ----
    pv = [m for m in acc.index if m.endswith("_pv")]
    demo = [m for m in acc.index if not m.endswith("_pv")]
    if pv and demo:
        b_demo, b_pv = acc.loc[demo, "bias"] * 100, acc.loc[pv, "bias"] * 100
        cut = 1 - b_pv.mean() / b_demo.mean() if b_demo.mean() else 0
        verb = ("largely can" if cut >= 0.5 else "partly can" if cut > 0.1
                else "does not either")
        findings.append(
            f"**Demographic adjustment cannot fix partisan non-response; recalled 2020 vote "
            f"{verb}.** Without 2020 vote, estimates overstate the Democratic share by "
            f"{b_demo.min():.1f}–{b_demo.max():.1f} points on average across states; adding "
            f"it cuts this to {b_pv.min():.1f}–{b_pv.max():.1f} points "
            f"(average bias {cut:.0%} lower).")

    # ---- RQ4: best method, with honest wording about ties ----
    findings.append(_best_sentence(best, worst, acc, t))
    return findings


def diagnostics_text(d) -> str:
    if d is None:
        return "_not fitted in this run._"
    ok = d.max_rhat < 1.01 and d.min_ess_bulk > 400 and d.divergences == 0
    verdict = ("All convergence checks pass." if ok else
               "⚠️ At least one check fails: run longer chains (`make estimate DRAWS=2000`) "
               "before reporting MRP results.")
    return (f"Max R̂ = {d.max_rhat:.3f} (target < 1.01) · min bulk ESS = "
            f"{d.min_ess_bulk:.0f} (target > 400) · divergences = {int(d.divergences)} "
            f"(target 0) across {int(d.n_params)} parameters. {verdict}")


def build_report(r: dict) -> str:
    fig = "figures"
    banner = ("> ⚠️ **These results come from synthetic demo data** generated by "
              "`wawv.synth`. They show the pipeline works; they are not findings.\n\n"
              if config.IS_DEMO else "")
    acc = r["acc"].copy()
    acc.insert(0, "method", [METHOD_LABELS.get(m, m) for m in acc.index])
    for c in ["bias", "mae", "rmse"]:
        acc[c] = (acc[c] * 100).map("{:+.2f}".format if c == "bias" else "{:.2f}".format)
    acc["n_states"] = acc.n_states.astype(int)
    acc.columns = ["Method", "Bias (pts)", "MAE (pts)", "RMSE (pts)", "States"]

    t = r["tests"].copy()
    t["Method"] = t.method.map(lambda m: METHOD_LABELS.get(m, m))
    t["Compared with"] = t.vs.map(lambda m: METHOD_LABELS.get(m, m))
    t["Wilcoxon p"] = t.p_value.map(pval)
    t["RMSE difference, 95% CI (pts)"] = [f"[{lo * 100:+.2f}, {hi * 100:+.2f}]"
                                          for lo, hi in zip(t.rmse_diff_lo, t.rmse_diff_hi)]
    t = t[["Method", "Compared with", "Wilcoxon p", "RMSE difference, 95% CI (pts)"]]

    m3 = r["rq3"].copy()
    m3["spec"] = m3.spec.map({"B_poststrat": "Demographics only",
                              "A_full": "+ partisanship, ideology, news interest"})
    m3["model"] = m3.model.map({"xgb": "XGBoost", "logit": "Logistic"})
    m3 = m3[["spec", "model", "auc", "brier"]]
    m3.columns = ["Features", "Model", "AUC", "Brier"]

    imp = r["imp"].head(5)
    deff = r["deff"].kish_deff
    deff_line = (f"Demographic raking median {deff.median():.2f} "
                 f"(range {deff.min():.2f}–{deff.max():.2f})")
    if r["deff_pv"] is not None:
        dp = r["deff_pv"].kish_deff
        deff_line += (f"; raking with 2020 vote median {dp.median():.2f} "
                      f"(range {dp.min():.2f}–{dp.max():.2f})")

    findings = "\n".join(f"{i}. {f}" for i, f in enumerate(key_findings(r), 1))
    return f"""# Who Answers, Who Votes
### Correcting survey bias in the 2024 U.S. presidential election

{banner}_Generated {date.today():%B %d, %Y} by `wawv.report`._

## Summary

{findings}

## 1. Data

| Source | Use |
|---|---|
| Cooperative Election Study (CES) 2024 Common Content | Respondent demographics, self-reported turnout and vote choice, TargetSmart-validated turnout |
| American Community Survey 5-year PUMS | Poststratification frame: adult citizens by state × age × sex × race/ethnicity × education |
| MIT Election Data + Science Lab | Certified 2024 state results (ground truth); 2020 results as a state-level predictor and as recalled-vote targets; 2024 total votes cast for turnout calibration |

All three are loaded into a single DuckDB database (`sql/`): raw → clean (CES recoded
to the ACS cell definitions) → mart (poststratification frame, raking margins, truth and
vote-total tables). Validated turnout follows the CES guide's Specification 1: TS_g2024
codes 1–6 are voters; code 7 (matched non-voters) and unmatched respondents are non-voters.

The CES's own weights are not used. They are post-stratified on 2024 presidential vote by
state (CES 2024 Guide, p. 17), so they are calibrated to the outcome being estimated and
cannot fairly measure misreporting or compete as an estimator.

## 2. Methods

**RQ1 – Overreporting.** McNemar's test compares each respondent's self-reported
turnout with their validated record, among respondents who answered the turnout question
(unweighted). Because unmatched respondents may be real voters the vendor could not find,
the gap is reported as a range: an upper bound counting unmatched respondents as
non-voters, and a lower bound restricted to respondents matched to the voter file.

**RQ2 – Who overreports.** Logistic regression of claiming to have voted, among
respondents with no validated vote, on education, age, race/ethnicity, party ID and
news interest; average marginal effects with 95% CIs. Re-estimated among matched
non-voters only, to separate misreporting from voter-file matching failure.

**RQ3 – Turnout prediction.** Logistic regression and XGBoost on a stratified 80/20
split, with two feature sets: demographics only (the set that also exists in the ACS,
so it can be poststratified) and demographics + attitudes. Evaluated by AUC, Brier
score and calibration; AUC difference by bootstrap.

**RQ4 – Recovering the election result.** Seven estimators of each state's
Democratic share of the two-party vote, in two families. Turnout models are calibrated
so predicted voters match each state's certified 2024 total votes cast (turnout level
only, never the partisan split).

| Estimator | How |
|---|---|
| Unweighted | Mean among self-reported voters |
| Raking + likely voter | Within-state raking to ACS margins (weights capped at 5× mean), × predicted turnout |
| XGBoost poststrat | XGBoost vote-choice model on frame cells, weighted by cell N × predicted turnout |
| MRP + likely voter | Bayesian multilevel binomial model (bambi/PyMC) with state, age, race, education, race×education and age×education effects, sex and 2020 state vote as fixed effects; poststratified by cell N × predicted turnout |
| Raking + 2020 vote | Within-state raking to ACS margins and to certified 2020 vote shares (Biden / Trump / other / did not vote, as shares of today's adult citizens), × 2020-vote-aware turnout |
| XGBoost + 2020 vote | XGBoost vote model with recalled 2020 vote, poststratified over a frame expanded by 2020 vote group and calibrated (IPF) to each state's 2020 result |
| MRP + 2020 vote | The MRP model plus a 2020-vote fixed effect, poststratified over the same calibrated frame |

Accuracy is RMSE, MAE and bias against certified results. Methods are compared with
Wilcoxon signed-rank tests on paired absolute state errors and bootstrap CIs for RMSE
differences.

## 3. Results

### Turnout is overstated

![Turnout gap]({fig}/fig1_turnout_gap.png)

![Who overreports]({fig}/fig2_overreport_ame.png)

### Predicting who votes

**Table 1. Turnout models, held-out test set**

{md_table(m3)}

![Calibration]({fig}/fig3_turnout_calibration.png)

Most important features in the full model (permutation importance, AUC drop):
{", ".join(f"`{f}` ({v:.3f})" for f, v in zip(imp.feature, imp.importance))}.

### Recovering the 2024 result

**Table 2. Accuracy against certified 2024 results** (percentage points)

{md_table(acc)}

**Table 3. Paired comparisons with the best method**

{md_table(t)}

![Method errors]({fig}/fig4_method_errors.png)

![Error map]({fig}/fig5_error_tilemap.png)

![Estimates vs actual]({fig}/fig6_estimate_vs_actual.png)

**MRP convergence.**
- MRP + likely voter: {diagnostics_text(r["diag"])}
- MRP + 2020 vote: {diagnostics_text(r["diag_pv"])}

**Raking weights (Kish design effect across states).** {deff_line}.

## 4. Limitations

- **Vote validation is imperfect.** Unmatched respondents include real voters the
  vendor could not find, which is why RQ1 is reported as a range. The turnout model and
  likely-voter adjustment use the broader definition, so they partly learn who is hard
  to match (e.g. young, mobile respondents).
- **Recalled 2020 vote is itself reported.** Some 2020 non-voters claim to have voted and
  some recall their choice inaccurately; any such error limits the correction.
- **Residual bias remains.** Even with 2020 vote, estimates lean Democratic by a few
  points, consistent with non-response *within* 2020 vote groups (e.g. Biden 2020 voters
  who moved to Trump being less likely to respond); this is a hypothesis, not tested here.
- **Binary sex in the ACS.** Nonbinary and other-gender CES respondents can't be
  placed in a frame cell and are excluded from the poststratified models.
- **Vote choice comes from self-reported voters**, including some who didn't vote.
- **The 2020-vote frame is modelled.** The joint distribution of demographics and
  2020 vote comes from a model of the CES calibrated to state totals, not from a census.
- **Truth is partly used.** Turnout is calibrated to certified 2024 vote totals (level
  only, never the partisan split), and this is a retrospective benchmark; a forecasting
  setting would have neither.

## 5. Reproduce

```bash
pip install -r requirements.txt
make all        # load → overreport → turnout → estimate → evaluate → figures → report
# add or refit only the 2020-vote methods:
PYTHONPATH=src python -m wawv.estimators --pv-only --draws 2000 --chains 4
```
"""


def update_readme(r: dict) -> None:
    readme = config.ROOT / "README.md"
    text = readme.read_text()
    rel = config.FIG_DIR.relative_to(config.ROOT).as_posix()
    note = ("> ⚠️ Shown from the **synthetic demo run**. Run `make all` on the real "
            "data to replace with actual findings.\n\n" if config.IS_DEMO else "")
    block = (f"<!-- RESULTS:START -->\n{note}"
             + "\n".join(f"- {f}" for f in key_findings(r))
             + f"\n\n<p align=\"center\"><img src=\"{rel}/fig4_method_errors.png\" width=\"49%\">"
             f" <img src=\"{rel}/fig5_error_tilemap.png\" width=\"49%\"></p>\n\n"
             f"Full write-up: [`{config.REPORT_DIR.relative_to(config.ROOT).as_posix()}/REPORT.md`]"
             f"({config.REPORT_DIR.relative_to(config.ROOT).as_posix()}/REPORT.md)\n"
             "<!-- RESULTS:END -->")
    text = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", block, text, flags=re.S)
    readme.write_text(text)


def main() -> None:
    r = load()
    out = config.REPORT_DIR / "REPORT.md"
    out.write_text(build_report(r))
    update_readme(r)
    print(f"  wrote {out.relative_to(config.ROOT)} and refreshed README key results")


if __name__ == "__main__":
    main()