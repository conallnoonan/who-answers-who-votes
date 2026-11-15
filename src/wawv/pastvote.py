"""Recalled-2020-vote adjustment: three estimators that correct partisan non-response.

Demographic adjustment can't fix a sample that over-represents one party's voters
*within* demographic groups. Recalled 2020 vote can: we know how each state actually
voted in 2020, so we can force the sample (raking) or the population frame (MRP,
XGBoost) to match it.

  raked_pv   rake each state to ACS demographic margins AND the state's 2020 result
  xgb_pv     XGBoost vote model with 2020 vote as a feature, poststratified on the
             2020-vote-expanded frame
  mrp_pv     the MRP model plus a 2020-vote fixed effect, poststratified on the same frame

Pipeline:
  1. pv_targets()      share of each state's adult citizens who voted Biden / Trump /
                       other / not at all in 2020 (certified results / ACS citizens)
  2. build_pv_frame()  split every ACS cell into 4 rows (one per 2020 vote group) using a
                       multinomial model, then calibrate (IPF) so each state's rows add up
                       to its certified 2020 result while each cell keeps its ACS size
  3. pv_turnout()      turnout model with 2020 vote as a predictor, shifted per state so
                       predicted turnout matches each state's actual 2024 vote total
  4. the three estimators above
"""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from . import config
from .db import query

PS = config.PS_VARS
PV = ["biden", "trump", "other", "nonvoter"]
PV_COLS = PS + ["pv20"]


def _onehot(cols, clf, passthrough=()):
    steps = [("oh", OneHotEncoder(handle_unknown="ignore"), cols)]
    if passthrough:
        steps.append(("num", "passthrough", list(passthrough)))
    return Pipeline([("pre", ColumnTransformer(steps)), ("clf", clf)])


# ---------------------------------------------------------------------------
# 1. Targets
# ---------------------------------------------------------------------------
def pv_targets(ps: pd.DataFrame) -> pd.DataFrame:
    """Rows = states, columns = PV, values = share of today's adult citizens.

    The denominator is today's adult citizens (ACS), so people who were too young
    to vote in 2020 fall into 'nonvoter', which is how the CES records them too.
    """
    v = query("SELECT * FROM mart.state_votes WHERE year = 2020").set_index("state_fips")
    cvap = ps.groupby("state_fips").n.sum()
    t = pd.DataFrame({"biden": v.dem / cvap,
                      "trump": v.rep / cvap,
                      "other": (v.total - v.dem - v.rep) / cvap})
    t["nonvoter"] = 1 - t.sum(axis=1)
    return t[PV].clip(lower=0).dropna()


# ---------------------------------------------------------------------------
# 2. Expanded, calibrated population frame
# ---------------------------------------------------------------------------
def build_pv_frame(ces: pd.DataFrame, ps: pd.DataFrame, targets: pd.DataFrame,
                   iters: int = 100) -> pd.DataFrame:
    """Return the ACS frame with 4 rows per cell and a column n_pv (adult citizens
    in that cell who were in that 2020 vote group)."""
    d = ces.dropna(subset=["pv20"])
    mdl = _onehot(PS, LogisticRegression(max_iter=5000)).fit(d[PS], d.pv20)
    share = (pd.DataFrame(mdl.predict_proba(ps[PS]), columns=mdl.classes_)
               .reindex(columns=PV, fill_value=0.0).to_numpy(dtype=float, copy=True))

    n = ps.n.to_numpy()
    states = ps.state_fips.to_numpy()
    for st in np.unique(states):
        m = states == st
        counts = share[m] * n[m, None]                      # cells x 4
        col_target = targets.loc[st, PV].to_numpy() * n[m].sum()
        for _ in range(iters):                             # 2-way IPF
            col = counts.sum(0)
            counts *= np.divide(col_target, col, out=np.zeros_like(col), where=col > 0)
            row = counts.sum(1)
            counts *= np.divide(n[m], row, out=np.zeros_like(row), where=row > 0)[:, None]
        share[m] = counts / n[m, None]

    frame = ps.loc[ps.index.repeat(len(PV))].reset_index(drop=True)
    frame["pv20"] = np.tile(PV, len(ps))
    frame["n_pv"] = (share * n[:, None]).ravel()
    return frame[frame.n_pv > 0].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Turnout model with 2020 vote, calibrated to actual 2024 turnout
# ---------------------------------------------------------------------------
def pv_turnout(ces: pd.DataFrame, frame: pd.DataFrame):
    """Add p_vote to the frame and to the respondents (returned as d)."""
    d = ces.dropna(subset=["pv20"]).copy()
    mdl = _onehot(PV_COLS, LogisticRegression(max_iter=5000)).fit(
        d[PV_COLS], d.validated_voted)
    lf = logit(np.clip(mdl.predict_proba(frame[PV_COLS])[:, 1], 1e-6, 1 - 1e-6))
    lr = logit(np.clip(mdl.predict_proba(d[PV_COLS])[:, 1], 1e-6, 1 - 1e-6))

    total24 = (query("SELECT state_fips, total FROM mart.state_votes WHERE year = 2024")
                 .set_index("state_fips").total)
    n = frame.n_pv.to_numpy()
    states = frame.state_fips.to_numpy()
    shift = {}
    for st in np.unique(states):
        m = states == st
        target = min(total24[st], 0.99 * n[m].sum())
        nm, lm = n[m], lf[m]
        shift[st] = brentq(lambda s, nm=nm, lm=lm, t=target: (nm * expit(lm + s)).sum() - t,
                           -15, 15)

    frame = frame.assign(p_vote=expit(lf + frame.state_fips.map(shift).to_numpy()))
    d["p_vote_pv"] = expit(lr + d.state_fips.map(shift).to_numpy())
    implied = (frame.n_pv * frame.p_vote).sum() / frame.n_pv.sum()
    print(f"  2020-vote turnout model calibrated to 2024 totals: implied = {implied:.3f}")
    return frame, d


# ---------------------------------------------------------------------------
# 4. Estimators
# ---------------------------------------------------------------------------
def est_raked_pv(d: pd.DataFrame, margins: pd.DataFrame, targets: pd.DataFrame):
    from .estimators import kish_deff, rake

    out, deffs = {}, {}
    for st, g in d.groupby("state_fips"):
        m = margins[margins.state_fips == st]
        mg = {var: m[m["var"] == var].set_index("level").n
              for var in ["age_grp", "sex", "race", "educ"]}
        mg["pv20"] = targets.loc[st, PV]
        w = rake(g, mg) * g.p_vote_pv
        v = g.vote_dem.notna()
        out[st] = np.average(g.vote_dem[v], weights=w[v])
        deffs[st] = kish_deff(w[v])
    pd.Series(deffs, name="kish_deff").rename_axis("state_fips").to_csv(
        config.TAB_DIR / "rq4_raking_pv_deff.csv")
    return pd.Series(out)


def est_xgb_pv(d: pd.DataFrame, frame: pd.DataFrame):
    from .estimators import poststratify

    v = d.dropna(subset=["vote_dem"])
    clf = _onehot(PV_COLS, XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                         subsample=0.8, eval_metric="logloss",
                                         random_state=config.SEED),
                  passthrough=["dem_2020_c"])
    clf.fit(v[PV_COLS + ["dem_2020_c"]], v.vote_dem.astype(int))
    f = frame.assign(p_dem=clf.predict_proba(frame[PV_COLS + ["dem_2020_c"]])[:, 1],
                     w=frame.n_pv * frame.p_vote)
    return poststratify(f, "p_dem")


def est_mrp_pv(d: pd.DataFrame, frame: pd.DataFrame, draws=1000, chains=4,
               max_pred_draws=400, chunk_size=1500):
    import arviz as az
    import bambi as bmb

    v = d.dropna(subset=["vote_dem"])
    cells = (v.groupby(PV_COLS + ["dem_2020_c"]).vote_dem
               .agg(dem="sum", trials="count").reset_index())
    cells[PS] = cells[PS].astype(str)
    model = bmb.Model(
        "p(dem, trials) ~ dem_2020_c + sex + pv20 + (1|state_fips) + (1|age_grp) "
        "+ (1|race) + (1|educ) + (1|race:educ) + (1|age_grp:educ)",
        cells, family="binomial")
        # mp_ctx="spawn": start sampler workers as fresh processes. The default "fork"
    # copies a parent whose BLAS threads (Apple Accelerate, used by the scikit-learn
    # models fitted just before) can't survive a fork, which crashes on macOS.
    idata = model.fit(draws=draws, tune=draws, chains=chains, mp_ctx="spawn",
                      target_accept=0.99, random_seed=config.SEED)

    summ = az.summary(idata, kind="diagnostics")
    pd.DataFrame([{"max_rhat": summ.r_hat.max(), "min_ess_bulk": summ.ess_bulk.min(),
                   "min_ess_tail": summ.ess_tail.min(),
                   "divergences": int(idata.sample_stats["diverging"].sum()),
                   "n_params": len(summ)}]).to_csv(
        config.TAB_DIR / "rq4_mrp_pv_diagnostics.csv", index=False)

    step = max(1, idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
               // max_pred_draws)
    thin = idata.sel(draw=slice(None, None, step))
    new = frame.copy()
    new[PS] = new[PS].astype(str)
    new["trials"] = 1
    chunks = []
    for start in range(0, len(new), chunk_size):
        part = new.iloc[start:start + chunk_size]
        with warnings.catch_warnings():  # older bambi needs the flag; newer warns
            warnings.simplefilter("ignore", FutureWarning)
            pred = model.predict(thin, data=part, kind="response_params", inplace=False,
                                 sample_new_groups=True)
        var = [k for k, a in pred.posterior.data_vars.items()
               if a.ndim == 3 and a.shape[-1] == len(part)][-1]
        chunks.append(pred.posterior[var].values.reshape(-1, len(part)))
        del pred
    p = np.concatenate(chunks, axis=1)                      # (draws, frame rows)

    w = (frame.n_pv * frame.p_vote).to_numpy()
    states = frame.state_fips.to_numpy()
    rows = []
    for st in np.unique(states):
        m = states == st
        draws_st = p[:, m] @ w[m] / w[m].sum()
        rows.append({"state_fips": st, "est": draws_st.mean(),
                     "lo": np.percentile(draws_st, 2.5),
                     "hi": np.percentile(draws_st, 97.5)})
    return pd.DataFrame(rows).set_index("state_fips")
