"""RQ4: five ways to estimate each state's 2024 Dem two-party share.

  raw        unweighted mean among self-reported voters
  ces_wt     CES's own post-election weight
  raked_lv   raked to ACS adult-citizen margins x predicted turnout
  mrp_lv     multilevel model (bambi/PyMC), poststratified with N x p(turnout)
  xgb_lv     XGBoost vote model, poststratified with N x p(turnout)

    python -m wawv.estimators [--draws 1000] [--skip-mrp]
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from . import config
from .db import query

PS = config.PS_VARS


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def kish_deff(w) -> float:
    """Kish design effect from unequal weighting: n * sum(w^2) / sum(w)^2."""
    w = np.asarray(w, float)
    return len(w) * np.sum(w ** 2) / np.sum(w) ** 2


def rake(df: pd.DataFrame, margins: dict, w0=None, iters: int = 100,
         tol: float = 1e-6, cap: float = 5.0) -> pd.Series:
    """Iterative proportional fitting.

    margins: {column: Series(level -> population count or share)}.
    Weights are capped at `cap` x mean each pass and returned with mean 1.
    """
    w = pd.Series(1.0 if w0 is None else np.asarray(w0, float), index=df.index)
    for _ in range(iters):
        w_old = w.copy()
        for var, target in margins.items():
            cur = w.groupby(df[var]).sum() / w.sum()
            t = target.reindex(cur.index).fillna(0)
            t = t / t.sum()
            w = w * df[var].map(t / cur).fillna(1.0)
        w = w.clip(upper=cap * w.mean())
        if (w - w_old).abs().max() < tol:
            break
    return w / w.mean()


def poststratify(ps: pd.DataFrame, p_col: str, w_col: str = "w") -> pd.Series:
    g = ps.assign(_pw=ps[p_col] * ps[w_col]).groupby("state_fips")
    return g._pw.sum() / g[w_col].sum()


def load_inputs():
    ces = query("""SELECT c.*, s.dem_2020_c FROM clean.ces c
                   JOIN mart.state_covs s USING (state_fips)
                   WHERE c.sex IS NOT NULL""")
    ps = query("""SELECT p.*, s.dem_2020_c FROM mart.poststrat p
                  JOIN mart.state_covs s USING (state_fips)""")
    margins = query("SELECT * FROM mart.ps_margins")
    turnout = joblib.load(config.MODEL_DIR / "turnout_ps.joblib")
    ces["p_vote"] = turnout.predict_proba(ces[PS])[:, 1]
    ps["p_vote"] = turnout.predict_proba(ps[PS])[:, 1]
    ps["w"] = ps.n * ps.p_vote          # expected voters in each cell
    return ces, ps, margins


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------
def est_raw(ces):
    v = ces.dropna(subset=["vote_dem"])
    return v.groupby("state_fips").vote_dem.mean()


def est_ces_weight(ces):
    v = ces.dropna(subset=["vote_dem"])
    g = v.assign(x=v.vote_dem * v.ces_wt).groupby("state_fips")
    return g.x.sum() / g.ces_wt.sum()


def est_raked_lv(ces, margins):
    out, deffs = {}, {}
    for st, d in ces.groupby("state_fips"):
        m = margins[margins.state_fips == st]
        mg = {var: m[m["var"] == var].set_index("level").n
              for var in ["age_grp", "sex", "race", "educ"]}
        w = rake(d, mg) * d.p_vote
        v = d.vote_dem.notna()
        out[st] = np.average(d.vote_dem[v], weights=w[v])
        deffs[st] = kish_deff(w[v])
    pd.Series(deffs, name="kish_deff").rename_axis("state_fips").to_csv(
        config.TAB_DIR / "rq4_raking_deff.csv")
    return pd.Series(out)


def est_xgb_lv(ces, ps):
    v = ces.dropna(subset=["vote_dem"])
    cols = PS + ["dem_2020_c"]
    X = pd.get_dummies(v[cols].astype({c: str for c in PS}), columns=PS)
    Xp = (pd.get_dummies(ps[cols].astype({c: str for c in PS}), columns=PS)
            .reindex(columns=X.columns, fill_value=0))
    clf = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                        subsample=0.8, eval_metric="logloss",
                        random_state=config.SEED).fit(X, v.vote_dem.astype(int))
    ps = ps.assign(p_dem=clf.predict_proba(Xp)[:, 1])
    return poststratify(ps, "p_dem")


def save_mrp_diagnostics(idata) -> None:
    """Convergence table (R-hat, ESS, divergences) + full posterior for the notebook."""
    import arviz as az

    summ = az.summary(idata, kind="diagnostics")
    n_div = int(idata.sample_stats["diverging"].sum())
    pd.DataFrame([{
        "max_rhat": summ.r_hat.max(),
        "min_ess_bulk": summ.ess_bulk.min(),
        "min_ess_tail": summ.ess_tail.min(),
        "divergences": n_div,
        "n_params": len(summ),
    }]).to_csv(config.TAB_DIR / "rq4_mrp_diagnostics.csv", index=False)
    summ.to_csv(config.TAB_DIR / "rq4_mrp_param_diagnostics.csv")
    try:
        idata.to_netcdf(str(config.MODEL_DIR / "mrp_idata.nc"), engine="h5netcdf")
    except Exception as e:  # saving draws is a convenience, not a requirement
        print(f"  (could not save posterior draws: {e})")


def est_mrp_lv(ces, ps, draws=1000, chains=4, max_pred_draws=400, chunk_size=1500):
    import bambi as bmb  # heavy import; only load when needed

    v = ces.dropna(subset=["vote_dem"])
    cells = (v.groupby(PS + ["dem_2020_c"]).vote_dem
               .agg(dem="sum", trials="count").reset_index())
    cells[PS] = cells[PS].astype(str)
    model = bmb.Model(
        "p(dem, trials) ~ dem_2020_c + (1|state_fips) + (1|age_grp) + (1|sex) "
        "+ (1|race) + (1|educ) + (1|race:educ) + (1|age_grp:educ)",
        cells, family="binomial")
    idata = model.fit(draws=draws, tune=draws, chains=chains,
                      target_accept=0.95, random_seed=config.SEED)
    save_mrp_diagnostics(idata)

    # Predict in chunks on a thinned posterior: (draws x cells) arrays for every
    # group term otherwise get large fast (4,000 draws x 30k cells ~ 1 GB each).
    step = max(1, idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"] // max_pred_draws)
    thin = idata.sel(draw=slice(None, None, step))
    new = ps.copy()
    new[PS] = new[PS].astype(str)
    new["trials"] = 1
    chunks = []
    for start in range(0, len(new), chunk_size):
        part = new.iloc[start:start + chunk_size]
        pred = model.predict(thin, data=part, kind="response_params", inplace=False)
        # the response-parameter array is the one with one value per new row
        var = [k for k, a in pred.posterior.data_vars.items()
               if a.ndim == 3 and a.shape[-1] == len(part)][-1]
        chunks.append(pred.posterior[var].values.reshape(-1, len(part)))
        del pred
    p = np.concatenate(chunks, axis=1)                          # (draws, cells)

    w = ps.w.to_numpy()
    states = ps.state_fips.to_numpy()
    rows = []
    for st in np.unique(states):
        m = states == st
        draws_st = p[:, m] @ w[m] / w[m].sum()
        rows.append({"state_fips": st, "est": draws_st.mean(),
                     "lo": np.percentile(draws_st, 2.5),
                     "hi": np.percentile(draws_st, 97.5)})
    return pd.DataFrame(rows).set_index("state_fips")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--skip-mrp", action="store_true")
    args = ap.parse_args()

    ces, ps, margins = load_inputs()
    ests = {
        "raw": est_raw(ces),
        "ces_wt": est_ces_weight(ces),
        "raked_lv": est_raked_lv(ces, margins),
        "xgb_lv": est_xgb_lv(ces, ps),
    }
    frames = [s.rename("est").rename_axis("state_fips").reset_index().assign(method=k)
              for k, s in ests.items()]
    if not args.skip_mrp:
        mrp = est_mrp_lv(ces, ps, draws=args.draws, chains=args.chains)
        frames.append(mrp.reset_index().assign(method="mrp_lv"))

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(config.TAB_DIR / "rq4_state_estimates.csv", index=False)
    print(out.groupby("method").est.describe())


if __name__ == "__main__":
    main()
