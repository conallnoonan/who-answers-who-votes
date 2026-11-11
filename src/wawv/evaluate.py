"""RQ4 evaluation: compare every method against the certified 2024 results.

    python -m wawv.evaluate [--reference mrp_lv]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from . import config
from .db import query


def bootstrap_rmse_diff(err_a, err_b, B=2000, seed=config.SEED):
    """95% CI for RMSE(a) - RMSE(b), resampling states."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(err_a), np.asarray(err_b)
    d = []
    for _ in range(B):
        i = rng.integers(0, len(a), len(a))
        d.append(np.sqrt(np.mean(a[i] ** 2)) - np.sqrt(np.mean(b[i] ** 2)))
    return np.percentile(d, [2.5, 97.5])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="mrp_lv",
                    help="method the others are tested against")
    args = ap.parse_args()

    est = pd.read_csv(config.TAB_DIR / "rq4_state_estimates.csv")
    truth = query("SELECT state_fips, dem_2p FROM mart.truth WHERE year = 2024")
    df = est.merge(truth, on="state_fips")
    df["err"] = df.est - df.dem_2p
    wide = df.pivot(index="state_fips", columns="method", values="err").dropna()
    df.to_csv(config.TAB_DIR / "rq4_state_errors.csv", index=False)

    summary = pd.DataFrame({
        "bias": wide.mean(),
        "mae": wide.abs().mean(),
        "rmse": np.sqrt((wide ** 2).mean()),
        "n_states": wide.count(),
    }).sort_values("rmse")
    print(summary.round(4))
    summary.to_csv(config.TAB_DIR / "rq4_accuracy_summary.csv")

    ref = args.reference if args.reference in wide else summary.index[0]
    tests = []
    for m in wide.columns.drop(ref):
        stat, p = wilcoxon(wide[m].abs(), wide[ref].abs())
        lo, hi = bootstrap_rmse_diff(wide[m], wide[ref])
        tests.append({"method": m, "vs": ref, "wilcoxon_W": stat, "p_value": p,
                      "rmse_diff_lo": lo, "rmse_diff_hi": hi})
    tests = pd.DataFrame(tests)
    print(tests.round(4))
    tests.to_csv(config.TAB_DIR / "rq4_paired_tests.csv", index=False)


if __name__ == "__main__":
    main()
