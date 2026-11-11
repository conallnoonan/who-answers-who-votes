"""Generate fake raw files with the same layout as the real ones, so the whole
pipeline can be run end to end before downloading anything.

    python -m wawv.synth            # writes to data/synthetic/
"""

import numpy as np
import pandas as pd
import pyreadstat

from . import config

OUT = config.ROOT / "data" / "synthetic"
FIPS = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56]
rng = np.random.default_rng(config.SEED)
inv = lambda x: 1 / (1 + np.exp(-x))
LEAN = dict(zip(FIPS, rng.normal(0, 0.5, len(FIPS))))  # true state partisanship


def people(n):
    """Draw demographics in ACS coding plus latent lean/turnout propensity."""
    st = rng.choice(FIPS, n)
    age = rng.integers(18, 90, n)
    sex = rng.integers(1, 3, n)
    rac1p = rng.choice([1, 2, 6, 8], n, p=[0.68, 0.13, 0.07, 0.12])
    hisp = np.where(rng.random(n) < 0.15, 2, 1)
    schl = rng.choice([14, 16, 19, 21, 22], n, p=[0.1, 0.27, 0.29, 0.21, 0.13])
    ed = np.select([schl <= 15, schl <= 17, schl <= 20, schl == 21], [0, 1, 2, 3], 4)
    turn_lp = -1.2 + 0.035 * (age - 18) + 0.35 * ed - 0.3 * (rac1p == 8)
    dem_lp = (np.vectorize(LEAN.get)(st) + 0.25 * (ed - 2) + 1.6 * (rac1p == 2)
              + 0.5 * (hisp == 2) - 0.01 * (age - 50) + 0.2 * (sex == 2))
    return pd.DataFrame(dict(ST=st, AGEP=age, SEX=sex, RAC1P=rac1p, HISP=hisp,
                             SCHL=schl, ed=ed, turn_lp=turn_lp, dem_lp=dem_lp))


def main() -> None:
    (OUT / "pums").mkdir(parents=True, exist_ok=True)

    # ---- "ACS PUMS" ------------------------------------------------------
    pop = people(80_000)
    pop["CIT"] = np.where(rng.random(len(pop)) < 0.07, 5, 1)
    pop["PWGTP"] = rng.integers(20, 200, len(pop))
    pop[["ST", "AGEP", "SEX", "RAC1P", "HISP", "SCHL", "CIT", "PWGTP"]].to_csv(
        OUT / "pums" / "psam_pusa.csv", index=False)

    # ---- "MIT Election Lab" truth (from the population model) ------------
    p = pop[pop.CIT != 5]
    voted = rng.random(len(p)) < inv(p.turn_lp)
    dem = rng.random(len(p)) < inv(p.dem_lp)
    rows = []
    for yr, shift in [(2020, 0.15), (2024, 0.0)]:
        for st in FIPS:
            m = (p.ST == st) & voted
            w = p.PWGTP[m]
            d = (w * dem[m]).sum() * np.exp(shift)
            r = (w * ~dem[m]).sum()
            rows += [dict(year=yr, state_fips=st, party_simplified="DEMOCRAT",
                          candidatevotes=int(d), totalvotes=int(d + r)),
                     dict(year=yr, state_fips=st, party_simplified="REPUBLICAN",
                          candidatevotes=int(r), totalvotes=int(d + r))]
    pd.DataFrame(rows).to_csv(OUT / "1976-2024-president.csv", index=False)

    # ---- "CES" -------------------------------------------------------------
    # Over-educated, politically interested sample, like a real opt-in panel.
    s = people(30_000)
    keep = rng.random(len(s)) < inv(-0.5 + 0.4 * s.ed)
    s = s[keep].sample(8_000, random_state=config.SEED).reset_index(drop=True)
    n = len(s)
    newsint = rng.integers(1, 5, n)
    validated = rng.random(n) < inv(s.turn_lp + 0.3 * (4 - newsint))
    overreport = (~validated) & (rng.random(n) < inv(-1 + 0.3 * s.ed + 0.4 * (4 - newsint)))
    self_voted = validated | overreport
    dem = rng.random(n) < inv(s.dem_lp)
    race = np.select([s.HISP == 2, s.RAC1P == 1, s.RAC1P == 2, s.RAC1P == 6], [3, 1, 2, 4], 8)
    ces = pd.DataFrame({
        "caseid": np.arange(n),
        "inputstate": s.ST,
        "birthyr": 2024 - s.AGEP,
        "gender4": np.where(rng.random(n) < 0.01, 3, s.SEX),
        "race": race,
        "hispanic": np.where(s.HISP == 2, 1, 2),
        "educ": s.ed.map({0: 1, 1: 2, 2: 3, 3: 5, 4: 6}),
        "pid7": np.clip(np.round(4 - 2 * (s.dem_lp + rng.normal(0, 1, n))), 1, 7),
        "ideo5": rng.integers(1, 6, n),
        "newsint": newsint,
        "commonpostweight": rng.lognormal(0, 0.5, n),
        "cc24_401": np.where(self_voted, 5, 1),
        "cc24_410": np.where(self_voted, np.where(dem, 1, 2), np.nan),
        "ts_g2024": np.where(validated, 3, np.nan),
    }).astype(float)
    pyreadstat.write_dta(ces, str(OUT / "CES24_Common.dta"))
    print(f"Synthetic raw data written to {OUT}")


if __name__ == "__main__":
    main()
