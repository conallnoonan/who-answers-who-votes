"""Build every report figure from the saved tables (no model refitting).

    python -m wawv.figures
"""
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, to_rgb
from matplotlib.transforms import blended_transform_factory
from sklearn.calibration import calibration_curve

from . import config
from .states import ABBR, STATES
from .viz import (
    CONTEXT,
    DIVERGING,
    HIGHLIGHT,
    INK,
    INK_2,
    METHOD_LABELS,
    MUTED,
    SERIES,
    apply_style,
    save,
    source_note,
    subtitle,
)

T, F = config.TAB_DIR, config.FIG_DIR
DEMO_NOTE = "  ·  SYNTHETIC DEMO DATA" if config.IS_DEMO else ""
SRC = "Source: Cooperative Election Study 2024; ACS 5-year PUMS; MIT Election Lab." + DEMO_NOTE


def _best_method() -> str:
    s = pd.read_csv(T / "rq4_accuracy_summary.csv", index_col=0)
    return "mrp_lv" if "mrp_lv" in s.index else s.index[0]


# ---------------------------------------------------------------------------
def fig1_turnout_gap():
    r = pd.read_csv(T / "rq1_overreport_mcnemar.csv").iloc[0]
    vals = [r.validated_rate_wtd, r.self_report_rate_wtd]
    labels = ["Validated in voter file", "Self-reported"]
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.barh(labels, vals, color=[CONTEXT, HIGHLIGHT], height=0.55)
    for y, v in enumerate(vals):
        ax.text(v + 0.01, y, f"{v:.0%}", va="center", fontsize=10, color=INK,
                fontweight="semibold")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(axis="y", visible=False)
    gap = (r.self_report_rate_wtd - r.validated_rate_wtd) * 100
    p = "p < .001" if r.p_value < 0.001 else f"p = {r.p_value:.3f}"
    ax.set_title("Respondents overstate turnout", pad=22)
    subtitle(ax, f"Weighted turnout, CES 2024. Gap = {gap:.1f} pts "
                 f"(McNemar χ² = {r.mcnemar_chi2:,.0f}, {p}).")
    source_note(fig, SRC)
    fig.subplots_adjust(bottom=0.22)
    save(fig, F / "fig1_turnout_gap.png")


def _clean_term(t: str) -> str:
    m = re.match(r"C\((\w+)(?:, Treatment\('?([\w-]+)'?\))?\)\[T\.([\w.\-+]+)\]", t)
    if not m:
        return t
    var, ref, lvl = m.groups()
    names = {"educ": "Education", "age_grp": "Age", "race": "Race/ethnicity",
             "pid7": "Party ID (7-pt)", "newsint": "News interest"}
    pretty = {"ba": "BA", "hs": "HS", "lt_hs": "Less than HS", "postgrad": "Postgrad",
              "some_college": "Some college", "asian": "Asian", "black": "Black",
              "hispanic": "Hispanic", "white": "White", "other": "Other"}
    lvl = lvl.removesuffix(".0")
    if var == "newsint":  # CES: 1 = most of the time ... 4 = hardly at all
        lvl = {"2": "some of the time", "3": "only now and then",
               "4": "hardly at all", "7": "don't know"}.get(lvl, lvl)
    ref = ref or {"age_grp": "18-29", "newsint": "most of the time"}.get(var, "")
    lvl, ref = pretty.get(lvl, lvl), pretty.get(ref, ref)
    return f"{names.get(var, var)}: {lvl}" + (f"  (vs {ref})" if ref else "")


def fig2_overreport_ame():
    a = pd.read_csv(T / "rq2_overreport_ame.csv")
    a = a[a.term.str.contains("educ|age_grp|race|newsint")].copy()
    a["label"] = a.term.map(_clean_term)
    a = a.iloc[::-1]
    lo, hi = a["Conf. Int. Low"] * 100, a["Cont. Int. Hi."] * 100
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(a) + 1.4))
    y = np.arange(len(a))
    ax.hlines(y, lo, hi, color=HIGHLIGHT, lw=2)
    ax.scatter(a["dy/dx"] * 100, y, s=36, color=HIGHLIGHT, zorder=3,
               edgecolor="white", linewidth=1.2)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, a.label)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in probability of claiming to have voted (pct. points)")
    ax.set_title("Who overreports?", pad=22)
    subtitle(ax, "Average marginal effects with 95% CIs, among respondents "
                 "with no validated vote.")
    source_note(fig, SRC)
    fig.subplots_adjust(bottom=0.12)
    save(fig, F / "fig2_overreport_ame.png")


def fig3_calibration():
    p = pd.read_csv(T / "rq3_test_predictions.csv")
    m = pd.read_csv(T / "rq3_turnout_metrics.csv").set_index(["spec", "model"])
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], ls=(0, (3, 3)), color=MUTED, lw=1)
    ax.text(0.30, 0.24, "perfect calibration", color=MUTED, fontsize=7.5,
            ha="center", va="center", rotation=45, rotation_mode="anchor")
    for color, name, label in [(SERIES[0], "xgb", "XGBoost"),
                               (SERIES[1], "logit", "Logistic")]:
        frac, mean_p = calibration_curve(p.y, p[f"B_poststrat__{name}"], n_bins=10,
                                         strategy="quantile")
        auc = m.loc[("B_poststrat", name), "auc"]
        ax.plot(mean_p, frac, color=color, marker="o", ms=5,
                markeredgecolor="white", label=f"{label}  (AUC {auc:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="Predicted turnout probability",
           ylabel="Observed validated turnout")
    ax.legend(loc="upper left")
    ax.set_title("Turnout model calibration", pad=22)
    subtitle(ax, "Held-out 20%, poststrat-compatible features; deciles of prediction.")
    source_note(fig, SRC)
    fig.subplots_adjust(bottom=0.16)
    save(fig, F / "fig3_turnout_calibration.png")


def fig4_method_errors():
    e = pd.read_csv(T / "rq4_state_errors.csv")
    s = pd.read_csv(T / "rq4_accuracy_summary.csv", index_col=0)
    order = s.sort_values("rmse", ascending=False).index.tolist()  # best at top
    best = _best_method()
    fig, ax = plt.subplots(figsize=(6.8, 0.6 * len(order) + 1.3))
    rng = np.random.default_rng(0)
    for i, m in enumerate(order):
        d = e[e.method == m].err.abs() * 100
        c = HIGHLIGHT if m == best else CONTEXT
        ax.scatter(d, i + rng.uniform(-0.17, 0.17, len(d)), s=14, color=c,
                   alpha=0.85, linewidth=0)
        r = s.loc[m, "rmse"] * 100
        ax.plot([r, r], [i - 0.3, i + 0.3], color=INK, lw=2)
        ax.text(1.02, i, f"RMSE {r:.1f}", va="center", fontsize=8.5,
                transform=blended_transform_factory(ax.transAxes, ax.transData),
                color=INK if m == best else INK_2,
                fontweight="semibold" if m == best else "normal")
    ax.set_yticks(range(len(order)), [METHOD_LABELS.get(m, m) for m in order])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, e.err.abs().max() * 100 * 1.04)
    ax.set_xlabel("Absolute error in Dem two-party share, by state (pct. points)")
    ax.set_title("Which correction recovers the real result?", pad=22)
    subtitle(ax, "Each dot is a state (50 + DC); black tick = RMSE across states.")
    source_note(fig, SRC)
    fig.subplots_adjust(bottom=0.17)
    save(fig, F / "fig4_method_errors.png")


def fig5_tilemap(method: str | None = None):
    method = method or _best_method()
    e = pd.read_csv(T / "rq4_state_errors.csv")
    e = STATES.merge(e[e.method == method][["state_fips", "err"]], on="state_fips",
                     how="left")
    e["pts"] = e.err * 100
    lim = max(1.0, np.nanmax(np.abs(e.pts)))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for _, r in e.iterrows():
        x, y = r.col, -r.row
        fc = DIVERGING(norm(r.pts)) if pd.notna(r.pts) else "#e1e0d9"
        ax.add_patch(plt.Rectangle((x - 0.46, y - 0.46), 0.92, 0.92, facecolor=fc,
                                   edgecolor="white", linewidth=2))
        lum = np.dot(to_rgb(fc), [0.299, 0.587, 0.114])
        tc = "white" if lum < 0.55 else INK
        ax.text(x, y + 0.12, r.abbr, ha="center", va="center", fontsize=8.5,
                fontweight="semibold", color=tc)
        if pd.notna(r.pts):
            ax.text(x, y - 0.2, f"{r.pts:+.1f}", ha="center", va="center",
                    fontsize=6.5, color=tc)
    ax.set(xlim=(-0.6, 11.6), ylim=(-7.6, 0.6), aspect="equal")
    ax.axis("off")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04,
                      pad=0.02, aspect=40, shrink=0.6, location="bottom")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, color=MUTED)
    cb.set_label("◀ overstates Republican share   |   overstates Democratic share ▶ "
                 "(pct. points)", fontsize=8, color=INK_2)
    ax.set_title(f"Where {METHOD_LABELS.get(method, method)} misses", pad=22)
    subtitle(ax, "Estimated minus certified 2024 Dem two-party share, by state.")
    source_note(fig, SRC, y=-0.04)
    save(fig, F / "fig5_error_tilemap.png")


def fig6_estimate_vs_actual(method: str | None = None):
    method = method or _best_method()
    est = pd.read_csv(T / "rq4_state_estimates.csv")
    err = pd.read_csv(T / "rq4_state_errors.csv")
    d = est[est.method == method].merge(
        err[err.method == method][["state_fips", "dem_2p", "err"]], on="state_fips")
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lims = [min(d.dem_2p.min(), d.est.min()) - 0.03, max(d.dem_2p.max(), d.est.max()) + 0.03]
    ax.plot(lims, lims, ls=(0, (3, 3)), color=MUTED, lw=1)
    if {"lo", "hi"} <= set(d.columns) and d.lo.notna().any():
        ax.vlines(d.dem_2p, d.lo, d.hi, color=HIGHLIGHT, alpha=0.35, lw=1.5)
    ax.scatter(d.dem_2p, d.est, s=30, color=HIGHLIGHT, edgecolor="white",
               linewidth=1.2, zorder=3)
    for _, r in d.reindex(d.err.abs().sort_values(ascending=False).index).head(4).iterrows():
        ax.annotate(ABBR.get(r.state_fips, r.state_fips), (r.dem_2p, r.est),
                    xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set(xlim=lims, ylim=lims, aspect="equal",
           xlabel="Certified 2024 Dem two-party share",
           ylabel=f"{METHOD_LABELS.get(method, method)} estimate")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_title("Estimates vs. certified results", pad=22)
    subtitle(ax, "Vertical bars = 95% credible intervals; labels = largest misses.")
    source_note(fig, SRC)
    fig.subplots_adjust(bottom=0.16)
    save(fig, F / "fig6_estimate_vs_actual.png")


def main() -> None:
    apply_style()
    for f in [fig1_turnout_gap, fig2_overreport_ame, fig3_calibration,
              fig4_method_errors, fig5_tilemap, fig6_estimate_vs_actual]:
        f()


if __name__ == "__main__":
    main()
