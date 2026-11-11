"""Interactive results explorer.

    streamlit run app/dashboard.py                 # real results
    WAWV_DATA_RAW=data/synthetic streamlit run app/dashboard.py   # demo results
"""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wawv import config  # noqa: E402
from wawv.states import STATES  # noqa: E402
from wawv.viz import CONTEXT, HIGHLIGHT, METHOD_LABELS  # noqa: E402

T = config.TAB_DIR
st.set_page_config(page_title="Who Answers, Who Votes", layout="wide")


@st.cache_data
def load():
    err = pd.read_csv(T / "rq4_state_errors.csv").merge(STATES, on="state_fips")
    err["label"] = err.method.map(lambda m: METHOD_LABELS.get(m, m))
    err["err_pts"] = err.err * 100
    acc = pd.read_csv(T / "rq4_accuracy_summary.csv", index_col=0).reset_index(names="method")
    acc["label"] = acc.method.map(lambda m: METHOD_LABELS.get(m, m))
    rq1 = pd.read_csv(T / "rq1_overreport_mcnemar.csv").iloc[0]
    return err, acc, rq1


if not (T / "rq4_state_errors.csv").exists():
    st.error(f"No results found in `{T}`. Run `make all` (or `make demo`) first.")
    st.stop()

err, acc, rq1 = load()

st.title("Who Answers, Who Votes")
st.caption("Correcting survey bias in the 2024 U.S. presidential election · "
           "CES 2024, ACS PUMS, MIT Election Lab")
if config.IS_DEMO:
    st.warning("Showing the synthetic demo run, not real findings.")

# ---- headline numbers -------------------------------------------------------
best = acc.sort_values("rmse").iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Self-reported turnout", f"{rq1.self_report_rate_wtd:.0%}")
c2.metric("Validated turnout", f"{rq1.validated_rate_wtd:.0%}",
          f"{(rq1.validated_rate_wtd - rq1.self_report_rate_wtd) * 100:.1f} pts",
          delta_color="off")
c3.metric("Best method", best.label)
c4.metric("Best state-level RMSE", f"{best.rmse * 100:.1f} pts")

# ---- controls (one row, above the charts) -----------------------------------
labels = acc.sort_values("rmse").label.tolist()
f1, f2 = st.columns([2, 3])
method_label = f1.selectbox("Estimator", labels, index=0)
region = f2.multiselect("Highlight states", sorted(STATES.abbr), default=[])
method = acc.loc[acc.label == method_label, "method"].iloc[0]
sel = err[err.method == method]

# ---- tile map ---------------------------------------------------------------
lim = max(1.0, err.err_pts.abs().max())
base = alt.Chart(sel).encode(
    x=alt.X("col:O", axis=None), y=alt.Y("row:O", axis=None),
    tooltip=[alt.Tooltip("name:N", title="State"),
             alt.Tooltip("est:Q", title="Estimate", format=".1%"),
             alt.Tooltip("dem_2p:Q", title="Certified", format=".1%"),
             alt.Tooltip("err_pts:Q", title="Error (pts)", format="+.2f")])
tiles = base.mark_rect(cornerRadius=3, stroke="white", strokeWidth=2).encode(
    color=alt.Color("err_pts:Q", title="Error (pts)",
                    scale=alt.Scale(domain=[-lim, 0, lim],
                                    range=["#b8282a", "#f0efec", "#184f95"]),
                    legend=alt.Legend(orient="bottom", gradientLength=260)),
    opacity=(alt.condition(alt.FieldOneOfPredicate(field="abbr", oneOf=region),
                           alt.value(1), alt.value(0.3)) if region else alt.value(1)))
text = base.mark_text(fontSize=11, fontWeight=600).encode(
    text="abbr:N",
    color=alt.condition(f"abs(datum.err_pts) > {lim * 0.55}", alt.value("white"),
                        alt.value("#0b0b0b")))
left, right = st.columns([3, 2])
left.subheader(f"Where {method_label} misses")
left.caption("Estimate minus certified Dem two-party share. Blue = overstates Democrats; "
             "red = overstates Republicans.")
left.altair_chart((tiles + text).properties(height=380), width="stretch")

# ---- method comparison ------------------------------------------------------
right.subheader("RMSE by method")
bars = alt.Chart(acc).mark_bar(cornerRadiusEnd=4, height=22).encode(
    y=alt.Y("label:N", sort=labels, title=None),
    x=alt.X("rmse_pts:Q", title="RMSE across states (pct. points)"),
    color=alt.condition(alt.datum.method == method, alt.value(HIGHLIGHT), alt.value(CONTEXT)),
    tooltip=[alt.Tooltip("label:N", title="Method"),
             alt.Tooltip("rmse_pts:Q", title="RMSE", format=".2f"),
             alt.Tooltip("bias_pts:Q", title="Bias", format="+.2f")],
).transform_calculate(rmse_pts="datum.rmse * 100", bias_pts="datum.bias * 100")
right.altair_chart(bars.properties(height=260), width="stretch")

# ---- state table --------------------------------------------------------------
st.subheader("All estimates, by state")
wide = (err.pivot_table(index=["abbr", "name", "dem_2p"], columns="label", values="err_pts")
          .reset_index().rename(columns={"abbr": "State", "name": "Name",
                                         "dem_2p": "Certified Dem share"}))
if region:
    wide = wide[wide.State.isin(region)]
st.dataframe(
    wide.style.format({"Certified Dem share": "{:.1%}",
                       **{c: "{:+.2f}" for c in labels}}),
    width="stretch", hide_index=True)
st.caption("Cells show estimate minus certified result, in percentage points.")
