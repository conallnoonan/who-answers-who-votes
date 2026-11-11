"""Regenerate notebooks/01_mrp_diagnostics.ipynb (kept as code so diffs stay readable).

    python notebooks/make_notebook.py
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# MRP convergence diagnostics\n\n"
       "Checks that the multilevel model in `wawv.estimators` converged before its "
       "estimates are reported. Run `make estimate` first; it saves the posterior to "
       "`models/mrp_idata.nc`.\n\n"
       "**Targets:** R̂ < 1.01 for every parameter, bulk and tail ESS > 400, zero "
       "divergent transitions."),
    code("import sys; sys.path.insert(0, '../src')\n"
         "import arviz as az\n"
         "import matplotlib.pyplot as plt\n"
         "from wawv import config\n"
         "from wawv.states import ABBR\n"
         "from wawv.viz import apply_style\n"
         "apply_style()\n"
         "idata = az.from_netcdf(config.MODEL_DIR / 'mrp_idata.nc')\n"
         "idata"),
    md("## 1. Summary table\nWorst R̂ and lowest ESS first."),
    code("summ = az.summary(idata, kind='diagnostics')\n"
         "print('divergences:', int(idata.sample_stats['diverging'].sum()))\n"
         "summ.sort_values('r_hat', ascending=False).head(15)"),
    md("## 2. Group-level standard deviations\nThese hierarchical scale parameters are "
       "the hardest to sample. Each chain's trace should look like overlapping noise."),
    code("post = idata.posterior\n"
         "sigmas = [v for v in post.data_vars if v.endswith('_sigma')]\n"
         "fig, axes = plt.subplots(len(sigmas), 2, figsize=(10, 1.8 * len(sigmas)),\n"
         "                         gridspec_kw={'width_ratios': [3, 1]})\n"
         "for row, v in zip(axes, sigmas):\n"
         "    for c in post.chain.values:\n"
         "        x = post[v].sel(chain=c).values\n"
         "        row[0].plot(x, lw=0.6, alpha=0.8)\n"
         "        row[1].hist(x, bins=30, alpha=0.5, orientation='horizontal')\n"
         "    row[0].set_ylabel(v.replace('_sigma', ''), rotation=0, ha='right')\n"
         "fig.suptitle('Traces and marginals of group SDs')\n"
         "fig.tight_layout()"),
    md("## 3. Energy\nThe marginal energy and the energy transitions should have similar "
       "spread; a much narrower transition distribution means the sampler struggles to "
       "reach the tails (low E-BFMI)."),
    code("import numpy as np\n"
         "E = idata.sample_stats['energy'].values\n"
         "dE = np.diff(E, axis=1)\n"
         "bfmi = (dE ** 2).mean(axis=1) / E.var(axis=1)\n"
         "print('E-BFMI by chain (want > 0.3):', bfmi.round(2))\n"
         "fig, ax = plt.subplots(figsize=(6, 3))\n"
         "ax.hist((E - E.mean()).ravel(), bins=50, alpha=0.6, label='marginal energy')\n"
         "ax.hist(dE.ravel(), bins=50, alpha=0.6, label='energy transition')\n"
         "ax.legend()"),
    md("## 4. State effects\nPosterior state intercepts (log-odds), with 95% intervals: "
       "partial pooling toward the 2020-vote prediction."),
    code("st = post['1|state_fips']\n"
         "dim = [d for d in st.dims if d not in ('chain', 'draw')][0]\n"
         "draws = st.stack(s=('chain', 'draw')).transpose(dim, 's').values\n"
         "labels = [ABBR.get(int(float(x)), x) for x in st[dim].values]\n"
         "order = np.argsort(draws.mean(1))\n"
         "lo, hi = np.percentile(draws, [2.5, 97.5], axis=1)\n"
         "fig, ax = plt.subplots(figsize=(5, 11))\n"
         "y = np.arange(len(order))\n"
         "ax.hlines(y, lo[order], hi[order], lw=1.5)\n"
         "ax.scatter(draws.mean(1)[order], y, s=14, zorder=3)\n"
         "ax.set_yticks(y, [labels[i] for i in order], fontsize=7)\n"
         "ax.axvline(0, color='grey', lw=1)\n"
         "ax.set_title('State random intercepts')"),
    md("## What to do if checks fail\n"
       "- Increase draws: `make estimate DRAWS=2000`\n"
       "- Raise `target_accept` to 0.99 in `est_mrp_lv`\n"
       "- Drop the weakest interaction term (often `age_grp:educ`) and refit"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {
    "name": "python3", "display_name": "Python 3", "language": "python"}})
out = Path(__file__).parent / "01_mrp_diagnostics.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")
