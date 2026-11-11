import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wawv.estimators import kish_deff, poststratify, rake  # noqa: E402


def test_rake_hits_margins():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.choice(["x", "y"], 2000, p=[0.8, 0.2]),
                       "b": rng.choice(["u", "v", "w"], 2000)})
    targets = {"a": pd.Series({"x": 50, "y": 50}),
               "b": pd.Series({"u": 20, "v": 30, "w": 50})}
    w = rake(df, targets, cap=100)
    for var, t in targets.items():
        got = w.groupby(df[var]).sum() / w.sum()
        assert np.allclose(got.sort_index(), (t / t.sum()).sort_index(), atol=1e-3)
    assert np.isclose(w.mean(), 1)


def test_kish_deff_equal_weights_is_one():
    assert np.isclose(kish_deff(np.ones(10)), 1.0)
    assert kish_deff([1, 1, 1, 5]) > 1


def test_poststratify_weighted_mean():
    ps = pd.DataFrame({"state_fips": [1, 1, 2], "p": [0.2, 0.8, 0.5], "w": [3, 1, 2]})
    out = poststratify(ps, "p")
    assert np.isclose(out[1], (0.2 * 3 + 0.8 * 1) / 4)
    assert np.isclose(out[2], 0.5)


def test_tile_grid_covers_50_states_plus_dc_without_overlap():
    from wawv.states import STATES
    assert len(STATES) == 51
    assert not STATES.duplicated(["row", "col"]).any()


def test_term_labels_are_readable():
    from wawv.figures import _clean_term
    assert _clean_term("C(educ, Treatment('hs'))[T.ba]") == "Education: BA  (vs HS)"
    assert _clean_term("C(newsint)[T.4.0]") == "News interest: hardly at all  (vs most of the time)"
