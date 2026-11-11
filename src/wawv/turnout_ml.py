"""RQ3: predicting validated turnout.

Model B uses only the variables shared with the ACS frame, so it can be carried
into poststratification (saved to models/turnout_ps.joblib). Model A adds
attitudes (pid7, ideo5, newsint) and is for interpretation only.

    python -m wawv.turnout_ml
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from . import config
from .db import query

ATTITUDES = ["pid7", "ideo5", "newsint"]


def make_pipe(cols: list[str], clf) -> Pipeline:
    pre = ColumnTransformer(
        [("oh", OneHotEncoder(handle_unknown="ignore"), cols)])
    return Pipeline([("pre", pre), ("clf", clf)])


def classifiers() -> dict:
    return {
        "logit": LogisticRegression(max_iter=5000, C=1.0),
        "xgb": XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", random_state=config.SEED),
    }


def bootstrap_auc_diff(y, p1, p2, B=1000, seed=config.SEED):
    """95% CI for AUC(p1) - AUC(p2) on the same test set."""
    rng = np.random.default_rng(seed)
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    diffs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    return np.percentile(diffs, [2.5, 97.5])


def main() -> None:
    df = query("SELECT * FROM clean.ces WHERE sex IS NOT NULL")
    df[ATTITUDES] = df[ATTITUDES].fillna(-1)
    y = df.validated_voted
    train, test = train_test_split(df, test_size=0.2, stratify=y,
                                   random_state=config.SEED)

    rows, preds = [], {}
    for spec, cols in {"B_poststrat": config.PS_VARS,
                       "A_full": config.PS_VARS + ATTITUDES}.items():
        for name, clf in classifiers().items():
            pipe = make_pipe(cols, clf).fit(train[cols], train.validated_voted)
            p = pipe.predict_proba(test[cols])[:, 1]
            preds[(spec, name)] = p
            rows.append({"spec": spec, "model": name,
                         "auc": roc_auc_score(test.validated_voted, p),
                         "brier": brier_score_loss(test.validated_voted, p)})
            if spec == "B_poststrat" and name == "xgb":
                # refit on all data for use in poststratification
                final = make_pipe(cols, classifiers()["xgb"]).fit(df[cols], y)
                joblib.dump(final, config.MODEL_DIR / "turnout_ps.joblib")
            if spec == "A_full" and name == "xgb":
                imp = permutation_importance(pipe, test[cols], test.validated_voted,
                                             scoring="roc_auc", n_repeats=10,
                                             random_state=config.SEED)
                (pd.DataFrame({"feature": cols, "importance": imp.importances_mean,
                               "sd": imp.importances_std})
                   .sort_values("importance", ascending=False)
                   .to_csv(config.TAB_DIR / "rq3_importance_full_model.csv", index=False))

    metrics = pd.DataFrame(rows)
    lo, hi = bootstrap_auc_diff(test.validated_voted,
                                preds[("B_poststrat", "xgb")],
                                preds[("B_poststrat", "logit")])
    metrics["auc_diff_xgb_minus_logit_lo"] = lo
    metrics["auc_diff_xgb_minus_logit_hi"] = hi
    print(metrics, f"\nAUC(xgb) - AUC(logit), model B, 95% CI: [{lo:.3f}, {hi:.3f}]")
    metrics.to_csv(config.TAB_DIR / "rq3_turnout_metrics.csv", index=False)

    # Held-out predictions, used by wawv.figures for the calibration plot
    (pd.DataFrame({f"{spec}__{name}": p for (spec, name), p in preds.items()})
       .assign(y=test.validated_voted.to_numpy())
       .to_csv(config.TAB_DIR / "rq3_test_predictions.csv", index=False))

if __name__ == "__main__":
    main()
