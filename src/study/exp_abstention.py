"""Experiment E: always answering vs. allowing "uncertain".

For a given method, the model's score is first turned into a calibrated probability (Platt scaling
fitted on the training data's own out-of-fold scores). Then:
  * Curve: keep only the most confident X% of test cases and measure accuracy on those (selective
    classification). X goes from 100% (always answer) down to 30%.
  * Operating point: the "anaemic" and "not anaemic" zones are set on training data so that each zone was
    at least 85% correct there; everything in between is "uncertain". Reported: how often the system answers,
    and how often it is right when it does.
Nothing about the test images (except their score) is used to choose thresholds.

Run:  python src/study/exp_abstention.py
Out:  results/study/E_abstention_curve.csv, E_abstention_operating_point.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import common as C

TARGET = float(os.environ.get("ZONE_TARGET", "0.85"))
COVER = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]


def fit_calibrated(make, X, y, groups=None):
    """Model + Platt calibration + decision zones, all from training data only."""
    sp = StratifiedGroupKFold(5, shuffle=True, random_state=0) if groups is not None else StratifiedKFold(5, shuffle=True, random_state=0)
    oof = np.zeros(len(y))
    for a, b in sp.split(X, y, groups):
        oof[b] = C.score(make().fit(X[a], y[a]), X[b])
    platt = LogisticRegression(C=1e6).fit(oof.reshape(-1, 1), y)
    p = platt.predict_proba(oof.reshape(-1, 1))[:, 1]
    hi, lo = 1.01, -0.01
    for t in np.unique(np.round(p, 3))[::-1]:
        m = p >= t
        if m.sum() >= 10 and y[m].mean() >= TARGET: hi = t
    for t in np.unique(np.round(p, 3)):
        m = p <= t
        if m.sum() >= 10 and (1 - y[m]).mean() >= TARGET: lo = t
    thr = 0.5
    return make().fit(X, y), platt, lo, hi, thr


def predict(model, platt, X):
    return platt.predict_proba(C.score(model, X).reshape(-1, 1))[:, 1]


def tests(df):
    S = C.site_index(df); y = df.anemic.values.astype(int)
    gh = S["Ghana"][df.usable.values[S["Ghana"]] == 1]; ed = np.r_[S["India"], S["Italy"]]
    G = df.group_strict.values
    # within-site tests: outer 5-fold CV, everything fitted inside each training fold
    within = {"Ghana (copies grouped)": (gh, G), "India": (S["India"], None), "Italy": (S["Italy"], None)}
    cross = {"Ghana → India": (gh, S["India"]), "Ghana → Italy": (gh, S["Italy"]), "India+Italy → Ghana": (ed, gh)}
    return y, G, within, cross


def run(make, X, df):
    y, G, within, cross = tests(df)
    out = {}
    for name, (idx, grp) in within.items():
        sp = StratifiedGroupKFold(5, shuffle=True, random_state=1) if grp is not None else StratifiedKFold(5, shuffle=True, random_state=1)
        p = np.zeros(len(idx)); lo_ = np.zeros(len(idx)); hi_ = np.zeros(len(idx))
        for a, b in sp.split(idx, y[idx], grp[idx] if grp is not None else None):
            tr, te = idx[a], idx[b]
            m, pl, lo, hi, _ = fit_calibrated(make, X[tr], y[tr], grp[tr] if grp is not None else None)
            p[b] = predict(m, pl, X[te]); lo_[b] = lo; hi_[b] = hi
        out[name] = (y[idx], p, lo_, hi_)
    for name, (tr, te) in cross.items():
        m, pl, lo, hi, _ = fit_calibrated(make, X[tr], y[tr], G[tr] if name.startswith("Ghana") else None)
        p = predict(m, pl, X[te]); out[name] = (y[te], p, np.full(len(te), lo), np.full(len(te), hi))
    return out


def summarise(res, label):
    curve, op = [], []
    for name, (yt, p, lo, hi) in res.items():
        conf = np.abs(p - 0.5); order = np.argsort(-conf); pred = (p >= 0.5).astype(int)
        for c in COVER:
            k = max(1, int(round(c * len(yt)))); sel = order[:k]
            curve.append({"method": label, "test": name, "coverage": c, "accuracy": (pred[sel] == yt[sel]).mean(),
                          "sensitivity": (pred[sel][yt[sel] == 1] == 1).mean() if (yt[sel] == 1).any() else np.nan,
                          "specificity": (pred[sel][yt[sel] == 0] == 0).mean() if (yt[sel] == 0).any() else np.nan})
        call = np.where(p >= hi, 1, np.where(p <= lo, 0, -1)); d = call >= 0
        op.append({"method": label, "test": name, "n": len(yt),
                   "accuracy if always answering": (pred == yt).mean(),
                   "share answered": d.mean(),
                   "accuracy when answered": (call[d] == yt[d]).mean() if d.any() else np.nan,
                   "said anaemic": int((call == 1).sum()), "said not anaemic": int((call == 0).sum()), "uncertain": int((~d).sum()),
                   "anaemic called 'not anaemic'": int(((call == 0) & (yt == 1)).sum()), "anaemic in test": int(yt.sum())})
    return curve, op


def main():
    B = pd.read_csv(os.path.join(C.STUDY, "B_models_features.csv")).sort_values("fair average", ascending=False)
    B = B[~B.features.str.startswith("deep")]
    best = B.iloc[0]
    configs = [(best.features, best.model, "best method (" + best.model + ", " + best.features + ")"),
               ("ratio colour", "logistic regression", "reference (logistic regression, ratio colour)")]
    df = C.load("none")
    curve, op = [], []
    for fs, mn, label in configs:
        c, o = summarise(run(C.MODELS[mn], C.X_of(df, fs), df), label); curve += c; op += o
    pd.DataFrame(curve).to_csv(os.path.join(C.STUDY, "E_abstention_curve.csv"), index=False)
    OP = pd.DataFrame(op); OP.to_csv(os.path.join(C.STUDY, "E_abstention_operating_point.csv"), index=False)
    pd.set_option("display.width", 250)
    print(OP.round(2).to_string(index=False))
    cv = pd.DataFrame(curve); print(cv[cv.coverage.isin([1.0, 0.7, 0.5])].pivot_table(index=["method", "test"], columns="coverage", values="accuracy").round(2))


if __name__ == "__main__":
    main()
