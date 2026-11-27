"""Experiment F: estimating the Hb number itself (regression), against a do-nothing baseline.

The baseline predicts the training set's average Hb for everyone. A method only "works" for Hb
estimation if its error is clearly below that.

Run:  python src/study/exp_hb.py
Out:  results/study/F_hb_regression.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from lightgbm import LGBMRegressor
import common as C

REG = {"baseline: predict the average": lambda: DummyRegressor(),
       "ridge regression": lambda: C.sc(Ridge(10.0)),
       "SVR (RBF)": lambda: C.sc(SVR(C=3, epsilon=0.3)),
       "random forest": lambda: RandomForestRegressor(300, min_samples_leaf=3, n_jobs=2, random_state=C.SEED),
       "LightGBM": lambda: LGBMRegressor(n_estimators=200, num_leaves=8, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                                         colsample_bytree=0.8, n_jobs=2, random_state=C.SEED, verbose=-1)}


def cv_mae(make, X, hb, idx, groups=None):
    sp = GroupKFold(5) if groups is not None else KFold(5, shuffle=True, random_state=C.SEED)
    pred = np.zeros(len(idx))
    for a, b in sp.split(idx, hb[idx], groups[idx] if groups is not None else None):
        pred[b] = make().fit(X[idx[a]], hb[idx[a]]).predict(X[idx[b]])
    return np.abs(pred - hb[idx]).mean(), np.corrcoef(pred, hb[idx])[0, 1]


def main():
    df = C.load("none"); hb = df.hb.values; S = C.site_index(df)
    gh = S["Ghana"][df.usable.values[S["Ghana"]] == 1]; ed = np.r_[S["India"], S["Italy"]]
    rows = []
    for fs in ["ratio colour", "colour + texture"]:
        X = C.X_of(df, fs)
        for mn, make in REG.items():
            r = {"features": fs, "model": mn}
            for name, idx, g in (("Ghana (copies grouped)", gh, df.group_strict.values), ("India", S["India"], None), ("Italy", S["Italy"], None)):
                r[f"{name}: MAE g/dL"], r[f"{name}: correlation"] = cv_mae(make, X, hb, idx, g)
            for name, tr, te in (("Ghana → India+Italy", gh, ed), ("India+Italy → Ghana", ed, gh)):
                p = make().fit(X[tr], hb[tr]).predict(X[te])
                r[f"{name}: MAE g/dL"] = np.abs(p - hb[te]).mean()
                r[f"{name}: correlation"] = np.corrcoef(p, hb[te])[0, 1] if p.std() > 0 else np.nan
            rows.append(r); print(fs, mn, flush=True)
    res = pd.DataFrame(rows); res.to_csv(os.path.join(C.STUDY, "F_hb_regression.csv"), index=False)
    print(res.round(2).T.to_string())


if __name__ == "__main__":
    main()
