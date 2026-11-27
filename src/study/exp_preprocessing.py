"""Experiment C: does colour correction help? (replaces the physical colour card)

Methods: none, grey-world, white-patch, sclera reference (white of the eye), and per-site standardisation.
The first four need the whole eye photo, so they are tested on Eyes-Defy only (India, Italy, and India↔Italy
transfer, which is where lighting/camera differences between sites matter most).
Per-site standardisation needs no photo, so it is also tested on the Ghana transfers.

Run:  python src/study/exp_preprocessing.py
Out:  results/study/C_preprocessing.csv
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import common as C

MODELS = ["logistic regression", "random forest", "LightGBM"]
FSETS = ["ratio colour", "ratio + percentiles", "colour + texture"]


def site_z(X, df):
    Z = X.copy()
    for s, i in C.site_index(df).items():
        Z[i] = (X[i] - X[i].mean(0)) / (X[i].std(0) + 1e-9)
    return Z


def main():
    rows = []
    for meth in ["none", "grayworld", "whitepatch", "sclera", "per-site standardisation"]:
        df = C.load("none" if meth == "per-site standardisation" else meth)
        y = df.anemic.values.astype(int); S = C.site_index(df)
        gh = S["Ghana"][df.usable.values[S["Ghana"]] == 1]; ed = np.r_[S["India"], S["Italy"]]
        for fs in FSETS:
            X = C.X_of(df, fs)
            if meth == "per-site standardisation": X = site_z(X, df)
            for mn in MODELS:
                make = C.MODELS[mn]; r = {"colour correction": meth, "features": fs, "model": mn}
                r["India"] = C.cv_test(make, X, y, S["India"])[0]
                r["Italy"] = C.cv_test(make, X, y, S["Italy"])[0]
                r["India → Italy"] = C.transfer_test(make, X, y, S["India"], S["Italy"])[0]
                r["Italy → India"] = C.transfer_test(make, X, y, S["Italy"], S["India"])[0]
                r["Ghana → India"] = C.transfer_test(make, X, y, gh, S["India"])[0]
                r["Ghana → Italy"] = C.transfer_test(make, X, y, gh, S["Italy"])[0]
                r["India+Italy → Ghana"] = C.transfer_test(make, X, y, ed, gh)[0]
                r["Eyes-Defy average (4 tests)"] = np.mean([r[k] for k in ("India", "Italy", "India → Italy", "Italy → India")])
                rows.append(r); print(meth, fs, mn, round(r["Eyes-Defy average (4 tests)"], 3), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(C.STUDY, "C_preprocessing.csv"), index=False)
    print(res.groupby("colour correction")[["India", "Italy", "India → Italy", "Italy → India", "Eyes-Defy average (4 tests)"]].mean().round(3))


if __name__ == "__main__":
    main()
