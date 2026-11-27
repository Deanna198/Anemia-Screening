"""Experiment A: how much does the evaluation method change the score?

Same models and features, four ways of testing on Ghana (CP-AnemiC) and three on Eyes-Defy:
  Ghana, random split, all 710 images     - what most published CP-AnemiC papers do
  Ghana, random split, exact copies removed (392)
  Ghana, copies grouped - strict           - near-duplicates (mirrored / re-cropped) kept on one side
  Ghana, copies grouped - lenient          - looser near-duplicate matching
  Eyes-Defy pooled (India+Italy scored together)
  Eyes-Defy pooled training, scored per site
  Within each site (trained and tested on one site)
  Cross-population (train on one population, test on another) - from experiment B

Run:  python src/study/exp_protocol.py
Out:  results/study/A_protocol.csv
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import common as C

MODELS = ["logistic regression", "random forest", "XGBoost", "SVM (RBF)"]
FSETS = ["ratio colour", "colour + texture"]


def main():
    df = C.load("none"); y = df.anemic.values.astype(int); S = C.site_index(df)
    gh_all = S["Ghana"]; gh = gh_all[df.usable.values[gh_all] == 1]
    ed = np.r_[S["India"], S["Italy"]]
    rows = []
    for fs in FSETS:
        X = C.X_of(df, fs)
        for mn in MODELS:
            make = C.MODELS[mn]; t = time.time(); r = {"features": fs, "model": mn}
            r["Ghana: random split, all 710 images"] = C.cv_test(make, X, y, gh_all)[0]
            r["Ghana: random split, exact copies removed"] = C.cv_test(make, X, y, gh)[0]
            r["Ghana: copies grouped (strict)"] = C.cv_test(make, X, y, gh, df.group_strict.values)[0]
            r["Ghana: copies grouped (lenient)"] = C.cv_test(make, X, y, gh, df.group_lenient.values)[0]
            _, _, s = C.cv_test(make, X, y, ed)
            r["Eyes-Defy: pooled, scored together"] = C.auc(y[ed], s)
            n_in = len(S["India"])
            r["Eyes-Defy: pooled, scored on India"] = C.auc(y[S["India"]], s[:n_in])
            r["Eyes-Defy: pooled, scored on Italy"] = C.auc(y[S["Italy"]], s[n_in:])
            r["India: within site"] = C.cv_test(make, X, y, S["India"])[0]
            r["Italy: within site"] = C.cv_test(make, X, y, S["Italy"])[0]
            r["Ghana → India"] = C.transfer_test(make, X, y, gh, S["India"])[0]
            r["Ghana → Italy"] = C.transfer_test(make, X, y, gh, S["Italy"])[0]
            r["India+Italy → Ghana"] = C.transfer_test(make, X, y, ed, gh)[0]
            rows.append(r); print(fs, mn, f"{time.time()-t:.0f}s", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(C.STUDY, "A_protocol.csv"), index=False)
    print(res.round(3).T.to_string())


if __name__ == "__main__":
    main()
