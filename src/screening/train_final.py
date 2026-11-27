"""Train the models used by the demo app, on all usable images from all three sites.

Methods shown in the app (chosen from the comparison study, results/study/):
  main answer : logistic regression on ratio colour features - best on the cross-population tests,
                which is the situation the app is in (a new phone, new people)
  also shown  : the other top methods from the study, for the side-by-side comparison

Each model gets a probability calibration and an "uncertain" zone, both set from the training data's own
out-of-fold scores (see src/study/exp_abstention.py).

Run:  python src/screening/train_final.py
Out:  models/screening_models.joblib
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "study"))
import numpy as np, joblib
import common as C
import exp_abstention as EA
from exp_abstention import fit_calibrated

ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.environ.get("ANEMIA_MODELS", os.path.join(ROOT, "models"))

METHODS = [  # (label in app, feature set, model)
    ("Logistic regression · ratio colour (main)", "ratio colour", "logistic regression"),
    ("Extra trees · ratio colour", "ratio colour", "extra trees"),
    ("Random forest · all features", "everything", "random forest"),
    ("LightGBM · colour + texture", "colour + texture", "LightGBM"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    df = C.load("none")
    keep = (df.site != "Ghana") | (df.usable == 1)
    d = df[keep].reset_index(drop=True); y = d.anemic.values.astype(int)
    groups = d.group_strict.values
    bundle = {"methods": [], "trained_on": {s: int((d.site == s).sum()) for s in ("Ghana", "India", "Italy")}}
    for label, fs, mn in METHODS:
        cols = sum((C.cols(d, f) for f in C.FEATURE_SETS[fs]), [])
        X = np.nan_to_num(d[cols].values.astype(np.float64))
        # each answer zone must have been at least 85% correct on training data; if no "anaemic" zone reaches
        # that, relax to 80%, then 75%, and record which level was used
        for target in (0.85, 0.80, 0.75):
            EA.TARGET = target
            model, platt, lo, hi, _ = fit_calibrated(C.MODELS[mn], X, y, groups)
            if hi <= 1.0 and lo >= 0.0: break
        bundle["methods"].append({"label": label, "features": cols, "model": model, "platt": platt, "lo": float(lo), "hi": float(hi),
                                  "zone_target": target})
        print(f"{label:45s} uncertain zone: {lo:.2f} - {hi:.2f}  (zones {target:.0%} correct on training data)")
    joblib.dump(bundle, os.path.join(OUT, "screening_models.joblib"))
    print("saved", os.path.join(OUT, "screening_models.joblib"), json.dumps(bundle["trained_on"]))


if __name__ == "__main__":
    main()
