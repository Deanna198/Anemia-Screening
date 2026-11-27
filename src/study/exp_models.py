"""Experiment B: every model x every feature set, scored on the fair tests.

Run:  python src/study/exp_models.py
Out:  results/study/B_models_features.csv, B_scores.pkl (out-of-sample scores for CIs and abstention)
"""
import os, sys, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import common as C

EXTRA = {}  # deep-feature sets, added when results/study/embeddings_*.npz exist (from the Kaggle notebook)
for f in sorted(os.listdir(C.STUDY)):
    if f.startswith("embeddings_") and f.endswith(".npz"):
        z = np.load(os.path.join(C.STUDY, f), allow_pickle=True)
        EXTRA["deep: " + f[11:-4]] = dict(zip(z["subject_id"], z["emb"]))


def main():
    df = C.load("none")
    rows, keep = [], {}
    fsets = list(C.FEATURE_SETS) + list(EXTRA)
    for fs in fsets:
        if fs in EXTRA:
            if not all(s in EXTRA[fs] for s in df.subject_id): print("skip", fs, "(missing subjects)"); continue
            X = np.stack([EXTRA[fs][s] for s in df.subject_id]).astype(np.float64)
        else:
            X = C.X_of(df, fs)
        for mn, make in C.MODELS.items():
            t = time.time()
            out, sc, idx = C.fair_suite(make, X, df, keep_scores=True)
            rows.append({"features": fs, "model": mn, **out}); keep[(fs, mn)] = sc
            print(f"{fs:22s} {mn:24s} fair={out['fair average']:.3f} cross={out['cross-population average']:.3f} {time.time()-t:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(C.STUDY, "B_models_features.csv"), index=False)
    res = pd.DataFrame(rows).sort_values("fair average", ascending=False)
    res.to_csv(os.path.join(C.STUDY, "B_models_features.csv"), index=False)
    pickle.dump({"scores": keep, "idx": idx, "y": df.anemic.values.astype(int), "group": df.group_strict.values,
                 "subject_id": df.subject_id.values}, open(os.path.join(C.STUDY, "B_scores.pkl"), "wb"))
    print(res.round(3).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
