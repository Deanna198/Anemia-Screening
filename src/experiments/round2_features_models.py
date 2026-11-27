"""Second round: richer colour features, age/sex, site standardisation, leave-one-site-out,
and a probability-based three-way answer. Classical models only (CPU)."""
import os, sys, warnings, json, time
import numpy as np, pandas as pd, cv2
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
SEED = 42
DATA = os.environ.get("ANEMIA_DATA", "/home/claude/work/input/anemia-data")
N_REP = int(os.environ.get("N_REP", "5"))

df = pd.read_csv(os.path.join(DATA, "features.csv"))
df = df[(df.usable == 1) & df.hb.notna()].reset_index(drop=True)
y = df.anemic.values.astype(int); hb = df.hb.values; cut = df.who_cutoff.values; site = df.site.values
CP = np.where(site == "Ghana")[0]; IN = np.where(site == "India")[0]; IT = np.where(site == "Italy")[0]
ED = np.concatenate([IN, IT])

# ---------- richer colour features ----------
def pixels(path):
    im = cv2.imread(os.path.join(DATA, path), cv2.IMREAD_UNCHANGED)
    h, w = im.shape[:2]; s = 600 / max(h, w)
    if s < 1: im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    mask = (im[:, :, 3] > 200).astype(np.uint8)
    k = max(3, int(0.01 * max(im.shape[:2])) | 1)
    inner = cv2.erode(mask, np.ones((k, k), np.uint8))
    if inner.sum() >= 200: mask = inner
    px = (im[:, :, 2::-1].astype(np.float32) / 255.0)[mask > 0]
    v = px.max(1); sat = (v - px.min(1)) / np.maximum(v, 1e-6)
    keep = ~((v > 0.92) & (sat < 0.25)) & (v > 0.08)
    return px[keep] if keep.sum() >= 100 else px

def rich(px):
    f = {}
    r, g, b = px[:, 0] + 1e-3, px[:, 1] + 1e-3, px[:, 2] + 1e-3
    lab = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).reshape(-1, 3)
    chans = {"logRG": np.log(r / g), "logRB": np.log(r / b), "logGB": np.log(g / b),
             "a": lab[:, 1], "b": lab[:, 2], "rch": r / (r + g + b), "gch": g / (r + g + b)}
    for n, c in chans.items():
        for q in (10, 25, 50, 75, 90):
            f[f"{n}_p{q}"] = np.percentile(c, q)
        f[f"{n}_iqr"] = f[f"{n}_p75"] - f[f"{n}_p25"]
    # the reddest third of the tissue (vessel-rich part) and the palest third
    e = chans["logRG"]; o = np.argsort(e); n3 = max(30, len(o) // 3)
    f["red_top_logRG"] = e[o[-n3:]].mean(); f["red_low_logRG"] = e[o[:n3]].mean()
    f["red_top_a"] = lab[o[-n3:], 1].mean(); f["red_low_a"] = lab[o[:n3], 1].mean()
    f["red_spread"] = f["red_top_logRG"] - f["red_low_logRG"]
    return f

t0 = time.time()
R = pd.DataFrame([rich(pixels(p)) for p in df.conj_path])
print("rich features", R.shape, f"{time.time()-t0:.0f}s", flush=True)

RATIO = [c for c in df.columns if c.startswith("ratio_")]
RAW = [c for c in df.columns if c.startswith("raw_")]
age = df.age_years.fillna(df.age_years.median()).values
male = (df.sex.astype(str).str.upper().str[0] == "M").astype(float).values
DEMO = np.c_[np.log1p(age), male, (age < 5).astype(float), (age < 15).astype(float)]
FE = {"ratio": df[RATIO].values, "all": df[RATIO + RAW].values, "rich": R.values,
      "ratio+rich": np.c_[df[RATIO].values, R.values]}
FE = {k: v.astype(np.float64) for k, v in FE.items()}

def site_z(X, idx_groups):
    """Standardise each group of rows by its own mean/sd (uses no labels)."""
    Z = X.copy()
    for g in idx_groups:
        Z[g] = (X[g] - X[g].mean(0)) / (X[g].std(0) + 1e-9)
    return Z

MODELS = {"logistic": lambda: Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000))]),
          "logistic_cv": lambda: Pipeline([("sc", StandardScaler()), ("m", LogisticRegressionCV(Cs=[0.01, 0.03, 0.1, 0.3, 1, 3], cv=4, class_weight="balanced", scoring="roc_auc", max_iter=3000))]),
          "rf": lambda: RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=SEED)}

def proba(m, X): return m.predict_proba(X)[:, 1]
def auc(i, s): return roc_auc_score(y[i], s) if len(np.unique(y[i])) == 2 else np.nan

rs = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REP, random_state=SEED)
strat = np.array([f"{a}_{b}" for a, b in zip(site[ED], y[ED])])
WITHIN = {"CP": [(CP[a], CP[b]) for a, b in rs.split(CP, y[CP])],
          "ED": [(ED[a], ED[b]) for a, b in rs.split(ED, strat)]}

def six(make, X):
    """Six headline AUCs + 3 leave-one-site-out AUCs. Returns dict and out-of-sample probabilities."""
    out = {}; keep = {}
    for name, folds in WITHIN.items():
        reps = {"Ghana": [], "India": [], "Italy": []}
        for rep in range(len(folds) // 5):
            fs = folds[rep * 5:(rep + 1) * 5]; idx = np.concatenate([te for _, te in fs]); p = np.zeros(len(idx)); pos = 0
            for tr, te in fs:
                p[pos:pos + len(te)] = proba(make().fit(X[tr], y[tr]), X[te]); pos += len(te)
            if rep == 0: keep["within_" + name] = (idx, p)
            for s in (["Ghana"] if name == "CP" else ["India", "Italy"]):
                m = site[idx] == s; reps[s].append(auc(idx[m], p[m]))
        for s, v in reps.items():
            if v: out["within_" + s] = float(np.mean(v))
    m = make().fit(X[CP], y[CP]); p = proba(m, X[ED]); keep["CP_to_ED"] = (ED, p)
    out["Ghana>India"] = auc(IN, p[:len(IN)]); out["Ghana>Italy"] = auc(IT, p[len(IN):])
    m = make().fit(X[ED], y[ED]); p = proba(m, X[CP]); keep["ED_to_CP"] = (CP, p); out["IndIta>Ghana"] = auc(CP, p)
    for tgt, name in ((CP, "Ghana"), (IN, "India"), (IT, "Italy")):
        tr = np.setdiff1d(np.arange(len(df)), tgt)
        p = proba(make().fit(X[tr], y[tr]), X[tgt]); out["LOSO>" + name] = auc(tgt, p); keep["LOSO_" + name] = (tgt, p)
    out["headline"] = float(np.mean([out[k] for k in ("within_Ghana", "within_India", "within_Italy", "Ghana>India", "Ghana>Italy", "IndIta>Ghana")]))
    out["loso_mean"] = float(np.mean([out["LOSO>Ghana"], out["LOSO>India"], out["LOSO>Italy"]]))
    return out, keep

rows = []; KEEP = {}
for fs, X0 in FE.items():
    for norm in ("none", "site_z"):
        for demo in (False, True):
            X = site_z(X0, [CP, IN, IT]) if norm == "site_z" else X0
            if demo: X = np.c_[X, DEMO]
            for mn in (["logistic", "rf"] if os.environ.get("FAST") else list(MODELS)):
                t0 = time.time(); o, k = six(MODELS[mn], X)
                key = (fs, norm, demo, mn); KEEP[key] = k
                rows.append({"features": fs, "norm": norm, "age_sex": demo, "model": mn, **o})
                print(f"{fs:11s} {norm:7s} demo={int(demo)} {mn:12s} head={o['headline']:.3f} loso={o['loso_mean']:.3f}  {time.time()-t0:.0f}s", flush=True)
res = pd.DataFrame(rows).sort_values("headline", ascending=False)
res.round(3).to_csv("exp2_results.csv", index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
print(res.round(3).to_string(index=False))
import pickle; pickle.dump({"KEEP": KEEP, "y": y, "site": site, "hb": hb, "cut": cut}, open("exp2_keep.pkl", "wb"))
