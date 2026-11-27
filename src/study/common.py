"""Shared pieces of the comparison study: data loading, feature sets, models, evaluation tests, metrics.

Every experiment script imports from here, so all methods are scored in exactly the same way.
"""
import os, warnings
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("ANEMIA_DATA", os.path.join(ROOT, "kaggle", "anemia-data"))   # standardised copy of both datasets
STUDY = os.environ.get("ANEMIA_STUDY", os.path.join(ROOT, "results", "study"))
SEED = 42
N_REP = int(os.environ.get("N_REP", "3"))      # repeats of each cross-validation
N_BOOT = int(os.environ.get("N_BOOT", "1000"))  # bootstrap resamples for confidence intervals


# ------------------------------------------------------------------ data
def load(method="none"):
    """One row per image. Ghana always uses method 'none' (no whole-eye photo exists)."""
    lab = pd.read_csv(os.path.join(DATA, "labels.csv"))
    grp = pd.read_csv(os.path.join(STUDY, "groups.csv"))
    F = pd.read_csv(os.path.join(STUDY, "features_v2.csv"))
    F = F[(F.method == "none") & F.subject_id.str.startswith("Image") | (F.method == method) & ~F.subject_id.str.startswith("Image")]
    df = lab.merge(grp, on="subject_id").merge(F, on="subject_id")
    df = df[df.hb.notna()].reset_index(drop=True)
    return df


def cols(df, family):
    c = {"raw": [x for x in df.columns if x.startswith("raw_")],
         "ratio": [x for x in df.columns if x.startswith("ratio_")],
         "rich": [x for x in df.columns if x.startswith("rich_")],
         "texture": [x for x in df.columns if x.startswith("tex_")]}
    return c[family]


FEATURE_SETS = {
    "raw colour": ["raw"],
    "ratio colour": ["ratio"],
    "ratio + percentiles": ["ratio", "rich"],
    "texture only": ["texture"],
    "colour + texture": ["ratio", "rich", "texture"],
    "everything": ["raw", "ratio", "rich", "texture"],
}


def X_of(df, fset):
    c = sum((cols(df, f) for f in FEATURE_SETS[fset]), [])
    return np.nan_to_num(df[c].values.astype(np.float64))


# ------------------------------------------------------------------ models
def sc(m): return Pipeline([("sc", StandardScaler()), ("m", m)])

MODELS = {
    "logistic regression": lambda: sc(LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000)),
    "SVM (RBF)": lambda: sc(SVC(C=1.0, gamma="scale", class_weight="balanced")),
    "k-nearest neighbours": lambda: sc(KNeighborsClassifier(15, weights="distance")),
    "MLP (small neural net)": lambda: sc(MLPClassifier((32,), alpha=1e-2, max_iter=2000, early_stopping=True, random_state=SEED)),
    "random forest": lambda: RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", n_jobs=2, random_state=SEED),
    "extra trees": lambda: ExtraTreesClassifier(300, min_samples_leaf=3, class_weight="balanced", n_jobs=2, random_state=SEED),
    "XGBoost": lambda: XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                     n_jobs=2, random_state=SEED, verbosity=0),
    "LightGBM": lambda: LGBMClassifier(n_estimators=200, num_leaves=8, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.8, class_weight="balanced", n_jobs=2, random_state=SEED, verbose=-1),
}


def score(model, X):
    if hasattr(model, "predict_proba"):
        try: return model.predict_proba(X)[:, 1]
        except AttributeError: pass
    return model.decision_function(X)


# ------------------------------------------------------------------ metrics
def auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def boot_ci(y, s, groups=None, n=N_BOOT, seed=SEED):
    """95% CI of AUC. Resamples groups (copies of the same eye) when given, else subjects."""
    rng = np.random.default_rng(seed)
    if groups is None: groups = np.arange(len(y))
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(ug))]
    vals = []
    for _ in range(n):
        pick = np.concatenate([members[k] for k in rng.integers(0, len(ug), len(ug))])
        a = auc(y[pick], s[pick])
        if not np.isnan(a): vals.append(a)
    return np.percentile(vals, [2.5, 97.5])


def paired_boot_diff(y, s1, s2, groups=None, n=N_BOOT, seed=SEED):
    """AUC(s1) - AUC(s2) on the same test images, with a 95% CI and a two-sided bootstrap p-value."""
    rng = np.random.default_rng(seed)
    if groups is None: groups = np.arange(len(y))
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(ug))]
    d = []
    for _ in range(n):
        pick = np.concatenate([members[k] for k in rng.integers(0, len(ug), len(ug))])
        if len(np.unique(y[pick])) < 2: continue
        d.append(auc(y[pick], s1[pick]) - auc(y[pick], s2[pick]))
    d = np.array(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return auc(y, s1) - auc(y, s2), np.percentile(d, [2.5, 97.5]), min(p, 1.0)


# ------------------------------------------------------------------ evaluation tests
def site_index(df):
    s = df.site.values
    return {k: np.where(s == k)[0] for k in ("Ghana", "India", "Italy")}


def cv_test(make, X, y, idx, groups=None, n_rep=N_REP, seed=SEED):
    """Cross-validation restricted to rows `idx`. Groups (if given) never straddle a split.
    Returns mean AUC over repeats, the per-repeat AUCs, and the out-of-fold scores of repeat 0."""
    aucs = []; first = None
    for rep in range(n_rep):
        sp = (StratifiedGroupKFold(5, shuffle=True, random_state=seed + rep) if groups is not None
              else StratifiedKFold(5, shuffle=True, random_state=seed + rep))
        s = np.zeros(len(idx))
        for tr, te in sp.split(idx, y[idx], groups[idx] if groups is not None else None):
            m = make().fit(X[idx[tr]], y[idx[tr]]); s[te] = score(m, X[idx[te]])
        aucs.append(auc(y[idx], s))
        if rep == 0: first = s
    return float(np.mean(aucs)), aucs, first


def transfer_test(make, X, y, tr, te):
    m = make().fit(X[tr], y[tr]); s = score(m, X[te])
    return auc(y[te], s), s


FAIR_TESTS = ["Ghana (copies grouped)", "India", "Italy", "Ghana → India", "Ghana → Italy", "India+Italy → Ghana"]


def fair_suite(make, X, df, keep_scores=False):
    """The six tests every method is ranked on, plus India↔Italy transfer.
    Ghana within-site CV keeps near-duplicate copies (strict grouping) together."""
    y = df.anemic.values.astype(int); S = site_index(df)
    G = df.group_strict.values
    gh = S["Ghana"][df.usable.values[S["Ghana"]] == 1]
    out, sc_ = {}, {}
    out["Ghana (copies grouped)"], _, sc_["Ghana (copies grouped)"] = cv_test(make, X, y, gh, G)
    out["India"], _, sc_["India"] = cv_test(make, X, y, S["India"])
    out["Italy"], _, sc_["Italy"] = cv_test(make, X, y, S["Italy"])
    ed = np.r_[S["India"], S["Italy"]]
    out["Ghana → India"], sc_["Ghana → India"] = transfer_test(make, X, y, gh, S["India"])
    out["Ghana → Italy"], sc_["Ghana → Italy"] = transfer_test(make, X, y, gh, S["Italy"])
    out["India+Italy → Ghana"], sc_["India+Italy → Ghana"] = transfer_test(make, X, y, ed, gh)
    out["India → Italy"], sc_["India → Italy"] = transfer_test(make, X, y, S["India"], S["Italy"])
    out["Italy → India"], sc_["Italy → India"] = transfer_test(make, X, y, S["Italy"], S["India"])
    out["fair average"] = float(np.mean([out[k] for k in FAIR_TESTS]))
    out["cross-population average"] = float(np.mean([out[k] for k in FAIR_TESTS[3:]]))
    idx = {"Ghana (copies grouped)": gh, "India": S["India"], "Italy": S["Italy"], "Ghana → India": S["India"],
           "Ghana → Italy": S["Italy"], "India+Italy → Ghana": gh, "India → Italy": S["Italy"], "Italy → India": S["India"]}
    return (out, sc_, idx) if keep_scores else out
