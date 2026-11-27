"""Colour-feature baseline for anaemia screening.

Run from the project root:  python src/baseline.py
Outputs (results/):
  baseline_classification.csv   AUC, sensitivity, specificity per experiment
  baseline_regression.csv       Hb estimation error per experiment
  baseline_roc.png              ROC curves
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SEED = 0


def classifiers():
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=2000)),
        "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                     class_weight="balanced", random_state=SEED, n_jobs=-1),
    }


def regressors():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "rf": RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                    random_state=SEED, n_jobs=-1),
    }


def cls_metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    return dict(n=len(y), anemic=int(y.sum()), auc=roc_auc_score(y, p),
                sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                accuracy=(tp + tn) / len(y))


def reg_metrics(y, p):
    d = p - y
    return dict(n=len(y), mae=np.abs(d).mean(), bias=d.mean(),
                loa_low=d.mean() - 1.96 * d.std(ddof=1),
                loa_high=d.mean() + 1.96 * d.std(ddof=1),
                r=np.corrcoef(y, p)[0, 1])


def main():
    df = pd.read_csv(os.path.join(ROOT, "data", "processed", "features.csv"))
    print("dropping", int((df.usable == 0).sum()), "images whose content is shared by several subjects")
    df = df[df.usable == 1].reset_index(drop=True)
    ratio = [c for c in df.columns if c.startswith("ratio_")]
    raw = [c for c in df.columns if c.startswith("raw_")]
    feature_sets = {"ratio": ratio, "raw+ratio": raw + ratio}
    cp = df[df.dataset == "cp_anemic"].reset_index(drop=True)
    eda = df[df.dataset == "eyes_defy"].reset_index(drop=True)

    cls_rows, reg_rows, curves = [], [], {}
    for fs_name, cols in feature_sets.items():
        # ---- within-dataset: 5-fold cross-validation ----
        for ds_name, d in (("CP-AnemiC", cp), ("Eyes-Defy", eda)):
            X, y, hb = d[cols].values, d.anemic.values, d.hb.values
            skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
            for m_name, m in classifiers().items():
                p = cross_val_predict(m, X, y, cv=skf, method="predict_proba")[:, 1]
                cls_rows.append(dict(experiment=f"within {ds_name} (5-fold CV)",
                                     features=fs_name, model=m_name, **cls_metrics(y, p)))
                curves[(f"within {ds_name}", fs_name, m_name)] = (y, p)
                if ds_name == "Eyes-Defy":
                    # India is mostly anaemic and Italy mostly not, so the pooled
                    # score partly reflects "which site"; report each site too.
                    for site in ("India", "Italy"):
                        k = (d.site == site).values
                        cls_rows.append(dict(experiment=f"within {ds_name} (5-fold CV) [{site} only]",
                                             features=fs_name, model=m_name, **cls_metrics(y[k], p[k])))
            kf = KFold(5, shuffle=True, random_state=SEED)
            for m_name, m in regressors().items():
                p = cross_val_predict(m, X, hb, cv=kf)
                reg_rows.append(dict(experiment=f"within {ds_name} (5-fold CV)",
                                     features=fs_name, model=m_name, **reg_metrics(hb, p)))
        # ---- cross-dataset: train on one, test on the other ----
        for tr_name, tr, te_name, te in (("CP-AnemiC", cp, "Eyes-Defy", eda),
                                         ("Eyes-Defy", eda, "CP-AnemiC", cp)):
            for m_name, m in classifiers().items():
                m.fit(tr[cols].values, tr.anemic.values)
                p = m.predict_proba(te[cols].values)[:, 1]
                exp = f"train {tr_name} -> test {te_name}"
                cls_rows.append(dict(experiment=exp, features=fs_name, model=m_name,
                                     **cls_metrics(te.anemic.values, p)))
                curves[(exp, fs_name, m_name)] = (te.anemic.values, p)
                if te_name == "Eyes-Defy":
                    for site in ("India", "Italy"):
                        k = (te.site == site).values
                        cls_rows.append(dict(experiment=f"{exp} [{site} only]", features=fs_name,
                                             model=m_name, **cls_metrics(te.anemic.values[k], p[k])))
            for m_name, m in regressors().items():
                m.fit(tr[cols].values, tr.hb.values)
                p = m.predict(te[cols].values)
                reg_rows.append(dict(experiment=f"train {tr_name} -> test {te_name}",
                                     features=fs_name, model=m_name, **reg_metrics(te.hb.values, p)))

    os.makedirs(RES, exist_ok=True)
    cls = pd.DataFrame(cls_rows).round(3)
    reg = pd.DataFrame(reg_rows).round(3)
    cls.to_csv(os.path.join(RES, "baseline_classification.csv"), index=False)
    reg.to_csv(os.path.join(RES, "baseline_regression.csv"), index=False)
    pd.set_option("display.width", 200)
    print(cls.to_string(index=False)); print(); print(reg.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for exp in ("within CP-AnemiC", "within Eyes-Defy",
                "train CP-AnemiC -> test Eyes-Defy", "train Eyes-Defy -> test CP-AnemiC"):
        y, p = curves[(exp, "ratio", "logreg")]
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2, label=f"{exp}  (AUC {roc_auc_score(y, p):.2f})")
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--")
    ax.set_xlabel("False positive rate (1 - specificity)"); ax.set_ylabel("Sensitivity")
    ax.set_title("Colour baseline: logistic regression, ratio features")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "baseline_roc.png"), dpi=200)


if __name__ == "__main__":
    main()
