import os, warnings, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
DATA="/home/claude/work/input/anemia-data"
df=pd.read_csv(DATA+"/features.csv"); df=df[(df.usable==1)&df.hb.notna()].reset_index(drop=True)
hb=df.hb.values; site=df.site.values; cut=df.who_cutoff.values
X=df[[c for c in df.columns if c.startswith("ratio_")]].values
S={s:np.where(site==s)[0] for s in ("Ghana","India","Italy")}
LR=lambda: Pipeline([("sc",StandardScaler()),("m",LogisticRegression(C=0.3,class_weight="balanced",max_iter=3000))])
RF=lambda: RandomForestClassifier(300,min_samples_leaf=3,class_weight="balanced",n_jobs=-1,random_state=0)
def cv_auc(make,idx,yy,rep=3):
    if yy[idx].sum()<8 or (1-yy[idx]).sum()<8: return np.nan
    out=[]
    for r in range(rep):
        p=np.zeros(len(idx))
        from sklearn.model_selection import StratifiedKFold
        for a,b in StratifiedKFold(5,shuffle=True,random_state=r).split(idx,yy[idx]):
            p[b]=make().fit(X[idx[a]],yy[idx[a]]).predict_proba(X[idx[b]])[:,1]
        out.append(roc_auc_score(yy[idx],p))
    return np.mean(out)
rows=[]
defs={"WHO cutoff (any anaemia)":(hb<cut), "Hb < 10":(hb<10), "Hb < 9":(hb<9), "Hb < 8":(hb<8), "cutoff - 1 (moderate+)":(hb<cut-1), "cutoff - 2":(hb<cut-2)}
for name,lab in defs.items():
    yy=lab.astype(int); r={"target":name}
    for s,idx in S.items():
        r[f"n_pos_{s}"]=int(yy[idx].sum())
        r[f"LR_{s}"]=cv_auc(LR,idx,yy); r[f"RF_{s}"]=cv_auc(RF,idx,yy)
    # cross: train Ghana -> India/Italy; train India+Italy -> Ghana
    ed=np.r_[S["India"],S["Italy"]]
    if yy[S["Ghana"]].sum()>=8:
        m=LR().fit(X[S["Ghana"]],yy[S["Ghana"]])
        for s in ("India","Italy"):
            i=S[s]; r[f"Gh>{s}"]=roc_auc_score(yy[i],m.predict_proba(X[i])[:,1]) if 0<yy[i].sum()<len(i) else np.nan
    if yy[ed].sum()>=8:
        m=LR().fit(X[ed],yy[ed]); i=S["Ghana"]; r["IndIta>Gh"]=roc_auc_score(yy[i],m.predict_proba(X[i])[:,1])
    rows.append(r)
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
print(pd.DataFrame(rows).round(2).to_string(index=False))
