import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
DATA="/home/claude/work/input/anemia-data"
df=pd.read_csv(DATA+"/features.csv"); df=df[(df.usable==1)&df.hb.notna()].reset_index(drop=True)
X=df[[c for c in df.columns if c.startswith("ratio_")]].values; y=df.anemic.values; site=df.site.values
S={s:np.where(site==s)[0] for s in ("Ghana","India","Italy")}; ED=np.r_[S["India"],S["Italy"]]
LR=lambda: Pipeline([("sc",StandardScaler()),("m",LogisticRegression(C=0.3,class_weight="balanced",max_iter=3000))])
RF=lambda: RandomForestClassifier(300,min_samples_leaf=3,class_weight="balanced",n_jobs=-1,random_state=0)
def thresholds(p,yt,target):
    """Smallest 'anaemic' zone and 'not anaemic' zone that were at least `target` correct on training OOF."""
    hi=1.01; lo=-0.01
    for t in np.unique(np.round(p,3))[::-1]:
        m=p>=t
        if m.sum()>=10 and yt[m].mean()>=target: hi=t
    for t in np.unique(np.round(p,3)):
        m=p<=t
        if m.sum()>=10 and (1-yt[m]).mean()>=target: lo=t
    return lo,hi
def run(make,tr,te,target):
    oof=cross_val_predict(make(),X[tr],y[tr],cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1]
    lo,hi=thresholds(oof,y[tr],target)
    p=make().fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    return np.where(p>=hi,1,np.where(p<=lo,0,-1))
def report(name,idx,call):
    t=y[idx]; a=call==1; n=call==0; d=call>=0
    return dict(test=name,n=len(idx),decided=round(d.mean(),2),said_anaemic=int(a.sum()),anaemic_right=round(t[a].mean(),2) if a.any() else None,
                said_not=int(n.sum()),not_right=round((1-t[n]).mean(),2) if n.any() else None,
                acc_when_decided=round((call[d]==t[d]).mean(),2) if d.any() else None,
                anaemic_missed_as_not=int((n&(t==1)).sum()), total_anaemic=int(t.sum()))
for mk,mn in ((LR,"logistic"),(RF,"random forest")):
  for target in (0.80,0.85):
    rows=[]
    for nm,idx in (("within Ghana",S["Ghana"]),("within India+Italy",ED)):
        call=np.zeros(len(idx),int)
        st=y[idx] if nm=="within Ghana" else np.array([f"{a}{b}" for a,b in zip(site[idx],y[idx])])
        for a,b in StratifiedKFold(5,shuffle=True,random_state=1).split(idx,st): call[b]=run(mk,idx[a],idx[b],target)
        if nm=="within Ghana": rows.append(report(nm,idx,call))
        else:
            for s in ("India","Italy"):
                m=site[idx]==s; rows.append(report("within "+s,idx[m],call[m]))
    for s in ("Ghana","India","Italy"):
        tr=np.setdiff1d(np.arange(len(y)),S[s]); rows.append(report("other two sites > "+s,S[s],run(mk,tr,S[s],target)))
    print(f"\n== {mn}, zones must be >= {int(target*100)}% correct on training data ==")
    print(pd.DataFrame(rows).to_string(index=False))
