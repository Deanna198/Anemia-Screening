"""Experiment H: do the anaemia models still work when the eyelid region is found automatically?

Uses the masks predicted by the segmentation model for photos it never saw in training
(results/study/pred_masks/ from Kaggle notebook 03, 5-fold CV over all Eyes-Defy photos).
Colour features are recomputed from those masks and the fair Eyes-Defy tests are re-run; the result is
compared with the same tests on the hand-drawn masks.

Run:  python src/study/exp_segmentation.py
Out:  results/study/H_segmentation.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, cv2
import common as C
from features_v2 import all_feats, clean_mask

PRED = os.path.join(C.STUDY, "pred_masks")


def main():
    if not os.path.isdir(PRED):
        print("no results/study/pred_masks/ yet - run notebook 03 on Kaggle and copy its pred_masks folder here"); return
    df = C.load("none"); ed = df[df.site != "Ghana"].reset_index(drop=True)
    rows = []
    for r in ed.itertuples():
        im = cv2.imread(os.path.join(C.DATA, r.eye_path))[:, :, ::-1].astype(np.float32) / 255
        pm = cv2.imread(os.path.join(PRED, r.subject_id + ".png"), 0)
        pm = cv2.resize(pm, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        h, w = im.shape[:2]; s = 1000 / max(h, w)
        if s < 1:
            im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            pm = cv2.resize(pm, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = (pm > 127).astype(np.uint8)
        rows.append({"subject_id": r.subject_id, **(all_feats(im, clean_mask(m)) if m.sum() > 200 else {})})
    Fp = pd.DataFrame(rows).set_index("subject_id")
    out = []
    y = ed.anemic.values.astype(int); S = {s: np.where(ed.site == s)[0] for s in ("India", "Italy")}
    for source in ("hand-drawn masks", "automatic masks"):
        D = ed.copy()
        if source == "automatic masks":
            cols = [c for c in Fp.columns]
            D[cols] = Fp.reindex(D.subject_id)[cols].values
        for fs in ("ratio colour", "colour + texture"):
            X = C.X_of(D, fs)
            for mn in ("logistic regression", "extra trees", "LightGBM"):
                make = C.MODELS[mn]
                out.append({"masks": source, "features": fs, "model": mn,
                            "India": C.cv_test(make, X, y, S["India"])[0], "Italy": C.cv_test(make, X, y, S["Italy"])[0],
                            "India → Italy": C.transfer_test(make, X, y, S["India"], S["Italy"])[0],
                            "Italy → India": C.transfer_test(make, X, y, S["Italy"], S["India"])[0]})
    R = pd.DataFrame(out); R["average"] = R[["India", "Italy", "India → Italy", "Italy → India"]].mean(1)
    R.to_csv(os.path.join(C.STUDY, "H_segmentation.csv"), index=False)
    print(R.round(3).to_string(index=False))
    print(R.groupby("masks")[["India", "Italy", "India → Italy", "Italy → India", "average"]].mean().round(3))


if __name__ == "__main__":
    main()
