"""Experiment G: does the quality check work?

1. False rejections: run the check on every real dataset image (these were accepted by the datasets' authors,
   so ideally almost none are rejected).
2. Detection: damage each Eyes-Defy photo in one way at a time (blur, motion blur, too dark, overexposed,
   glare spot, eyelid barely visible) and count how often the check catches it, and with the right reason.

Run:  python src/study/exp_quality.py
Out:  results/study/G_quality.csv, G_quality_measures.csv, G_quality_examples.jpg
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, cv2
import common as C
from screening import quality as Q
from features_v2 import cutout, eye_and_mask

EXPECT = {"blur": "sharp", "motion blur": "sharp", "too dark": "not too dark", "overexposed": "not overexposed",
          "glare spot": "little glare", "eyelid barely visible": "conjunctiva visible"}


def damage(rgb, mask, kind, rng):
    im = rgb.copy(); m = mask.copy()
    if kind == "blur": im = cv2.GaussianBlur(im, (0, 0), 6)
    elif kind == "motion blur":
        k = np.zeros((31, 31), np.float32); k[15, :] = 1 / 31
        M = cv2.getRotationMatrix2D((15, 15), rng.uniform(0, 180), 1); k = cv2.warpAffine(k, M, (31, 31)); k /= k.sum()
        im = cv2.filter2D(im, -1, k)
    elif kind == "too dark": im = (im.astype(np.float32) * 0.18).astype(np.uint8)
    elif kind == "overexposed": im = np.clip(im.astype(np.float32) * 2.2 + 30, 0, 255).astype(np.uint8)
    elif kind == "glare spot":
        ys, xs = np.nonzero(m); c = len(ys) // 2; idx = np.argsort(xs)
        cy, cx = ys[idx[c]], xs[idx[c]]; r = int(np.sqrt(m.sum() / np.pi) * 0.9)
        cv2.ellipse(im, (int(cx), int(cy)), (r * 2, r), 0, 0, 360, (250, 250, 250), -1)
    elif kind == "eyelid barely visible":
        ys, xs = np.nonzero(m); x0 = np.percentile(xs, 47); x1 = np.percentile(xs, 53)
        keep = np.zeros_like(m); keep[:, int(x0):int(x1) + 1] = 1; m = m * keep
        m = cv2.erode(m, np.ones((9, 9), np.uint8))
    return im, m


def main():
    lab = C.load("none"); rng = np.random.default_rng(0)
    rows, meas = [], []
    for r in lab.itertuples():
        if r.site == "Ghana":
            f, m = cutout(r.conj_path); rgb = (f * 255).astype(np.uint8)
        else:
            f, m = eye_and_mask(r.eye_path, r.mask_path); rgb = (f * 255).astype(np.uint8)
        ok, rep = Q.check(rgb, m)
        meas.append({"subject_id": r.subject_id, "site": r.site, "usable": r.usable, "passed": ok, **rep["measures"],
                     "failed": "; ".join(c["check"] for c in rep["checks"] if not c["passed"])})
        if r.site != "Ghana":
            for kind in EXPECT:
                im2, m2 = damage(rgb, m, kind, rng); ok2, rep2 = Q.check(im2, m2)
                failed = [c["check"] for c in rep2["checks"] if not c["passed"]]
                rows.append({"subject_id": r.subject_id, "damage": kind, "caught": not ok2, "right reason": EXPECT[kind] in failed})
    M = pd.DataFrame(meas); M.to_csv(os.path.join(C.STUDY, "G_quality_measures.csv"), index=False)
    D = pd.DataFrame(rows)
    fr = M.groupby("site").passed.agg(lambda s: 1 - s.mean()).rename("share of real images rejected")
    det = D.groupby("damage")[["caught", "right reason"]].mean()
    out = pd.concat([fr.rename(lambda s: f"real images, {s}").to_frame("value"),
                     det["caught"].rename(lambda s: f"caught: {s}").to_frame("value"),
                     det["right reason"].rename(lambda s: f"caught for the right reason: {s}").to_frame("value")])
    out.to_csv(os.path.join(C.STUDY, "G_quality.csv"))
    print(out.round(3).to_string())
    print(M.groupby("site")[["pixels", "glare", "brightness", "clipped", "sharpness"]].quantile([0.01, 0.05, 0.5]).round(3))
    print(M[~M.passed][["subject_id", "failed"]].head(20).to_string())


if __name__ == "__main__":
    main()
