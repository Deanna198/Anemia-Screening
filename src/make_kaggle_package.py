"""Build a small, standardised copy of the data for Kaggle.

Run from the project root:  python src/make_kaggle_package.py
Output: kaggle/anemia-data/  and  kaggle/anemia-data.zip

Contents
  labels.csv, features.csv     as in data/processed (paths replaced by pkg paths)
  conj/<subject_id>.png        conjunctiva cut-out, RGBA, alpha = mask, cropped to the mask
  eye/<subject_id>.jpg         whole-eye photo, 800 px wide      (Eyes-Defy only)
  mask/<subject_id>.png        palpebral conjunctiva mask, 0/255 (Eyes-Defy only)
The script can be re-run; finished subjects are skipped.
"""
import os, sys, time, zipfile
import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "kaggle", "anemia-data")
BUDGET_S = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9   # stop early, resume on next run


def load_cutout(conj_path, eye_path):
    """Return (bgr uint8, mask uint8 0/1) at cut-out resolution, plus the photo if any."""
    img = cv2.imread(conj_path, cv2.IMREAD_UNCHANGED)
    photo = cv2.imread(eye_path, cv2.IMREAD_COLOR) if eye_path else None
    if img.dtype == np.uint16:           # India_2: keep the mask, take colours from the photo
        alpha = (img[:, :, 3] / 257.0).astype(np.uint8)
        bgr = cv2.resize(photo, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        return bgr, (alpha > 200).astype(np.uint8), photo
    has_alpha = img.shape[2] == 4 and (img[:, :, 3] > 200).mean() < 0.9
    if has_alpha:
        mask = (img[:, :, 3] > 200).astype(np.uint8)
    else:                                 # white-background cut-outs
        mask = (img[:, :, :3].min(axis=2) < 245).astype(np.uint8)
    return img[:, :, :3], mask, photo


def main():
    t0 = time.time()
    for sub in ("conj", "eye", "mask"):
        os.makedirs(os.path.join(PKG, sub), exist_ok=True)
    feats = pd.read_csv(os.path.join(ROOT, "data", "processed", "features.csv"))
    done = 0
    for _, r in feats.iterrows():
        out_c = os.path.join(PKG, "conj", r.subject_id + ".png")
        if os.path.exists(out_c):
            done += 1
            continue
        if time.time() - t0 > BUDGET_S:
            print("paused at", done, "/", len(feats)); return False
        eye = os.path.join(ROOT, r.eye_path) if isinstance(r.eye_path, str) else None
        bgr, mask, photo = load_cutout(os.path.join(ROOT, r.conj_path), eye)
        if photo is not None:
            h, w = mask.shape
            cv2.imwrite(os.path.join(PKG, "eye", r.subject_id + ".jpg"),
                        cv2.resize(photo, (w, h), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(os.path.join(PKG, "mask", r.subject_id + ".png"), mask * 255)
        ys, xs = np.where(mask > 0)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        rgba = np.dstack([bgr[y0:y1, x0:x1], mask[y0:y1, x0:x1] * 255])
        cv2.imwrite(out_c + ".tmp.png", rgba)
        os.replace(out_c + ".tmp.png", out_c)
        done += 1
    out = feats.copy()
    out["conj_path"] = "conj/" + out.subject_id + ".png"
    is_ed = out.dataset == "eyes_defy"
    out["eye_path"] = np.where(is_ed, "eye/" + out.subject_id + ".jpg", "")
    out["mask_path"] = np.where(is_ed, "mask/" + out.subject_id + ".png", "")
    out.to_csv(os.path.join(PKG, "features.csv"), index=False)
    out[[c for c in out.columns if not c.startswith(("raw_", "ratio_", "n_pixels"))]].to_csv(
        os.path.join(PKG, "labels.csv"), index=False)
    zpath = os.path.join(ROOT, "kaggle", "anemia-data.zip")
    with zipfile.ZipFile(zpath + ".part", "w", zipfile.ZIP_STORED) as z:
        for d, _, fs in os.walk(PKG):
            for f in sorted(fs):
                p = os.path.join(d, f)
                z.write(p, os.path.relpath(p, PKG))
    os.replace(zpath + ".part", zpath)
    print("done", done, "subjects ->", zpath, round(os.path.getsize(zpath) / 1e6, 1), "MB")
    return True


if __name__ == "__main__":
    main()
