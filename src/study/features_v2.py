"""Feature extraction for the comparison study.

For every image and every colour-correction method this computes four feature families:
  raw     - mean/sd of R, G, B, L*, V (depends on lighting and camera)
  ratio   - brightness-free colour: chromaticities, log ratios (erythema index), Lab a*/b*, saturation
  rich    - percentiles of the ratio channels + the reddest/palest third of the tissue
  texture - vessel and surface texture: vessel density, GLCM contrast/homogeneity/energy

Colour-correction methods (the 'preprocessing' axis of the study):
  none        - pixels as captured
  grayworld   - divide by the average colour of the whole eye photo (assumes the scene averages to grey)
  whitepatch  - divide by the brightest 1% of the whole eye photo (assumes the brightest thing is white)
  sclera      - divide by the colour of the white of the eye just above the eyelid (a built-in white reference;
                this is what replaces the colour card)
The last three need the whole eye photo, which only Eyes-Defy-Anemia has; CP-AnemiC is supplied
already cropped, so it only gets 'none'.

Run:  python src/study/features_v2.py
Out:  results/study/features_v2.csv  (one row per image x method)
"""
import os
import numpy as np, pandas as pd, cv2
from skimage.feature import graycomatrix, graycoprops

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("ANEMIA_DATA", os.path.join(ROOT, "kaggle", "anemia-data"))
OUT = os.environ.get("ANEMIA_FEATS_OUT", os.path.join(ROOT, "results", "study", "features_v2.csv"))
METHODS = ["none", "grayworld", "whitepatch", "sclera"]


# ---------------------------------------------------------------- pixels
def clean_mask(mask):
    k = max(3, int(0.01 * max(mask.shape)) | 1)
    inner = cv2.erode(mask, np.ones((k, k), np.uint8))
    return inner if inner.sum() >= 200 else mask


def drop_glare(px):
    v = px.max(1); sat = (v - px.min(1)) / np.maximum(v, 1e-6)
    keep = ~((v > 0.92) & (sat < 0.25)) & (v > 0.08)
    return keep if keep.sum() >= 100 else np.ones(len(px), bool)


def cutout(path):
    """CP-AnemiC cut-out -> (rgb image float, mask)."""
    im = cv2.imread(os.path.join(DATA, path), cv2.IMREAD_UNCHANGED)
    if im.dtype == np.uint16: im = (im / 257).astype(np.uint8)
    h, w = im.shape[:2]; s = 1000 / max(h, w)
    if s < 1: im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    if im.shape[2] == 4 and (im[:, :, 3] > 200).mean() < 0.9: mask = (im[:, :, 3] > 200).astype(np.uint8)
    else: mask = (im[:, :, :3].min(2) < 245).astype(np.uint8)
    return im[:, :, 2::-1].astype(np.float32) / 255, clean_mask(mask)


def eye_and_mask(eye_path, mask_path):
    im = cv2.imread(os.path.join(DATA, eye_path)); m = cv2.imread(os.path.join(DATA, mask_path), 0)
    h, w = im.shape[:2]; s = 1000 / max(h, w)
    if s < 1:
        im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
    return im[:, :, ::-1].astype(np.float32) / 255, clean_mask((m > 127).astype(np.uint8))


def sclera_white(rgb, mask):
    """Colour of the sclera: bright, unsaturated pixels in a band just above the conjunctiva mask."""
    h = rgb.shape[0]
    up = np.zeros_like(mask); band = int(0.25 * h)
    ys, xs = np.nonzero(mask)
    x0, x1 = xs.min(), xs.max()
    for x in range(x0, x1 + 1, 2):
        col = ys[xs == x]
        if len(col): top = col.min(); up[max(0, top - band):max(0, top - 5), x:x + 2] = 1
    up[mask > 0] = 0
    px = rgb[up > 0]
    if len(px) < 200: return None
    v = px.max(1); sat = (v - px.min(1)) / np.maximum(v, 1e-6)
    cand = px[(sat < np.percentile(sat, 30)) & (v < 0.97)]
    if len(cand) < 50: return None
    bright = cand[cand.mean(1) >= np.percentile(cand.mean(1), 60)]
    return bright.mean(0)


def correct(rgb, mask, method):
    if method == "none": return rgb, True
    if method == "grayworld": ref = rgb.reshape(-1, 3).mean(0); ref = ref / ref.mean()
    elif method == "whitepatch":
        flat = rgb.reshape(-1, 3); b = flat.mean(1)
        ref = flat[b >= np.percentile(b, 99)].mean(0)
    elif method == "sclera":
        ref = sclera_white(rgb, mask)
        if ref is None: return rgb, False
    ref = np.maximum(ref, 1e-3)
    if method == "grayworld": out = rgb / ref            # average colour becomes neutral grey
    else: out = rgb / ref * 0.9                           # the reference white becomes 90% grey
    return np.clip(out, 0, 1), True


# ---------------------------------------------------------------- features
def colour_feats(px):
    f = {}
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_RGB2Lab).reshape(-1, 3)
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    for n, c in (("R", r), ("G", g), ("B", b), ("L", lab[:, 0]), ("V", hsv[:, 2])):
        f[f"raw_{n}_mean"] = c.mean(); f[f"raw_{n}_std"] = c.std()
    tot = r + g + b + 1e-6
    rr, gg, bb = r + 1e-3, g + 1e-3, b + 1e-3
    ch = {"rch": r / tot, "gch": g / tot, "logRG": np.log(rr / gg), "logRB": np.log(rr / bb),
          "logGB": np.log(gg / bb), "a": lab[:, 1], "b": lab[:, 2], "S": hsv[:, 1]}
    for n, c in ch.items():
        f[f"ratio_{n}_mean"] = c.mean()
    f["ratio_a_std"] = lab[:, 1].std(); f["ratio_S_std"] = hsv[:, 1].std()
    for n in ("rch", "logRG", "logRB", "a", "b"):
        c = ch[n]
        for q in (10, 25, 50, 75, 90): f[f"rich_{n}_p{q}"] = np.percentile(c, q)
        f[f"rich_{n}_iqr"] = f[f"rich_{n}_p75"] - f[f"rich_{n}_p25"]
    e = ch["logRG"]; o = np.argsort(e); n3 = max(30, len(o) // 3)
    f["rich_red_top"] = e[o[-n3:]].mean(); f["rich_red_low"] = e[o[:n3]].mean()
    f["rich_red_spread"] = f["rich_red_top"] - f["rich_red_low"]
    return f


def texture_feats(rgb, mask):
    """Vessel density and surface texture inside the mask (scale-normalised to 256 px wide)."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = rgb[y0:y1, x0:x1]; m = mask[y0:y1, x0:x1]
    s = 256 / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(8, int(crop.shape[1] * s)), max(8, int(crop.shape[0] * s))), interpolation=cv2.INTER_AREA)
    m = cv2.resize(m, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = cv2.erode(m, np.ones((5, 5), np.uint8)) if cv2.erode(m, np.ones((5, 5), np.uint8)).sum() > 100 else m
    green = (crop[:, :, 1] * 255).astype(np.uint8)
    # vessels are darker in green than their surroundings: black top-hat
    th = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    vals = th[m > 0].astype(np.float32)
    f = {"tex_vessel_mean": vals.mean() / 255, "tex_vessel_frac": float((vals > 12).mean()),
         "tex_vessel_p90": np.percentile(vals, 90) / 255}
    g = green.copy(); g[m == 0] = 0
    q = (g // 16).astype(np.uint8)
    P = graycomatrix(q, [1, 3], [0, np.pi / 2], levels=16, symmetric=True, normed=False)
    P[0, :, :, :] = 0; P[:, 0, :, :] = 0              # ignore background
    P = P.astype(np.float64); P /= np.maximum(P.sum(axis=(0, 1), keepdims=True), 1)
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        f[f"tex_glcm_{prop}"] = float(np.nan_to_num(graycoprops(P, prop)).mean())
    lap = cv2.Laplacian(green, cv2.CV_32F)
    f["tex_sharpness"] = float(lap[m > 0].var())
    return f


def all_feats(rgb, mask):
    px = rgb[mask > 0]
    px = px[drop_glare(px)]
    return {**colour_feats(px), **texture_feats(rgb, mask), "n_pixels": len(px)}


def main():
    lab = pd.read_csv(os.path.join(DATA, "labels.csv"))
    rows = []
    for i, r in lab.iterrows():
        if r.site == "Ghana":
            rgb, m = cutout(r.conj_path)
            rows.append({"subject_id": r.subject_id, "method": "none", "ref_ok": 1, **all_feats(rgb, m)})
        else:
            rgb0, m = eye_and_mask(r.eye_path, r.mask_path)
            for meth in METHODS:
                rgb, ok = correct(rgb0, m, meth)
                rows.append({"subject_id": r.subject_id, "method": meth, "ref_ok": int(ok), **all_feats(rgb, m)})
        if (i + 1) % 100 == 0: print(i + 1, "/", len(lab), flush=True)
    F = pd.DataFrame(rows)
    F.to_csv(OUT, index=False)
    print("saved", OUT, F.shape, "| sclera reference found for",
          int(F[(F.method == "sclera")].ref_ok.sum()), "of", int((F.method == "sclera").sum()), "Eyes-Defy photos")


if __name__ == "__main__":
    main()
