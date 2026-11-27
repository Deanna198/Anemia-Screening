"""Colour features from the palpebral conjunctiva cut-outs.

Run from the project root:  python src/features.py
Output: data/processed/features.csv

Both datasets provide the conjunctiva as an RGBA cut-out (alpha = mask), so no
segmentation model is needed for this baseline.
"""
import os
import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = os.path.join(ROOT, "data", "processed", "labels.csv")
OUT = os.path.join(ROOT, "data", "processed", "features.csv")
MAX_SIDE = 1000  # downscale very large cut-outs; colour statistics are unaffected


def conjunctiva_pixels(path, eye_path=None):
    """Return an (N, 3) float array of RGB pixels in [0, 1] inside the mask."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3:
        raise ValueError(f"cannot read cut-out: {path}")
    if img.dtype == np.uint16:
        # One Eyes-Defy cut-out (India/2) is a 16-bit file with a different tone
        # curve. Keep only its mask and take the colours from the original photo.
        alpha = (img[:, :, 3] / 257.0).astype(np.uint8)
        photo = cv2.imread(eye_path, cv2.IMREAD_COLOR)
        if photo is None or abs(photo.shape[0] / photo.shape[1] - img.shape[0] / img.shape[1]) > 0.02:
            raise ValueError(f"16-bit cut-out without a matching photo: {path}")
        alpha = cv2.resize(alpha, (photo.shape[1], photo.shape[0]), interpolation=cv2.INTER_NEAREST)
        img = np.dstack([photo, alpha])
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    # The cut-outs come in three formats: transparent background (alpha mask),
    # or a white background with a fully opaque / missing alpha channel.
    has_alpha = img.shape[2] == 4 and (img[:, :, 3] > 200).mean() < 0.9
    if has_alpha:
        mask = (img[:, :, 3] > 200).astype(np.uint8)
    else:
        mask = (img[:, :, :3].min(axis=2) < 245).astype(np.uint8)
    # shrink the mask slightly so edge pixels (skin, lashes, sclera) are excluded
    k = max(3, int(0.01 * max(img.shape[:2])) | 1)
    inner = cv2.erode(mask, np.ones((k, k), np.uint8))
    if inner.sum() >= 200:
        mask = inner
    rgb = img[:, :, 2::-1].astype(np.float32) / 255.0
    px = rgb[mask > 0]
    # remove specular highlights (wet surface glare) and near-black pixels
    v = px.max(axis=1)
    sat = (v - px.min(axis=1)) / np.maximum(v, 1e-6)
    keep = ~((v > 0.92) & (sat < 0.25)) & (v > 0.08)
    if keep.sum() >= 100:
        px = px[keep]
    return px


def colour_features(px):
    f = {}
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    lab = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).reshape(-1, 3)
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    # --- raw colour (depends on lighting and camera) ---
    for name, ch in (("R", r), ("G", g), ("B", b), ("L", lab[:, 0]), ("V", hsv[:, 2])):
        f[f"raw_{name}_mean"] = ch.mean()
        f[f"raw_{name}_std"] = ch.std()
    # --- ratio colour (brightness cancels out) ---
    tot = r + g + b + 1e-6
    f["ratio_r_chroma"] = (r / tot).mean()
    f["ratio_g_chroma"] = (g / tot).mean()
    f["ratio_b_chroma"] = (b / tot).mean()
    f["ratio_log_R_G"] = np.log((r + 1e-3) / (g + 1e-3)).mean()   # erythema index
    f["ratio_log_R_B"] = np.log((r + 1e-3) / (b + 1e-3)).mean()
    f["ratio_log_G_B"] = np.log((g + 1e-3) / (b + 1e-3)).mean()
    f["ratio_a_mean"] = lab[:, 1].mean()
    f["ratio_a_std"] = lab[:, 1].std()
    f["ratio_a_p10"] = np.percentile(lab[:, 1], 10)
    f["ratio_a_p90"] = np.percentile(lab[:, 1], 90)
    f["ratio_b_mean"] = lab[:, 2].mean()
    f["ratio_S_mean"] = hsv[:, 1].mean()
    f["ratio_S_std"] = hsv[:, 1].std()
    f["n_pixels"] = len(px)
    return f


def main():
    labels = pd.read_csv(LABELS)
    rows = []
    for i, r in labels.iterrows():
        eye = os.path.join(ROOT, r.eye_path) if isinstance(r.eye_path, str) and r.eye_path else None
        px = conjunctiva_pixels(os.path.join(ROOT, r.conj_path), eye)
        rows.append({"subject_id": r.subject_id, **colour_features(px)})
        if (i + 1) % 100 == 0:
            print(i + 1, "/", len(labels), flush=True)
    feats = pd.DataFrame(rows)
    labels.merge(feats, on="subject_id").to_csv(OUT, index=False)
    print("saved", OUT, feats.shape)


if __name__ == "__main__":
    main()
