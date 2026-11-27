"""Build one clean label table for both datasets.

Run from the project root:  python src/prepare_labels.py
Output: data/processed/labels.csv  (one row per subject)
"""
import hashlib
import os
import re
import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed", "labels.csv")


def who_cutoff(age_years, sex):
    """WHO 2024 haemoglobin cutoff (g/dL) below which a person is anaemic.
    Non-pregnant, sea level, non-smoker."""
    if age_years < 2:
        return 10.5          # 6-23 months
    if age_years < 5:
        return 11.0          # 24-59 months
    if age_years < 12:
        return 11.5          # 5-11 years
    if age_years < 15:
        return 12.0          # 12-14 years
    return 13.0 if sex == "M" else 12.0   # 15+ years


def cp_anemic():
    base = os.path.join(RAW, "cp_anemic")
    df = pd.read_excel(os.path.join(base, "Anemia_Data_Collection_Sheet.xlsx"))
    rows = []
    for _, r in df.iterrows():
        folder = "Anemic" if r["REMARK"].strip().lower() == "anemic" else "Non-anemic"
        rel = os.path.join("data", "raw", "cp_anemic", folder, r["IMAGE_ID"] + ".png")
        rows.append(dict(
            dataset="cp_anemic", site="Ghana", subject_id=r["IMAGE_ID"],
            conj_path=rel, eye_path="",
            hb=float(r["HB_LEVEL"]), age_years=r["Age(Months)"] / 12.0,
            sex=r["GENDER"].strip()[0].upper(),
            dataset_label=int(folder == "Anemic"),
        ))
    return rows


def pick_palpebral(files):
    """Palpebral-only cut-out. File names contain typos (papebral, palplebral)
    and a few '(1)' duplicates, so match loosely."""
    c = [f for f in files if f.lower().endswith(".png")
         and "forniceal" not in f.lower() and re.search(r"pa\w*bral", f.lower())]
    c.sort(key=lambda f: ("(" in f, f))
    return c[0] if c else None


def eyes_defy(site):
    base = os.path.join(RAW, "eyes_defy_anemia", site)
    df = pd.read_excel(os.path.join(base, site + ".xlsx"))
    rows = []
    for _, r in df.iterrows():
        n = int(r["Number"])
        # Italian sheet uses decimal commas for some rows; one row has no Hb.
        hb = pd.to_numeric(str(r["Hgb"]).replace(",", "."), errors="coerce")
        if pd.isna(hb):
            continue
        folder = os.path.join(base, str(n))
        files = [f for f in os.listdir(folder) if not f.startswith(".")]
        pal = pick_palpebral(files)
        jpg = [f for f in files if f.lower().endswith((".jpg", ".jpeg"))]
        if pal is None:
            continue
        rel = os.path.join("data", "raw", "eyes_defy_anemia", site, str(n))
        rows.append(dict(
            dataset="eyes_defy", site=site, subject_id=f"{site}_{n}",
            conj_path=os.path.join(rel, pal),
            eye_path=os.path.join(rel, jpg[0]) if jpg else "",
            hb=float(hb), age_years=float(r["Age"]),
            sex=str(r["Gender"]).strip()[0].upper(),
            dataset_label=-1,
        ))
    return rows


def thumbnail(img):
    """16x8 thumbnail of the conjunctiva region only (background removed, cropped
    to the mask), so that re-saved copies of one picture look almost identical."""
    if img.dtype != np.uint8:
        img = (img / 257).astype(np.uint8)
    if img.shape[2] == 4 and (img[:, :, 3] > 200).mean() < 0.9:
        mask = img[:, :, 3] > 200
    else:
        mask = img[:, :, :3].min(axis=2) < 245
    ys, xs = np.where(mask)
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1, :3].copy()
    crop[~mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]] = 0
    small = cv2.resize(crop, (16, 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return small, crop.shape[:2]


def flag_duplicates(df):
    """Find images whose content is shared by more than one subject.

    CP-AnemiC contains many images that are pixel-identical (or re-saved copies)
    but listed as different children with different Hb values. Their labels
    cannot be trusted, so they are marked usable = 0.
    """
    hashes, thumbs, sizes = [], [], []
    for p in df.conj_path:
        img = cv2.imread(os.path.join(ROOT, p), cv2.IMREAD_UNCHANGED)
        hashes.append(hashlib.md5(str(img.shape).encode() + img.tobytes()).hexdigest())
        t, size = thumbnail(img)
        thumbs.append(t)
        sizes.append(size)
    df["content_hash"] = hashes
    df["n_copies"] = df.groupby("content_hash").content_hash.transform("size")
    # near-duplicates: same picture re-saved, so the bytes differ slightly
    # -> same region size (within 2 px) and an almost identical thumbnail
    T, S = np.stack(thumbs), np.array(sizes)
    near = np.zeros(len(df), bool)
    for i in range(len(df)):
        d = np.abs(T - T[i]).mean(axis=1)
        same_size = (np.abs(S - S[i]).max(axis=1) <= 2)
        d[i] = 1e9
        near[i] = (d[same_size] < 5.0).any()
    df["near_duplicate"] = near.astype(int)
    df["usable"] = ((df.n_copies == 1) & (~near)).astype(int)
    return df


def main():
    rows = cp_anemic() + eyes_defy("India") + eyes_defy("Italy")
    df = pd.DataFrame(rows)
    df["who_cutoff"] = [who_cutoff(a, s) for a, s in zip(df.age_years, df.sex)]
    df["anemic"] = (df.hb < df.who_cutoff).astype(int)
    df["conj_path"] = df.conj_path.str.replace("\\", "/")
    df["eye_path"] = df.eye_path.str.replace("\\", "/")
    missing = [p for p in df.conj_path if not os.path.exists(os.path.join(ROOT, p))]
    assert not missing, missing[:5]
    df = flag_duplicates(df)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.groupby(["dataset", "site"]).agg(
        n=("hb", "size"), usable=("usable", "sum"), anemic=("anemic", "sum"), hb_mean=("hb", "mean"),
        hb_min=("hb", "min"), hb_max=("hb", "max"), age_mean=("age_years", "mean")).round(2))
    cp = df[df.dataset == "cp_anemic"]
    print("CP-AnemiC: dataset label vs WHO-2024 label disagree on",
          int((cp.dataset_label != cp.anemic).sum()), "of", len(cp))


if __name__ == "__main__":
    main()
