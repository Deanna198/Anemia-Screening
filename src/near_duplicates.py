"""Find near-duplicate images in CP-AnemiC (same eye mirrored, re-cropped or re-saved)
and give every image a group id, so copies never end up on both sides of a train/test split.

Two steps:
1. Shortlist: for each image, its 12 closest neighbours by a cheap fingerprint
   (perceptual hash of the image and its mirror image + colour histogram).
2. Verify: SIFT keypoint matching with a geometric check (RANSAC). Two photos of the
   same eye share many keypoints that line up under one transform; two different eyes don't.

Run:  python src/near_duplicates.py
Out:  results/study/groups.csv (strict and lenient group ids) and groups_pairs.csv (every checked pair)
"""
import os, sys, itertools
import numpy as np, pandas as pd, cv2
from PIL import Image
import imagehash

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("ANEMIA_DATA", os.path.join(ROOT, "kaggle", "anemia-data"))
OUT = os.environ.get("ANEMIA_GROUPS_OUT", os.path.join(ROOT, "results", "study", "groups.csv"))
STRICT, LENIENT = 25, 15   # matching keypoints needed to call two images the same eye (checked by eye: see results/study)
K = 12


def load(path):
    im = cv2.imread(os.path.join(DATA, path), cv2.IMREAD_UNCHANGED)
    if im.dtype == np.uint16: im = (im / 257).astype(np.uint8)
    if im.ndim == 3 and im.shape[2] == 4:
        a = im[:, :, 3:4].astype(np.float32) / 255
        im = (im[:, :, :3] * a).astype(np.uint8)          # transparent background -> black
    h, w = im.shape[:2]; s = 400 / max(h, w)
    return cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def fingerprint(im):
    pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    hs = [imagehash.phash(pil), imagehash.phash(pil.transpose(Image.FLIP_LEFT_RIGHT)),
          imagehash.phash(pil.transpose(Image.FLIP_TOP_BOTTOM))]
    m = (im.max(2) > 10).astype(np.uint8)
    hist = cv2.calcHist([cv2.cvtColor(im, cv2.COLOR_BGR2HSV)], [0, 1], m, [24, 16], [0, 180, 0, 256])
    return hs, cv2.normalize(hist, None).flatten()


def sift_inliers(a, b, sift, bf):
    best = 0
    ka, da = a
    for kb, db in b:  # b holds the original and the mirrored version
        if da is None or db is None or len(ka) < 8 or len(kb) < 8: continue
        m = bf.knnMatch(da, db, k=2)
        good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
        if len(good) < 8: continue
        pa = np.float32([ka[g.queryIdx].pt for g in good]); pb = np.float32([kb[g.trainIdx].pt for g in good])
        H, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if inl is not None: best = max(best, int(inl.sum()))
    return best


def main():
    lab = pd.read_csv(os.path.join(DATA, "labels.csv"))
    gh = lab[lab.site == "Ghana"].reset_index(drop=True)
    sift = cv2.SIFT_create(nfeatures=800); bf = cv2.BFMatcher(cv2.NORM_L2)
    ims = [load(p) for p in gh.conj_path]
    fps = [fingerprint(im) for im in ims]
    kps = []
    for im in ims:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        k1, d1 = sift.detectAndCompute(g, None)
        k2, d2 = sift.detectAndCompute(cv2.flip(g, 1), None)
        kps.append(((k1, d1), (k2, d2)))
    n = len(gh)
    # 1. shortlist
    H = np.array([[int(fps[i][0][0] - fps[j][0][k]) for k in range(3)] for i in range(n) for j in range(n)]).min(1).reshape(n, n)
    Hist = np.array([[cv2.compareHist(fps[i][1], fps[j][1], cv2.HISTCMP_BHATTACHARYYA) for j in range(n)] for i in range(n)])
    score = H / 32.0 + Hist
    np.fill_diagonal(score, np.inf)
    cand = set()
    for i in range(n):
        for j in np.argsort(score[i])[:K]: cand.add((min(i, j), max(i, j)))
    # same file content is a duplicate by definition
    for h, idx in gh.groupby("content_hash").groups.items():
        for i, j in itertools.combinations(sorted(idx), 2): cand.add((i, j))
    print(f"{n} images, {len(cand)} candidate pairs to verify", flush=True)
    # 2. verify
    pairs = []
    for c, (i, j) in enumerate(sorted(cand)):
        same_file = gh.content_hash[i] == gh.content_hash[j]
        inl = 999 if same_file else sift_inliers(kps[i][0], kps[j], sift, bf)
        pairs.append((i, j, inl, int(H[i, j]), same_file))
        if c % 1000 == 0: print(c, flush=True)
    P = pd.DataFrame(pairs, columns=["i", "j", "inliers", "phash_dist", "same_file"])
    P["a"] = gh.subject_id[P.i].values; P["b"] = gh.subject_id[P.j].values
    P["hb_a"] = gh.hb[P.i].values; P["hb_b"] = gh.hb[P.j].values
    P.to_csv(OUT.replace(".csv", "_pairs.csv"), index=False)
    # 3. connected components at two strictness levels
    out = lab[["subject_id"]].copy()
    u = (gh.usable == 1).values
    for name, thr in (("group_strict", STRICT), ("group_lenient", LENIENT)):
        parent = list(range(n))
        def find(x):
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for r in P[P.inliers >= thr].itertuples():
            parent[find(r.i)] = find(r.j)
        root = {gh.subject_id[i]: "G%03d" % find(i) for i in range(n)}
        out[name] = out.subject_id.map(root).fillna(out.subject_id)      # Eyes-Defy: every photo is its own group
        g = out.set_index("subject_id").loc[gh.subject_id[u], name]; vc = g.value_counts()
        print(f"{name} (>= {thr} matching keypoints): {int(vc[vc > 1].sum())} of the {u.sum()} usable Ghana images "
              f"share a group with another usable image")
    out.to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
