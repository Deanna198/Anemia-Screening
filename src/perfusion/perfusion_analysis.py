"""Perfusion experiment: does eyelid colour change when blood flow changes but Hb does not?

Hb cannot change in a few minutes, so any colour change between a resting photo and a photo after
standing up quickly or light exercise is caused by blood flow (perfusion), not anaemia. The question is how big
that change is compared with the colour difference that corresponds to a real change in Hb.

PROTOCOL (see data/perfusion/README.md)
  For each volunteer, same phone, same room, same lighting, flash off, same distance:
    <id>_rest.jpg      after 5 minutes sitting quietly
    <id>_after.jpg     straight after the change (e.g. 30 s of squats, or lying 5 min then standing up)
    <id>_recovery.jpg  (optional) 5 minutes later, sitting again
  Put the photos in data/perfusion/. Optional hand-drawn masks: <id>_rest_mask.png etc. (white = eyelid);
  otherwise models/conj_seg.onnx finds the eyelid automatically.

Run:  python src/perfusion/perfusion_analysis.py
Out:  results/perfusion/perfusion_measures.csv, perfusion_summary.md, perfusion_change.png
"""
import os, sys, glob, re
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.dirname(HERE); ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC); sys.path.insert(0, os.path.join(SRC, "study"))
import numpy as np, pandas as pd, cv2
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from features_v2 import all_feats, clean_mask
from screening.pipeline import Segmenter
from screening import quality
import common as C

IN = os.environ.get("PERFUSION_DIR", os.path.join(ROOT, "data", "perfusion"))
OUT = os.environ.get("PERFUSION_OUT", os.path.join(ROOT, "results", "perfusion"))
MEASURES = {"ratio_logRG_mean": "redness, log(R/G)", "ratio_a_mean": "redness, Lab a*", "ratio_rch_mean": "red share, R/(R+G+B)",
            "ratio_S_mean": "saturation"}
STATES = ["rest", "after", "recovery"]


def hb_slopes():
    """How much each measure changes per 1 g/dL of Hb, from Eyes-Defy adults (hand-drawn masks)."""
    df = C.load("none"); d = df[df.site != "Ghana"]
    return {m: stats.linregress(d.hb.values, d[m].values).slope for m in MEASURES}


def measure(path, seg):
    rgb = cv2.imread(path)[:, :, ::-1]
    mp = path.rsplit(".", 1)[0] + "_mask.png"
    if os.path.exists(mp): m = (cv2.imread(mp, 0) > 127).astype(np.uint8); src = "hand-drawn"
    else: m = seg(rgb); src = "automatic"
    if m is None or m.sum() < 200: return None
    ok, q = quality.check(rgb, m)
    h, w = m.shape; s = 1000 / max(h, w); f = rgb.astype(np.float32) / 255
    if s < 1:
        f = cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_NEAREST)
    F = all_feats(f, clean_mask(m))
    return {"mask": src, "quality_ok": ok, **{k: F[k] for k in MEASURES}}


def main():
    os.makedirs(OUT, exist_ok=True); seg = Segmenter()
    rows = []
    for p in sorted(glob.glob(os.path.join(IN, "*.*"))):
        mt = re.match(r"(.+)_(rest|after|recovery)\.(jpe?g|png|heic)$", os.path.basename(p), re.I)
        if not mt or p.endswith("_mask.png"): continue
        r = measure(p, seg)
        if r is None: print("no eyelid found in", p); continue
        rows.append({"volunteer": mt.group(1), "state": mt.group(2).lower(), "file": os.path.basename(p), **r})
    if not rows:
        print(f"no photos found in {IN} (expected names like v01_rest.jpg, v01_after.jpg)"); return
    D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "perfusion_measures.csv"), index=False)
    slope = hb_slopes()
    W = D.pivot_table(index="volunteer", columns="state", values=list(MEASURES))
    lines = [f"# Perfusion experiment\n\n{D.volunteer.nunique()} volunteers, {len(D)} photos "
             f"({(~D.quality_ok).sum()} failed the quality check and should be retaken).\n",
             "| Measure | Mean change, rest → after | 95% CI | p (paired t) | Same size as an Hb change of |",
             "|---|---|---|---|---|"]
    for m, name in MEASURES.items():
        if (m, "after") not in W or (m, "rest") not in W: continue
        d = (W[(m, "after")] - W[(m, "rest")]).dropna()
        if len(d) < 2: continue
        t = stats.ttest_1samp(d, 0); ci = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
        lines.append(f"| {name} | {d.mean():+.4f} | {ci[0]:+.4f} to {ci[1]:+.4f} | {t.pvalue:.3f} | {d.mean() / slope[m]:+.2f} g/dL |")
    lines.append("\nThe last column divides the colour change by how much that measure changes per 1 g/dL of Hb across "
                 "the Eyes-Defy adults. Example: +0.8 g/dL means blood flow alone moved the colour as much as 0.8 g/dL more Hb would.")
    open(os.path.join(OUT, "perfusion_summary.md"), "w").write("\n".join(lines)); print("\n".join(lines))
    fig, ax = plt.subplots(1, len(MEASURES), figsize=(3.2 * len(MEASURES), 3.4))
    order = [s for s in STATES if s in D.state.unique()]
    for a, (m, name) in zip(ax, MEASURES.items()):
        for v, g in D.groupby("volunteer"):
            g = g.set_index("state").reindex(order)
            a.plot(range(len(order)), g[m].values, color="#2a78d6", alpha=0.6, lw=1.5, marker="o", ms=4)
        a.set_xticks(range(len(order))); a.set_xticklabels(order); a.set_title(name, loc="left", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "perfusion_change.png"), dpi=130)


if __name__ == "__main__":
    main()
