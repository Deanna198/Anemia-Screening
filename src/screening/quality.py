"""Quality check: decide whether a conjunctiva image is good enough to measure, and if not, say why.

Each check returns a plain-language reason the app can show ("Photo is blurry - hold the phone still").
Thresholds were set from the two datasets (see src/study/exp_quality.py): they reject very few of the real
dataset images and catch most artificially damaged ones.

Usage:
    from screening.quality import check
    ok, report = check(rgb, mask)        # rgb: HxWx3 uint8 RGB, mask: HxW 0/1 (conjunctiva)
"""
import numpy as np, cv2

# thresholds (see exp_quality.py for how they were chosen)
MIN_PIXELS = 1000          # conjunctiva area after resizing the long side to 1000 px
MIN_SPAN = 0.20            # in a whole-eye photo, the eyelid region should span at least 20% of the photo width
MAX_GLARE = 0.15           # share of conjunctiva pixels that are specular highlights
MIN_BRIGHT = 0.18          # median brightness (V, 0-1) of the conjunctiva
MAX_CLIPPED = 0.03         # share of conjunctiva pixels with two or more channels saturated (near 255)
MIN_SHARP = 0.42           # sharpness: fine-detail contrast inside the conjunctiva (image long side 1000 px)


def _prep(rgb, mask):
    h, w = mask.shape; s = 1000 / max(h, w)
    if s < 1:
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    return rgb, (mask > 0).astype(np.uint8)


def measures(rgb, mask):
    rgb, mask = _prep(rgb, mask)
    n = int(mask.sum())
    if n < 50: return {"pixels": n, "span": 0.0, "glare": 1.0, "brightness": 0.0, "clipped": 1.0, "sharpness": 0.0}
    xs_ = np.nonzero(mask.any(0))[0]
    whole_photo = mask.mean() < 0.4                      # a cut-out is mostly mask; a whole-eye photo is not
    span = (xs_.max() - xs_.min() + 1) / mask.shape[1] if whole_photo else 1.0
    px = rgb[mask > 0].astype(np.float32) / 255
    v = px.max(1); sat = (v - px.min(1)) / np.maximum(v, 1e-6)
    ys, xs = np.nonzero(mask)
    crop = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]; m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    m2 = cv2.erode(m, np.ones((9, 9), np.uint8)); m2 = m2 if m2.sum() > 100 else m
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    detail = g - cv2.GaussianBlur(g, (0, 0), 1)          # what a blur removes first
    return {"pixels": n, "span": float(span), "glare": float(((v > 0.92) & (sat < 0.25)).mean()), "brightness": float(np.median(v)),
            "clipped": float(((px >= 0.985).sum(1) >= 2).mean()), "sharpness": float(detail[m2 > 0].std())}


def check(rgb, mask):
    q = measures(rgb, mask)
    tests = [
        ("conjunctiva visible", q["pixels"] >= MIN_PIXELS and q["span"] >= MIN_SPAN,
         "Not enough inner eyelid is visible - gently pull the lower eyelid down and move closer."),
        ("sharp", q["sharpness"] >= MIN_SHARP, "The photo is blurry - hold the phone still and tap to focus on the eyelid."),
        ("not too dark", q["brightness"] >= MIN_BRIGHT, "The photo is too dark - face a window or switch on a light."),
        ("not overexposed", q["clipped"] <= MAX_CLIPPED, "The photo is overexposed - turn off the flash or move away from direct light."),
        ("little glare", q["glare"] <= MAX_GLARE, "Too much glare on the eyelid - tilt the phone slightly to move the reflection."),
    ]
    report = [{"check": name, "passed": bool(ok), "message": "" if ok else msg} for name, ok, msg in tests]
    return all(r["passed"] for r in report), {"measures": q, "checks": report}
