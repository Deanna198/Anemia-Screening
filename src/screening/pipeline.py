"""The screening pipeline used by the demo app:

    photo -> conjunctiva mask (automatic, or drawn by the user) -> quality check -> colour features
          -> each method's calibrated probability -> anaemic / not anaemic / uncertain

Nothing here is trained; it loads models/screening_models.joblib (train_final.py) and, if present,
models/conj_seg.onnx (notebook 03).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "study")); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, cv2, joblib
from screening import quality
from features_v2 import all_feats, clean_mask

ROOT = os.path.dirname(os.path.dirname(HERE))
MODELS = os.path.join(ROOT, "models")


def who_cutoff(age_years, sex):
    """WHO 2024 haemoglobin cutoff (g/dL); non-pregnant, sea level."""
    if age_years < 2: return 10.5
    if age_years < 5: return 11.0
    if age_years < 12: return 11.5
    if age_years < 15: return 12.0
    return 13.0 if str(sex).upper().startswith("M") else 12.0


class Segmenter:
    """Runs conj_seg.onnx (U-Net from notebook 03) on a whole-eye photo."""
    MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, path=os.path.join(MODELS, "conj_seg.onnx")):
        self.ok = os.path.exists(path)
        if self.ok:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            _, _, self.H, self.W = self.sess.get_inputs()[0].shape        # input size the model was trained at

    def _predict(self, im):
        x = cv2.resize(im, (self.W, self.H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = ((x - self.MEAN) / self.STD).transpose(2, 0, 1)[None].astype(np.float32)
        logit = self.sess.run(None, {"image": x})[0][0, 0]
        logit = (logit + self.sess.run(None, {"image": np.ascontiguousarray(x[:, :, :, ::-1])})[0][0, 0][:, ::-1]) / 2
        m = (logit > 0).astype(np.uint8)
        n, cc, st, _ = cv2.connectedComponentsWithStats(m)
        if n > 1: m = (cc == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        conf = float((1 / (1 + np.exp(-logit)))[m > 0].sum())          # how much confident eyelid it found
        return cv2.resize(m, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST), conf

    def __call__(self, rgb):
        if not self.ok: return None
        # the model was trained on upright portrait photos; try the orientations a phone photo can have
        ks = (0, 2) if rgb.shape[0] >= rgb.shape[1] else (1, 3)
        best = None
        for k in ks:
            m, conf = self._predict(np.ascontiguousarray(np.rot90(rgb, k)))
            if best is None or conf > best[1]: best = (np.ascontiguousarray(np.rot90(m, -k)), conf)
        return best[0]


class Screener:
    def __init__(self, models_path=os.path.join(MODELS, "screening_models.joblib")):
        try:
            self.bundle = joblib.load(models_path)
        except Exception as e:   # missing, or saved with a different scikit-learn version: retrain here (1-2 min)
            print(f"could not load {models_path} ({type(e).__name__}); training the screening models now...")
            from screening import train_final
            train_final.main(); self.bundle = joblib.load(models_path)
        self.segment = Segmenter()

    def run(self, rgb, mask, age_years=25, sex="F"):
        """rgb: HxWx3 uint8 RGB; mask: HxW (nonzero = conjunctiva)."""
        mask = (mask > 0).astype(np.uint8)
        ok, qrep = quality.check(rgb, mask)
        out = {"quality_ok": ok, "quality": qrep, "who_cutoff": who_cutoff(age_years, sex), "methods": []}
        if mask.sum() < 50:
            out["answer"] = "retake"; return out
        h, w = mask.shape; s = 1000 / max(h, w)
        f = rgb.astype(np.float32) / 255
        if s < 1:
            f = cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_NEAREST)
        feats = all_feats(f, clean_mask(mask))
        for m in self.bundle["methods"]:
            x = np.nan_to_num(np.array([[feats[c] for c in m["features"]]], np.float64))
            sc = m["model"].predict_proba(x)[:, 1] if hasattr(m["model"], "predict_proba") else m["model"].decision_function(x)
            p = float(m["platt"].predict_proba(np.asarray(sc).reshape(-1, 1))[0, 1])
            call = "anaemic" if p >= m["hi"] else "not anaemic" if p <= m["lo"] else "uncertain"
            out["methods"].append({"method": m["label"], "probability": p, "answer": call, "lo": m["lo"], "hi": m["hi"]})
        main = out["methods"][0]
        out["answer"] = main["answer"] if ok else "retake"
        out["probability"] = main["probability"]
        return out
