"""CPU version of the conjunctiva segmentation model (no pretrained weights, no GPU needed).

A small U-Net trained from scratch on the Eyes-Defy-Anemia eye photos and hand-drawn masks.
It is the working version for the demo; notebook 03 (Kaggle, pretrained encoder) is the stronger one and
replaces this file's output when you copy its conj_seg.onnx into models/.

Honest test: 20% of photos (stratified by site) are held out, never seen in training, and scored.

Run:  python src/segmentation/train_seg_cpu.py          (about 20-30 minutes on a laptop CPU)
Out:  models/conj_seg.onnx, results/segmentation/seg_holdout_scores.csv, seg_holdout_examples.jpg
"""
import os, time
import numpy as np, pandas as pd, cv2
import torch, torch.nn as nn, torch.nn.functional as Fn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("ANEMIA_DATA", os.path.join(ROOT, "kaggle", "anemia-data"))
MODELS = os.environ.get("ANEMIA_MODELS", os.path.join(ROOT, "models"))
RES = os.environ.get("ANEMIA_SEGRES", os.path.join(ROOT, "results", "segmentation"))
EPOCHS = int(os.environ.get("EPOCHS", "60"))
W, H = 224, 288
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(max(1, os.cpu_count() or 1))
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def block(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                      nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SmallUNet(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        ch = [c, 2 * c, 4 * c, 8 * c, 16 * c]
        self.down = nn.ModuleList([block(3, ch[0])] + [block(ch[i], ch[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([block(ch[i + 1] + ch[i], ch[i]) for i in range(3, -1, -1)])
        self.head = nn.Conv2d(ch[0], 1, 1)
    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x if i == 0 else Fn.max_pool2d(x, 2)); skips.append(x)
        for u, s in zip(self.up, skips[-2::-1]):
            x = u(torch.cat([Fn.interpolate(x, size=s.shape[2:], mode="bilinear", align_corners=False), s], 1))
        return self.head(x)


def to_t(b): return torch.from_numpy(((b.astype(np.float32) / 255 - MEAN) / STD).transpose(0, 3, 1, 2).copy())


def aug(im, m, rng):
    if rng.random() < 0.5: im, m = im[:, ::-1], m[:, ::-1]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), rng.uniform(-15, 15), rng.uniform(0.9, 1.1))
    M[:, 2] += rng.uniform(-0.05, 0.05, 2) * [W, H]
    im = cv2.warpAffine(np.ascontiguousarray(im), M, (W, H), borderMode=cv2.BORDER_REFLECT)
    m = cv2.warpAffine(np.ascontiguousarray(m), M, (W, H), flags=cv2.INTER_NEAREST)
    im = im.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
    im = im * rng.uniform(0.85, 1.15, size=3)
    return np.clip(im, 0, 255).astype(np.uint8), m


def keep_largest(m):
    n, cc, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    return m if n <= 1 else (cc == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)


def main():
    os.makedirs(MODELS, exist_ok=True); os.makedirs(RES, exist_ok=True)
    lab = pd.read_csv(os.path.join(DATA, "labels.csv")); lab = lab[lab.mask_path.notna()].reset_index(drop=True)
    IM = np.stack([cv2.resize(cv2.imread(os.path.join(DATA, p)), (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1] for p in lab.eye_path])
    MK = np.stack([(cv2.resize(cv2.imread(os.path.join(DATA, p), 0), (W, H), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8) for p in lab.mask_path])
    rng = np.random.default_rng(0)
    test = np.concatenate([rng.choice(np.where(lab.site == s)[0], int(round(0.2 * (lab.site == s).sum())), replace=False) for s in ("India", "Italy")])
    train = np.setdiff1d(np.arange(len(lab)), test)
    net = SmallUNet(); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    steps = EPOCHS * int(np.ceil(len(train) / 8)); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.1)
    t0 = time.time()
    for ep in range(EPOCHS):
        net.train(); order = rng.permutation(train); tot = 0
        for i in range(0, len(order), 8):
            b = order[i:i + 8]
            if len(b) < 2: continue
            pr = [aug(IM[k], MK[k], rng) for k in b]
            x = to_t(np.stack([p[0] for p in pr])); t = torch.from_numpy(np.stack([p[1] for p in pr])[:, None].astype(np.float32))
            o = net(x); p = torch.sigmoid(o); inter = (p * t).sum((1, 2, 3))
            loss = Fn.binary_cross_entropy_with_logits(o, t) + (1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tot += loss.item() * len(b)
        if ep % 10 == 0 or ep == EPOCHS - 1: print(f"epoch {ep+1}/{EPOCHS} loss {tot/len(train):.3f} ({time.time()-t0:.0f}s)", flush=True)
    net.eval(); rows = []
    with torch.no_grad():
        P = torch.sigmoid(net(to_t(IM[test]))).numpy()[:, 0]
    for k, p in zip(test, P):
        pm = keep_largest(p > 0.5).astype(bool); g = MK[k].astype(bool)
        rows.append({"subject_id": lab.subject_id[k], "site": lab.site[k], "dice": 2 * (pm & g).sum() / max(pm.sum() + g.sum(), 1),
                     "iou": (pm & g).sum() / max((pm | g).sum(), 1)})
    R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, "seg_holdout_scores.csv"), index=False)
    print(R.groupby("site")[["dice", "iou"]].mean().round(3)); print("overall Dice", round(R.dice.mean(), 3), "IoU", round(R.iou.mean(), 3))
    tiles = []
    for k in R.sort_values("dice").index[[0, len(R) // 4, len(R) // 2, -1]]:
        i = test[k]; v = IM[i].astype(np.float32).copy(); pm = keep_largest(P[k] > 0.5).astype(bool)
        v[MK[i] > 0] = 0.5 * v[MK[i] > 0] + [0, 120, 0]; v[pm] = 0.6 * v[pm] + [100, 0, 100]
        v = np.clip(v, 0, 255).astype(np.uint8)[:, :, ::-1].copy()
        cv2.putText(v, f"Dice {R.dice[k]:.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2); tiles.append(v)
    cv2.imwrite(os.path.join(RES, "seg_holdout_examples.jpg"), np.hstack(tiles))
    kw = dict(input_names=["image"], output_names=["logit"], dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}}, opset_version=17)
    try: torch.onnx.export(net, to_t(IM[:1]), os.path.join(MODELS, "conj_seg.onnx"), dynamo=False, **kw)
    except TypeError: torch.onnx.export(net, to_t(IM[:1]), os.path.join(MODELS, "conj_seg.onnx"), **kw)
    print("saved", os.path.join(MODELS, "conj_seg.onnx"))


if __name__ == "__main__":
    main()
