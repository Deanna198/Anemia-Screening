"""Build the study's summary tables and figures from the experiment outputs.

Run after the experiments (and again after copying the Kaggle outputs into results/study/):
    python src/study/report.py
Out: results/study/*_summary.csv, B_top_methods.csv, B_significance.csv, results/study/figures/*.png, STUDY_RESULTS.md
"""
import os, sys, glob, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import common as C

FIG = os.path.join(C.STUDY, "figures"); os.makedirs(FIG, exist_ok=True)
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "legend.frameon": False})
R = lambda f: pd.read_csv(os.path.join(C.STUDY, f)) if os.path.exists(os.path.join(C.STUDY, f)) else None
W = lambda d, f: d.to_csv(os.path.join(C.STUDY, f), index=False)
MD = []


def bars(ax, labels, series, names, ylabel, ylim=(0.4, 1.0)):
    n = len(series); w = 0.8 / n; x = np.arange(len(labels))
    for i, (vals, nm) in enumerate(zip(series, names)):
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w * 0.92, color=SER[i], label=nm, zorder=3)
    ax.axhline(0.5, color=INK2, lw=1, ls=":", zorder=2); ax.text(-0.45, 0.505, "coin toss", ha="left", va="bottom", color=INK2, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0); ax.set_ylim(*ylim); ax.set_ylabel(ylabel); ax.grid(axis="x", visible=False)


# ---------------------------------------------------------------- A
def part_a():
    A = R("A_protocol.csv")
    if A is None: return
    keys = [c for c in A.columns if c not in ("features", "model")]
    S = A.set_index(["features", "model"])[keys].T
    S.columns = [f"{m} · {f}" for f, m in S.columns]
    S.insert(0, "average of all 8", S.mean(1)); S = S.reset_index().rename(columns={"index": "evaluation method"})
    W(S.round(3), "A_protocol_summary.csv")
    sub = A[A.features == "colour + texture"].set_index("model")
    gk = ["Ghana: random split, all 710 images", "Ghana: random split, exact copies removed", "Ghana: copies grouped (strict)", "India+Italy → Ghana"]
    lab = ["random split,\nall 710 images", "random split,\nexact copies\nremoved", "near-copies\nkept together", "trained on\nIndia+Italy"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [4, 3]})
    bars(ax[0], lab, [sub.loc[m, gk].values for m in sub.index], list(sub.index), "AUC on Ghana (CP-AnemiC)")
    ax[0].set_title("Ghana: the score depends on how you test", loc="left", color=INK)
    ek = ["Eyes-Defy: pooled, scored together", "India: within site", "Italy: within site"]
    bars(ax[1], ["India + Italy\nscored together", "India\nalone", "Italy\nalone"], [sub.loc[m, ek].values for m in sub.index], list(sub.index), "AUC")
    ax[1].set_title("Eyes-Defy: pooling sites inflates the score", loc="left", color=INK)
    ax[0].legend(loc="upper right", fontsize=8, title="colour + texture features", title_fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig1_protocol.png"), dpi=140); plt.close()
    a = S.set_index("evaluation method")["average of all 8"]
    MD.append("## A. Evaluation method\n\nAverage AUC of 8 method combinations (4 models × 2 feature sets):\n\n| Evaluation | AUC |\n|---|---|\n" +
              "\n".join(f"| {k} | {v:.2f} |" for k, v in a.items()) +
              f"\n\nRandom forest on colour + texture: **{sub.loc['random forest', gk[0]]:.2f}** with the random split used in most "
              f"CP-AnemiC papers, **{sub.loc['random forest', gk[2]]:.2f}** with near-copies kept together, and "
              f"**{sub.loc['random forest', gk[3]]:.2f}** when trained on another population.\n")


# ---------------------------------------------------------------- B (+ fine-tuned networks)
def finetune_rows():
    rows = []
    for f in sorted(glob.glob(os.path.join(C.STUDY, "finetune_scores_*.csv"))):
        D = pd.read_csv(f); name = os.path.basename(f)[16:-4]
        r = {"features": "image (fine-tuned network)", "model": name}
        for t, g in D.groupby("test"): r[t] = C.auc(g.y.values, g.score.values)
        if all(t in r for t in C.FAIR_TESTS):
            r["fair average"] = np.mean([r[t] for t in C.FAIR_TESTS]); r["cross-population average"] = np.mean([r[t] for t in C.FAIR_TESTS[3:]])
            rows.append(r)
    return pd.DataFrame(rows)


def part_b():
    B = R("B_models_features.csv")
    if B is None: return
    B = pd.concat([B, finetune_rows()], ignore_index=True).sort_values("fair average", ascending=False)
    W(B.round(3), "B_all_methods.csv")
    cols = ["features", "model"] + C.FAIR_TESTS + ["India → Italy", "Italy → India", "fair average", "cross-population average"]
    top = B[cols].head(12).copy()
    # 95% CIs and paired comparison against the reference, from the saved out-of-sample scores
    K = pickle.load(open(os.path.join(C.STUDY, "B_scores.pkl"), "rb"))
    y, grp = K["y"], K["group"]; ref = K["scores"][("ratio colour", "logistic regression")]
    sig = []
    for _, r in top.iterrows():
        sc = K["scores"].get((r.features, r.model))
        if sc is None: continue
        for t in C.FAIR_TESTS:
            idx = K["idx"][t]; g = grp[idx] if t.startswith("Ghana (") or t.endswith("Ghana") else None
            lo, hi = C.boot_ci(y[idx], sc[t], g)
            d, (dlo, dhi), p = C.paired_boot_diff(y[idx], sc[t], ref[t], g)
            sig.append({"features": r.features, "model": r.model, "test": t, "AUC": C.auc(y[idx], sc[t]), "95% CI low": lo, "95% CI high": hi,
                        "difference vs reference": d, "diff CI low": dlo, "diff CI high": dhi, "p-value": p})
    SG = pd.DataFrame(sig); W(SG.round(3), "B_significance.csv"); W(top.round(3), "B_top_methods.csv")
    # heatmaps
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2)); plt.subplots_adjust(wspace=0.45)
    for a, key in zip(ax, ["fair average", "cross-population average"]):
        P = B[B.features != "image (fine-tuned network)"].pivot_table(index="model", columns="features", values=key)
        order = [c for c in list(C.FEATURE_SETS) + sorted(c for c in P.columns if c.startswith("deep")) if c in P.columns]
        P = P[order].loc[P.mean(1).sort_values(ascending=False).index]
        im = a.imshow(P.values, cmap="Blues", vmin=0.5, vmax=0.8, aspect="auto")
        a.set_xticks(range(P.shape[1])); a.set_xticklabels([c.replace(" + ", " +\n").replace("deep: ", "deep:\n").replace(" colour", "\ncolour").replace(" only", "\nonly") for c in P.columns], fontsize=8)
        a.set_yticks(range(P.shape[0])); a.set_yticklabels(P.index, fontsize=9); a.grid(False)
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                v = P.values[i, j]; a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.7 else INK)
        a.set_title(("Average of the 6 fair tests" if key == "fair average" else "Average of the 3 cross-population tests"), loc="left", color=INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="AUC")
    plt.savefig(os.path.join(FIG, "fig2_models_features.png"), dpi=140, bbox_inches="tight"); plt.close()
    t1 = top.iloc[0]; best_cross = B.sort_values("cross-population average", ascending=False).iloc[0]
    n_sig = SG[(SG["p-value"] < 0.05) & (SG["difference vs reference"] > 0)].groupby(["features", "model"]).size()
    MD.append("## B. Models × features (fair tests)\n\n| Rank | Method | Fair average | Cross-population average |\n|---|---|---|---|\n" +
              "\n".join(f"| {i+1} | {r.model} · {r.features} | {r['fair average']:.3f} | {r['cross-population average']:.3f} |" for i, (_, r) in enumerate(top.head(8).iterrows())) +
              f"\n\nBest on the cross-population tests: **{best_cross.model} · {best_cross.features}** ({best_cross['cross-population average']:.3f}).\n"
              f"Tests where a top-12 method beats the reference (logistic regression · ratio colour) with p < 0.05: "
              f"{', '.join(f'{m} · {f}: {n}' for (f, m), n in n_sig.items()) or 'none'}.\n")


# ---------------------------------------------------------------- C
def part_c():
    Cc = R("C_preprocessing.csv")
    if Cc is None: return
    keys = ["India", "Italy", "India → Italy", "Italy → India", "Ghana → India", "Ghana → Italy", "India+Italy → Ghana", "Eyes-Defy average (4 tests)"]
    S = Cc.groupby("colour correction", sort=False)[keys].mean().reset_index(); W(S.round(3), "C_preprocessing_summary.csv")
    order = ["none", "grayworld", "whitepatch", "sclera"]
    S2 = S.set_index("colour correction").loc[order]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    tk = ["India", "Italy", "India → Italy", "Italy → India"]
    bars(ax, tk, [S2.loc[m, tk].values for m in order], ["no correction", "grey-world", "white-patch", "sclera (white of the eye)"], "AUC (average of 9 method combinations)")
    ax.legend(loc="upper left", fontsize=8, ncol=2); ax.set_title("Colour correction on Eyes-Defy-Anemia", loc="left", color=INK)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig3_preprocessing.png"), dpi=140); plt.close()
    MD.append("## C. Colour correction\n\n| Correction | India | Italy | India → Italy | Italy → India | Eyes-Defy average |\n|---|---|---|---|---|---|\n" +
              "\n".join(f"| {k} | {r['India']:.3f} | {r['Italy']:.3f} | {r['India → Italy']:.3f} | {r['Italy → India']:.3f} | {r['Eyes-Defy average (4 tests)']:.3f} |" for k, r in S.set_index('colour correction').iterrows()) + "\n")


# ---------------------------------------------------------------- E
def part_e():
    cv, op = R("E_abstention_curve.csv"), R("E_abstention_operating_point.csv")
    if cv is None: return
    tests = list(dict.fromkeys(cv.test)); meths = list(dict.fromkeys(cv.method))
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, sharey=True)
    for a, t in zip(ax.flat, tests):
        for i, m in enumerate(meths):
            d = cv[(cv.test == t) & (cv.method == m)]
            a.plot(d.coverage * 100, d.accuracy * 100, color=SER[i], lw=2, marker="o", ms=4, label=m.split(" (", 1)[1].rstrip(")") + (" (best)" if m.startswith("best") else " (reference)"))
        a.set_title(t, loc="left", color=INK, fontsize=10); a.invert_xaxis(); a.set_ylim(20, 100)
    for a in ax[1]: a.set_xlabel("share of cases answered (%)")
    for a in ax[:, 0]: a.set_ylabel("accuracy on answered cases (%)")
    ax[0, 0].legend(fontsize=8, loc="lower left")
    plt.suptitle("Answering fewer, more confident cases raises accuracy", x=0.01, ha="left", color=INK)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig4_abstention.png"), dpi=140); plt.close()
    o = op[op.method == meths[0]]
    MD.append(f"## E. Uncertain answers ({meths[0]})\n\n| Test | Accuracy, always answering | Share answered | Accuracy when answered |\n|---|---|---|---|\n" +
              "\n".join(f"| {r.test} | {r['accuracy if always answering']:.0%} | {r['share answered']:.0%} | {r['accuracy when answered']:.0%} |" for _, r in o.iterrows()) + "\n")


# ---------------------------------------------------------------- F
def part_f():
    F = R("F_hb_regression.csv")
    if F is None: return
    mae = [c for c in F.columns if c.endswith("MAE g/dL")]
    S = F[["features", "model"] + mae].copy(); W(S.round(2), "F_hb_summary.csv")
    sub = F[F.features == "ratio colour"].set_index("model")
    fig, ax = plt.subplots(figsize=(10, 4.2)); x = np.arange(len(mae)); n = len(sub); w = 0.8 / n
    for i, (m, r) in enumerate(sub.iterrows()):
        ax.bar(x + (i - (n - 1) / 2) * w, r[mae].values, w * 0.92, color=INK2 if m.startswith("baseline") else SER[i - 1], label=m, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([m.replace(": MAE g/dL", "") for m in mae], fontsize=9); ax.grid(axis="x", visible=False)
    ax.set_ylabel("average Hb error (g/dL), lower is better"); ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.set_title("Estimating the Hb number: modestly better than guessing the average within a site, not across sites", loc="left", color=INK)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig5_hb.png"), dpi=140); plt.close()
    MD.append("## F. Hb estimation (ratio colour)\n\n| Model | " + " | ".join(m.replace(": MAE g/dL", "") for m in mae) + " |\n|---|" + "---|" * len(mae) + "\n" +
              "\n".join(f"| {m} | " + " | ".join(f"{v:.2f}" for v in r[mae].values) + " |" for m, r in sub.iterrows()) + "\n")


# ---------------------------------------------------------------- G + duplicates figure
def part_g():
    G = R("G_quality.csv")
    if G is not None:
        MD.append("## G. Quality check\n\n| Measure | Value |\n|---|---|\n" + "\n".join(f"| {r.iloc[0]} | {r.iloc[1]:.0%} |" for _, r in G.iterrows()) + "\n")
    P = R("groups_pairs.csv"); lab = pd.read_csv(os.path.join(C.DATA, "labels.csv")).set_index("subject_id")
    if P is None: return
    Q = P[~P.same_file & (P.inliers >= 25)]
    Q = Q[(lab.usable.reindex(Q.a).values == 1) & (lab.usable.reindex(Q.b).values == 1)].sort_values("inliers", ascending=False).head(8)
    def ld(p):
        im = cv2.imread(os.path.join(C.DATA, p), cv2.IMREAD_UNCHANGED)
        if im.shape[2] == 4: im = (im[:, :, :3] * (im[:, :, 3:] / 255)).astype(np.uint8)
        return cv2.resize(im, (240, 150))[:, :, ::-1]
    fig, ax = plt.subplots(4, 2, figsize=(10, 9.5))
    for a, r in zip(ax.flat, Q.itertuples()):
        a.imshow(np.hstack([ld(lab.conj_path[r.a]), np.full((150, 8, 3), 252, np.uint8), ld(lab.conj_path[r.b])]))
        a.set_title(f"{r.a}: Hb {r.hb_a:g}   |   {r.b}: Hb {r.hb_b:g}", fontsize=9, color=INK); a.axis("off")
    plt.suptitle("Same eye, two records, different Hb (CP-AnemiC, after removing exact copies)", x=0.01, ha="left", color=INK)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig6_duplicates.jpg"), dpi=110); plt.close()


def part_h():
    S, H = R("seg_cv_scores.csv"), R("H_segmentation.csv")
    if S is not None:
        d = S.groupby("site")[["dice", "iou"]].mean()
        MD.append(f"## H. Automatic segmentation (notebook 03, 5-fold CV over all {len(S)} Eyes-Defy photos)\n\n"
                  f"Dice {S.dice.mean():.3f} (India {d.loc['India','dice']:.3f}, Italy {d.loc['Italy','dice']:.3f}); "
                  f"IoU {S.iou.mean():.3f}; photos with Dice below 0.5: {(S.dice < 0.5).sum()}.\n")
    if H is not None:
        g = H.groupby("masks")[["India", "Italy", "India → Italy", "Italy → India", "average"]].mean()
        MD.append("Same anaemia tests with automatic vs hand-drawn masks (average of 6 method combinations):\n\n"
                  "| Masks | India | Italy | India → Italy | Italy → India | Average |\n|---|---|---|---|---|---|\n" +
                  "\n".join(f"| {k} | " + " | ".join(f"{v:.3f}" for v in r.values) + " |" for k, r in g.iterrows()) + "\n")


if __name__ == "__main__":
    for f in (part_a, part_b, part_c, part_e, part_f, part_g, part_h):
        try: f()
        except Exception as e: print("!!", f.__name__, repr(e))
    open(os.path.join(C.STUDY, "STUDY_RESULTS.md"), "w").write("# Comparison study — results\n\n_Built by `src/study/report.py`. AUC: 0.5 = coin toss, 1.0 = perfect._\n\n" + "\n".join(MD))
    print("\n".join(MD))
