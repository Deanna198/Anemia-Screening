"""Demo app: conjunctival anaemia screening + comparison-study dashboard.

Run from the project folder:
    python app/app.py
Then open the address it prints (http://127.0.0.1:7860). To use a phone on the same Wi-Fi, run
    python app/app.py --lan
and open http://<laptop-ip>:7860 on the phone.
"""
import os, sys, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd, cv2
import gradio as gr
from screening.pipeline import Screener

STUDY = os.path.join(ROOT, "results", "study")
SCR = Screener()

COLOURS = {"anaemic": "#b42318", "not anaemic": "#067647", "uncertain": "#b54708", "retake": "#475467"}
TEXT = {"anaemic": "Likely anaemic", "not anaemic": "Likely not anaemic",
        "uncertain": "Uncertain - the model is not confident either way", "retake": "Please retake the photo"}


def painted_mask(ed):
    """Mask from the brush layer of the image editor (anything painted counts)."""
    if not ed or not ed.get("layers"): return None
    m = np.zeros(ed["background"].shape[:2], np.uint8)
    for L in ed["layers"]:
        if L is None: continue
        a = L[:, :, 3] if L.ndim == 3 and L.shape[2] == 4 else (L.max(2) if L.ndim == 3 else L)
        m |= (a > 0).astype(np.uint8)
    return m if m.sum() > 50 else None


def overlay(rgb, mask):
    v = rgb.copy().astype(np.float32)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    v[mask > 0] = 0.75 * v[mask > 0] + 0.25 * np.array([0, 255, 255])
    v = v.astype(np.uint8); cv2.drawContours(v, cnts, -1, (0, 255, 255), max(2, rgb.shape[1] // 300))
    return v


def screen(ed, age, sex):
    if not ed or ed.get("background") is None:
        return None, "Upload or take a photo first.", None, None
    rgb = ed["background"][:, :, :3].astype(np.uint8)
    mask, how = painted_mask(ed), "drawn by you"
    if mask is None:
        mask, how = SCR.segment(rgb), "found automatically"
    if mask is None or mask.sum() < 50:
        return None, ("**No eyelid region.** The automatic segmentation model (`models/conj_seg.onnx`) is not installed yet, "
                      "so paint over the inner lower eyelid with the brush and press *Screen* again."), None, None
    r = SCR.run(rgb, mask, float(age or 25), sex)
    ans = r["answer"]
    md = [f"<div style='font-size:1.6em;font-weight:700;color:{COLOURS[ans]}'>{TEXT[ans]}</div>"]
    if ans != "retake":
        md.append(f"Main method's probability of anaemia: **{r['probability']:.0%}** "
                  f"(it answers only below {r['methods'][0]['lo']:.0%} or above {r['methods'][0]['hi']:.0%}).")
    else:
        md += ["**Why:**"] + [f"- {c['message']}" for c in r["quality"]["checks"] if not c["passed"]]
    md.append(f"\nAnaemia here means Hb below **{r['who_cutoff']} g/dL** (WHO 2024 cutoff for age {age:g}, {sex}). "
              f"Eyelid region {how}.")
    md.append("\n<small>Screening demonstration only - not a diagnosis. Only a blood test measures haemoglobin.</small>")
    q = pd.DataFrame([{"check": c["check"], "result": "pass" if c["passed"] else "FAIL", "note": c["message"]} for c in r["quality"]["checks"]])
    meth = pd.DataFrame([{"method": m["method"], "probability of anaemia": f"{m['probability']:.0%}", "answer": m["answer"]} for m in r["methods"]])
    return overlay(rgb, mask), "\n\n".join(md), q, meth


# ------------------------------------------------------------------ dashboard
def table(name):
    p = os.path.join(STUDY, name)
    return pd.read_csv(p).round(3) if os.path.exists(p) else pd.DataFrame({"missing": [f"run the study first: {name}"]})


def fig(name):
    p = os.path.join(STUDY, "figures", name)
    return p if os.path.exists(p) else None


SECTIONS = {
    "1. How the evaluation method changes the score": ("A_protocol_summary.csv", "fig1_protocol.png",
        "The same models score much higher when copies of the same eye are allowed on both sides of the split "
        "(how most CP-AnemiC papers test) than when copies are grouped or the model is tested on a new population."),
    "2. Models × features (fair tests)": ("B_top_methods.csv", "fig2_models_features.png",
        "Average AUC over six fair tests. The top methods are within each other's uncertainty."),
    "3. Colour correction (replaces the colour card)": ("C_preprocessing_summary.csv", "fig3_preprocessing.png",
        "Does correcting the photo's colour with the white of the eye, grey-world or white-patch help?"),
    "4. Always answering vs. allowing 'uncertain'": ("E_abstention_operating_point.csv", "fig4_abstention.png",
        "Accuracy rises when the system is allowed to say 'uncertain' on its least confident cases."),
    "5. Estimating the Hb number": ("F_hb_summary.csv", "fig5_hb.png",
        "Error in g/dL compared with simply guessing the average Hb."),
    "6. Quality check": ("G_quality.csv", None, "How often real images are rejected, and how often damaged ones are caught."),
    "7. Near-duplicate images in CP-AnemiC": (None, "fig6_duplicates.jpg", "Pairs of images of the same eye filed under different children with different Hb values."),
    "8. Automatic eyelid segmentation": ("H_segmentation.csv", "../seg_examples.png",
        "U-Net, 5-fold cross-validation over 217 photos: Dice 0.85. The anaemia tests give the same accuracy on the "
        "automatic masks as on the hand-drawn ones (green = hand-drawn, purple = predicted)."),
}


def show(section):
    t, f, note = SECTIONS[section]
    return note, (table(t) if t else pd.DataFrame()), (fig(f) if f else None)


def build():
    with gr.Blocks(title="Conjunctival anaemia screening") as demo:
        gr.Markdown("# Conjunctival anaemia screening\nA comparison study of colour-feature and deep-learning methods on "
                    "CP-AnemiC (Ghana) and Eyes-Defy-Anemia (India, Italy).")
        with gr.Tab("Screen a photo"):
            with gr.Row():
                with gr.Column():
                    ed = gr.ImageMask(type="numpy", image_mode="RGB", label="Eye photo (pull the lower eyelid down). "
                                      "Optional: paint over the inner eyelid with the brush.", sources=["upload", "webcam"],
                                      brush=gr.Brush(colors=["#00ffff"], default_size=25))
                    with gr.Row():
                        age = gr.Number(25, label="Age (years)", minimum=0.5, maximum=100)
                        sex = gr.Radio(["F", "M"], value="F", label="Sex")
                    go = gr.Button("Screen", variant="primary")
                with gr.Column():
                    res = gr.Markdown()
                    seen = gr.Image(label="Region measured", interactive=False)
                    q = gr.Dataframe(label="Quality check", interactive=False)
                    meth = gr.Dataframe(label="What each method says", interactive=False)
            go.click(screen, [ed, age, sex], [seen, res, q, meth])
        with gr.Tab("Study results"):
            sec = gr.Dropdown(list(SECTIONS), value=list(SECTIONS)[0], label="Result")
            note = gr.Markdown(); img = gr.Image(label="", interactive=False); tab = gr.Dataframe(interactive=False, wrap=True)
            sec.change(show, sec, [note, tab, img]); demo.load(show, sec, [note, tab, img])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--lan", action="store_true"); ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build().launch(server_name="0.0.0.0" if a.lan else "127.0.0.1", server_port=a.port)
