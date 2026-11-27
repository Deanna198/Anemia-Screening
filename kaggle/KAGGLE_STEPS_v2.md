# Running notebooks 02 and 03 on Kaggle

Both use the same private dataset as notebook 01, updated to **version 2** (it now includes `groups.csv`,
the near-duplicate groups).

## 1. Update the dataset (once, about 5 minutes)
1. kaggle.com → **Your Work → Datasets → anemia-data** → **New Version**.
2. Upload `kaggle/anemia-data.zip` from the project folder (it was rebuilt on 18 Nov and contains `groups.csv`).
3. Keep visibility **Private** (Eyes-Defy-Anemia's terms do not allow re-publishing it) → **Create**.

## 2. Notebook 02 — deep methods (about 40–70 min on a GPU)
1. **Create → New Notebook → File → Import Notebook** → `notebooks/02_deep_methods.ipynb`.
2. **Add Input → Your Datasets → anemia-data** (version 2).
3. **Session options:** Accelerator **GPU T4 x2** (or P100), Internet **On**.
4. **Save Version → Save & Run All (Commit)**. You can close the tab; it keeps running.
5. When it finishes: open the version → **Output → Download All**, and put these files into
   `MajorProject/results/study/`:
   `embeddings_resnet50.npz`, `embeddings_efficientnet_b0.npz`, `embeddings_convnext_tiny.npz`,
   `embeddings_dinov2_small.npz`, `finetune_scores_efficientnet_b0.csv`, `finetune_scores_convnext_tiny.csv`.

## 3. Notebook 03 — segmentation (about 25–40 min on a GPU)
Same steps with `notebooks/03_segmentation.ipynb`. From its output:
- `conj_seg.onnx` → `MajorProject/models/` (replaces the CPU version; the app picks it up automatically)
- the `pred_masks` folder and `seg_cv_scores.csv`, `seg_examples.png` → `MajorProject/results/study/`

You can run 02 and 03 at the same time in two notebooks (Kaggle allows two GPU sessions).

## 4. Back on the laptop
```
python src/study/exp_models.py        # adds the 4 deep feature sets to the comparison (about 15 min)
python src/study/exp_segmentation.py  # automatic vs hand-drawn masks
python src/study/report.py            # adds the fine-tuned networks, rebuilds tables and figures
```
Or tell Claude the files are in place and it will run these and summarise the results.
