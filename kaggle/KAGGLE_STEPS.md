# Running notebook 01 on Kaggle

Files you need (both are in this project folder):
- `kaggle/anemia-data.zip` (72 MB) — the cleaned data
- `notebooks/01_model_comparison_and_hb_range.ipynb` — the notebook

## 1. Upload the data (once)
1. kaggle.com → **Create** → **New Dataset**.
2. Drag in `anemia-data.zip`. Title it `anemia-data`.
3. Set visibility to **Private**. Eyes-Defy-Anemia's terms do not allow re-publishing it.
4. **Create**. Wait until it finishes processing.

## 2. Create the notebook
1. **Create** → **New Notebook**.
2. **File → Import Notebook** → upload `01_model_comparison_and_hb_range.ipynb`.
3. Right-hand panel → **Add Input** → **Your Datasets** → `anemia-data`.
4. Right-hand panel → **Session options**:
   - **Accelerator: GPU T4 x2** (or P100)
   - **Internet: On** (needs a phone-verified account; used once to download pretrained networks)

## 3. Run
- **Run All** for a quick look, or **Save Version → Save & Run All (Commit)** so it keeps running
  with the tab closed. Expect roughly 30–60 minutes.

## 4. Bring the results back
- Open the finished version → **Output** → **Download All**.
- Unzip into `MajorProject/results/kaggle_run_01/`.

What comes back: `01_leaderboard.csv`, `01_all_results.csv`, `02_hb_range.csv`, four plots,
`final_hb_model.joblib`, `embeddings.npz`, `summary.json`.

## What is inside anemia-data.zip
- `labels.csv`, `features.csv` — one row per subject (Hb, age, sex, WHO 2024 cutoff, colour features)
- `conj/` — conjunctiva cut-outs (927: 710 CP-AnemiC, 217 Eyes-Defy)
- `eye/`, `mask/` — whole-eye photos and conjunctiva masks for Eyes-Defy (for the segmentation notebook later)

Built by `src/make_kaggle_package.py`.
