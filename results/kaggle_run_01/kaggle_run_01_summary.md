# Kaggle run 01 — summary (16 Nov 2026)

Notebook: `notebooks/01_model_comparison_and_hb_range.ipynb`, GPU T4, 1713 s, 5 repeats, 30 CNN epochs.
Copied from the run log. The full CSVs, plots and `final_hb_model.joblib` are on Kaggle
(notebook → Output → Download).

## Top configurations (AUC)

| Features | Model | Headline | Within Ghana | Within India | Within Italy | Ghana→India | Ghana→Italy | India+Italy→Ghana |
|---|---|---|---|---|---|---|---|---|
| colour_ratio + EfficientNet summary | random forest | 0.734 | 0.744 | 0.644 | 0.916 | 0.656 | 0.843 | 0.599 |
| colour_all | logistic | 0.730 | 0.638 | 0.635 | 0.855 | 0.709 | 0.922 | 0.621 |
| colour_ratio | random forest | 0.725 | 0.746 | 0.641 | 0.925 | 0.663 | 0.820 | 0.557 |
| colour_all | random forest | 0.725 | 0.762 | 0.656 | 0.917 | 0.651 | 0.779 | 0.586 |
| colour_ratio | logistic | 0.725 | 0.655 | 0.628 | 0.865 | 0.697 | 0.892 | 0.616 |
| EfficientNet-B0 embedding | SVM | 0.678 | 0.711 | 0.655 | 0.881 | 0.601 | 0.675 | 0.546 |

Fine-tuned EfficientNet-B0 ranked 36 of 61.

## Hb estimation (ridge on colour features, tuned)

| Test | MAE g/dL | MAE of always guessing the mean |
|---|---|---|
| Within Ghana | 1.64 | 1.71 |
| Within India | 1.47 | 1.69 |
| Within Italy | 1.26 | 1.58 |
| Ghana → India+Italy | 2.11 | – |
| India+Italy → Ghana | 1.87 | – |

## Hb range (conformal, 80% target)

| Test | Coverage | Width g/dL | Decided | Accuracy when decided |
|---|---|---|---|---|
| Within Ghana | 0.80 | 5.11 | 1% | – |
| Within India | 0.76 | 4.10 | 16% | 0.73 |
| Within Italy | 0.84 | 4.10 | 25% | 0.97 |
| Ghana → India | 0.79 | 5.28 | 0% | – |
| Ghana → Italy | 0.52 | 5.28 | 0% | – |
| India+Italy → Ghana | 0.64 | 4.18 | 26% | 0.71 |
