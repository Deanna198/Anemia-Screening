# Comparison study — results

_Built by `src/study/report.py`. AUC: 0.5 = coin toss, 1.0 = perfect._

## A. Evaluation method

Average AUC of 8 method combinations (4 models × 2 feature sets):

| Evaluation | AUC |
|---|---|
| Ghana: random split, all 710 images | 0.80 |
| Ghana: random split, exact copies removed | 0.71 |
| Ghana: copies grouped (strict) | 0.70 |
| Ghana: copies grouped (lenient) | 0.69 |
| Eyes-Defy: pooled, scored together | 0.88 |
| Eyes-Defy: pooled, scored on India | 0.66 |
| Eyes-Defy: pooled, scored on Italy | 0.89 |
| India: within site | 0.66 |
| Italy: within site | 0.90 |
| Ghana → India | 0.64 |
| Ghana → Italy | 0.77 |
| India+Italy → Ghana | 0.58 |

Random forest on colour + texture: **0.91** with the random split used in most CP-AnemiC papers, **0.76** with near-copies kept together, and **0.61** when trained on another population.

## B. Models × features (fair tests)

| Rank | Method | Fair average | Cross-population average |
|---|---|---|---|
| 1 | random forest · everything | 0.741 | 0.696 |
| 2 | extra trees · everything | 0.741 | 0.678 |
| 3 | LightGBM · deep: dinov2_small | 0.740 | 0.697 |
| 4 | extra trees · ratio colour | 0.739 | 0.714 |
| 5 | extra trees · colour + texture | 0.738 | 0.677 |
| 6 | XGBoost · deep: dinov2_small | 0.737 | 0.694 |
| 7 | logistic regression · ratio colour | 0.736 | 0.738 |
| 8 | convnext_tiny · image (fine-tuned network) | 0.734 | 0.726 |

Best on the cross-population tests: **logistic regression · ratio colour** (0.738).
Tests where a top-12 method beats the reference (logistic regression · ratio colour) with p < 0.05: LightGBM · colour + texture: 1, extra trees · colour + texture: 1, extra trees · everything: 1, random forest · everything: 1, extra trees · ratio + percentiles: 1, extra trees · ratio colour: 1.

## C. Colour correction

| Correction | India | Italy | India → Italy | Italy → India | Eyes-Defy average |
|---|---|---|---|---|---|
| none | 0.677 | 0.892 | 0.862 | 0.680 | 0.778 |
| grayworld | 0.668 | 0.904 | 0.711 | 0.628 | 0.728 |
| whitepatch | 0.654 | 0.885 | 0.818 | 0.654 | 0.753 |
| sclera | 0.641 | 0.879 | 0.811 | 0.638 | 0.743 |
| per-site standardisation | 0.674 | 0.894 | 0.762 | 0.661 | 0.748 |

## E. Uncertain answers (best method (random forest, everything))

| Test | Accuracy, always answering | Share answered | Accuracy when answered |
|---|---|---|---|
| Ghana (copies grouped) | 73% | 35% | 87% |
| India | 68% | 28% | 89% |
| Italy | 89% | 93% | 86% |
| Ghana → India | 31% | 52% | 37% |
| Ghana → Italy | 81% | 70% | 92% |
| India+Italy → Ghana | 59% | 59% | 66% |

## F. Hb estimation (ratio colour)

| Model | Ghana (copies grouped) | India | Italy | Ghana → India+Italy | India+Italy → Ghana |
|---|---|---|---|---|---|
| baseline: predict the average | 1.71 | 1.72 | 1.62 | 2.54 | 2.08 |
| ridge regression | 1.64 | 1.41 | 1.16 | 2.14 | 2.08 |
| SVR (RBF) | 1.54 | 1.58 | 1.22 | 2.43 | 2.17 |
| random forest | 1.46 | 1.49 | 1.19 | 2.29 | 1.99 |
| LightGBM | 1.51 | 1.57 | 1.23 | 2.20 | 1.97 |

## G. Quality check

| Measure | Value |
|---|---|
| real images, Ghana | 8% |
| real images, India | 0% |
| real images, Italy | 0% |
| caught: blur | 100% |
| caught: eyelid barely visible | 100% |
| caught: glare spot | 100% |
| caught: motion blur | 86% |
| caught: overexposed | 99% |
| caught: too dark | 100% |
| caught for the right reason: blur | 100% |
| caught for the right reason: eyelid barely visible | 100% |
| caught for the right reason: glare spot | 100% |
| caught for the right reason: motion blur | 86% |
| caught for the right reason: overexposed | 96% |
| caught for the right reason: too dark | 100% |

## H. Automatic segmentation (notebook 03, 5-fold CV over all 217 Eyes-Defy photos)

Dice 0.850 (India 0.841, Italy 0.857); IoU 0.746; photos with Dice below 0.5: 0.

Same anaemia tests with automatic vs hand-drawn masks (average of 6 method combinations):

| Masks | India | Italy | India → Italy | Italy → India | Average |
|---|---|---|---|---|---|
| automatic masks | 0.664 | 0.918 | 0.857 | 0.692 | 0.783 |
| hand-drawn masks | 0.673 | 0.897 | 0.863 | 0.686 | 0.780 |
