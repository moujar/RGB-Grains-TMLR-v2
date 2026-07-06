# RGB-Grains — Exploratory Data Analysis

Generated from `.` · **32,768 grain crops** across **4 folders**.

## 1. Inventory

| dataset                                    | kind   |   grains |
|:-------------------------------------------|:-------|---------:|
| SCOOP-R2022-bacs_processed                 | bac    |    11913 |
| perfomix_2019-2020_IE_HSI_mix_processed    | mix    |     6750 |
| perfomix_2020-2021_IE_HSI_var1-8_processed | pure   |    13370 |
| withdraw_var5_2020_x39y20-var5             | pure   |      735 |

## 2. Classes

**pure** — 8 classes, 14,105 grains, imbalance ratio 1.53 (max/min class).

**bac** — 4 classes (bacs → varieties), 11,913 grains, imbalance ratio 1.50 (max/min class).

**mix** — 34 classes, 6,750 grains, imbalance ratio 14.14 (max/min class).

## 3. Split-relevant structure (microplots)

- **pure**: 17 microplots; 2–3 microplots per class (median 2). Splits are by microplot, so this bounds the number of achievable cross-validation folds.
- **SCOOP**: 11 microplots; 2–3 microplots per class (median 3). Splits are by microplot, so this bounds the number of achievable cross-validation folds.

## 4. Morphology & radiometry (sampled)

Sampled **14,750 grains** for pixel-level stats.

- Active-pixel area: median **8511**, IQR [7402, 9564], range [0, 19961].
- **295** sampled grains fall below the 2nd-percentile area (candidate dust/half-grain outliers → README §5.5 cleaning).
- Mean foreground reflectance per band (all sampled):
  - band 22: mean 0.202 ± 0.047
  - band 53: mean 0.326 ± 0.069
  - band 89: mean 0.370 ± 0.077
