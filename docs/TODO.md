# RGB Grains TMLR v2

## Status (as of the 2026-07 restructuring)

| Item | Status |
|---|---|
| 5.1 Easy fixes (all 5 bullets) | **Done** |
| 5.2 Mengtsu et al. dataset | **Blocked** — no data present in this repo; add a `dataset_choice` mode in `rgb_grains/data/dataset.py` (follow the `SCOOP` branch) once the files are available |
| 5.2 bis SCOOP/BACS dataset | **Done** — `--dataset-choice SCOOP --splitting-choice bacs_2train_1test` |
| 5.3 Downsampling image resolution | **Done** — `--downsample-kernel`/`--downsample-mode` on `rgb_grains.train` / `rgb_grains.pipeline` |
| 5.4 Factoring the code better | **Partially done** — full package restructuring (this commit), de-duplicated the two `npz_to_jpg` scripts. The hand-rolled ConvNeXt-Tiny/augmentation code was deliberately left untouched — see README "Project status / TODO" for why. |
| 5.5 Data cleaning (by size) | **Done** — `rgb_grains/data/cleaning.py`, `--clean-data` flag (both in-loader filtering and the pipeline's one-time physical move-to-`_excluded`) |
| 5.5 Data cleaning (manual tagging / outlier detection) | Not done (bigger work, no strong need identified yet) |

See `README.md` for usage. The original request, verbatim, follows.

## 5. TODOs

To Oudoum & the M1 team, or everyone in general, there are a number of things I would like to have, based on my newer version of the model's code:

---

### 5.1 Easy fixes

- **(urgent) code:** Clean up the best model recording. There is some confusion between (1) best val acc, (2) best val acc + SWA, and (3) last model. At some point the code was averaging logits of (1) and (2). I am not sure I cleaned this up properly. In the end, only the logits of one of them should be used — preferably the ones of best val acc + SWA since that point, or simply best bal acc (since SWA did not seem to play a major role in our case).
- **(easy)** Display some examples of inputs after the data augmentation (imshow ~100 augmented grains).
- **(easy)** Display some examples of failed predictions (imshow the original grains).
- **(easy)** Confusion matrices: also display per-class recall and per-class precision on the sides (one horizontally, one vertically — make the sensible choice).
- **(relatively easy)** Add computation of train balanced accuracy on the best model, using minimal augmentations (32 views) only, to assess overfitting (this bal acc should be higher than the validation one, hopefully not by a lot).

---

### 5.2 Mengtsu et al. Dataset

**(big work, worth it)** Adapt the code pipeline to apply to the data of Mengtsu et al. I downloaded their files (had to click them one by one). I will share that next week.

- Build appropriate dataset mode (or new dataset class, as you prefer).
- Run the model on it (I can do the long runs).
- Report results.

This is quite a nice added value for resubmission to TMLR, especially if we can show that fine-tuning + data augmentation allow us to improve over the results of Mengtsu et al, or do as well at much smaller compute cost. I expect it will.

---

### 5.2 bis \[optional but desirable\] Martin's dataset (SCOOP BACS)

**(relatively easy)** Add a mode to handle that data. Split by microplot, again. There are 3 bacs per species. As Martin said:

> Les grains viennent de 4 variétés, qui sont répétées dans 3 bacs chacun (équivalent à 3 micro-parcelles différentes), toujours en culture pure.

| Variety    | Bacs        |
|------------|-------------|
| EL4X-199   | 14, 18, 43  |
| EL4X-35    | 17, 71, 74  |
| EL4X-482   | 41, 42, 50  |
| GQ4X-83    | 7, 57, 92   |

- K=4 class classification; split by bac (~microplot).
- Take 2 bacs for train and 1 for test. Cross-validate on 3 folds (choice of which bac is the test one). Ignore the combinatorics of choosing the test bac for each species (like I did for the perfomix data).
- This is basically what is written in the case `if splitting_choice == "muPlot_year1only"`: one has only to edit the beginning of `load_datasets_microplot_split()` and use that mode, `"muPlot_year1only"`, maybe hacking `yearChosen` so it works too. There is no mixed data in that dataset, but it's ok — `df_mixed` can be empty and things will work fine.

---

### 5.3 \[optional but desirable\] Downsampling image resolution

**(relatively easy)** Deliberately reduce the resolution of input images during dataloading (with either meanpool or maxpool, using increasing 2×2, 3×3, 4×4, etc. pooling kernel sizes) to test how much accuracy depends on resolution (related to Phuoc's task).

- Implement a couple of downresolution options (meanpool/maxpool, with kernel size as an argument so it can be controlled when running the script).
- Run the fine-tuning + test (either collab, or I can do it if you coded it well) for each downresolution (starting from no downresolution at all, for control).
- If the downresolution tests still provide good accuracy, then it's worth asking Timothée to take nice-resolution RGB pictures with some decent (non-HSI) camera.
- After we manage to segment these RGB grains (low res or preferably higher resolution), especially if the downresolution tests are positive: run the model on those (adapt dataloader to this new "dataset"). Not too much hope for good results there.

---

### 5.4 \[very optional\] Factoring the code better

**(Optional, can become big work)** Refactor the code to make it much cleaner and professional: less code duplication, only one model (convnext), no hard-coding of data augmentations when they are available via torch, etc.

---

### 5.5 \[very optional\] Data cleaning

**(Optional, can become big work)** Data cleaning: there is sometimes a bit of dust or other artifacts of data collection that are seen as grains by watershed and then used for train or test. We should get rid of those (not a lot, really). Ideally, with some clean way to pre-process the data (area of non-zero values, etc.) or by hand (feeding an exclusion list):

- **(relatively easy) By size:** Areas of active pixels less than a threshold → put in the exclusion list, or directly move from the `_processed` folder to an `_excluded` folder.
  - **BACS:** A rather simple criterion is `np.sum(img > 0.01) < 110000` — if so, it's small enough to remove with little loss (some half grains will be lost, which is ok).
  - **Perfomix:** To be studied.
- **(easy but time-consuming) Manually:** With a simple tagging tool (user deletes anomalous jpgs, then collects the jpg paths, which are converted to npz paths and feed an exclusion list).
- **(bigger work) By OD (Outlier Detection):** Remove all outliers (no labels used).
