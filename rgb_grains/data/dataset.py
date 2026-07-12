from __future__ import annotations
import os
import glob
import re
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
from torch.utils.data import Dataset

from rgb_grains.utils.tools import *
from rgb_grains.data.cleaning import filter_excluded_files


# ── CONSTANTS ──
CH_SCALE = [1567.0, 8316.0, 18126.0]  # spectral bands [22, 53, 89]
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]


class GrainDataset_ConvNeXt(Dataset):
    def __init__(
        self,
        X,
        y=None,
        augment=False,
        crop_size=176,
        num_classes=8,
        return_one_hot=True,
        downsample_kernel=1,
        downsample_mode="mean",
    ):
        """
        downsample_kernel/downsample_mode: TODO 5.3 resolution ablation. The crop is
        pooled by a downsample_kernel x downsample_kernel window (mean or max) and then
        upsampled (nearest) back to crop_size, so the tensor shape fed to the model is
        unchanged but its effective resolution is reduced. kernel=1 is a no-op.
        """
        self.X, self.y = X, y
        self.aug, self.cs = augment, crop_size
        self.nc = num_classes
        self.return_one_hot = return_one_hot
        self.scale = torch.tensor(CH_SCALE).view(3, 1, 1)
        self.mean = torch.tensor(IMGNET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMGNET_STD).view(3, 1, 1)
        self.downsample_kernel = int(downsample_kernel)
        self.downsample_mode = downsample_mode
        if self.downsample_mode not in ("mean", "max"):
            raise ValueError(f"downsample_mode must be 'mean' or 'max', got {downsample_mode!r}")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        img = self.X[idx]
        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)
        x = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        x = (x / self.scale).clamp_(0.0, 1.0)
        _, H, W = x.shape
        s = self.cs

        if self.aug:
            # random crop with spatial jitter around centre
            if H > s and W > s:
                mo = min(28, (H - s) // 2)
                cy, cx = H // 2, W // 2
                top = cy - s // 2 + int(torch.randint(-mo, mo + 1, (1,)))
                left = cx - s // 2 + int(torch.randint(-mo, mo + 1, (1,)))
                top = max(0, min(top, H - s))
                left = max(0, min(left, W - s))
                x = x[:, top : top + s, left : left + s]
            # D4 symmetry
            if torch.rand(1) < 0.5:
                x = x.flip(2)
            if torch.rand(1) < 0.5:
                x = x.flip(1)
            k = int(torch.randint(0, 4, (1,)))
            if k:
                x = torch.rot90(x, k, [1, 2])
            # per-channel brightness
            if torch.rand(1) < 0.6:
                x = (x * (0.75 + 0.50 * torch.rand(3, 1, 1))).clamp_(0.0, 1.0)
            # global contrast
            if torch.rand(1) < 0.4:
                m = x.mean()
                x = (m + (x - m) * (0.70 + 0.60 * torch.rand(1))).clamp_(0.0, 1.0)
            # additive Gaussian noise
            if torch.rand(1) < 0.35:
                x = (x + 0.025 * torch.randn_like(x)).clamp_(0.0, 1.0)
            # random erasing (up to 2 patches, filled with channel mean)
            for _ in range(2):
                if torch.rand(1) < 0.25:
                    _, sh, sw = x.shape
                    eh = int(torch.randint(12, max(13, sh // 4), (1,)))
                    ew = int(torch.randint(12, max(13, sw // 4), (1,)))
                    et = int(torch.randint(0, sh - eh, (1,)))
                    el = int(torch.randint(0, sw - ew, (1,)))
                    x[:, et : et + eh, el : el + ew] = x.mean()
        else:
            if H > s and W > s:
                top = (H - s) // 2
                left = (W - s) // 2
                x = x[:, top : top + s, left : left + s]

        if self.downsample_kernel > 1:
            k = self.downsample_kernel
            pool = F.avg_pool2d if self.downsample_mode == "mean" else F.max_pool2d
            xs = x.shape[-1]
            x = pool(x.unsqueeze(0), kernel_size=k, ceil_mode=True)
            x = F.interpolate(x, size=(xs, xs), mode="nearest").squeeze(0)

        x = (x - self.mean) / self.std
        if self.y is not None:
            return x, self.y[idx]
        return x



def mixed_criterion(crit, logits, ya, yb, lam):
    return lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)


# Variety mapping: ACCROC(0), AUBUSSON(1), BAGOU(2), BELEPI(3), BERGAMO(4), BOREGAR(5), EXPERT(6), KALAHARI(7)
# Define variety to number mapping (alphabetical order 0-7)
_variety_to_num = {
    'ACCROC': 0, 'AUBUSSON': 1, 'BAGOU': 2, 'BELEPI': 3,
    'BERGAMO': 4, 'BOREGAR': 5, 'EXPERT': 6, 'KALAHARI': 7
}

_scoop_bacs_by_class = {
    0: [14, 18, 43],  # EL4X-199
    1: [17, 71, 74],  # EL4X-35
    2: [41, 42, 50],  # EL4X-482
    3: [7, 57, 92],   # GQ4X-83
}
_scoop_class_by_bac = {
    bac: class_idx
    for class_idx, bacs in _scoop_bacs_by_class.items()
    for bac in bacs
}

# Load CSV and build mix dictionary
def _load_mix_dict(csv_path=None):
    """
    Load perfomix_mixtures.csv and return dict mapping mix names to variety number arrays.
    Maps mix names (mix01-mix52) to variety numbers (0-7, alphabetical order)

    Defaults to the copy bundled with this package (rgb_grains/data/perfomix_mixtures.csv).
    """
    if csv_path is None:
        csv_path = Path(__file__).parent / "perfomix_mixtures.csv"
    df = pd.read_csv(csv_path, sep='\t')
    mix_dict = {}
    for mix_name, group in df.groupby('mix'):
        ## mix name must be of the for mix07 and not mix7 when there are some 0s.
        mix_name_f = f"mix{int(mix_name[3:]):02d}"
        varieties = group['var'].tolist()
        nums = sorted([_variety_to_num[v] for v in varieties])
        ## we add moth names, for robustness - input filenames ar not super consistently formatted..
        mix_dict[mix_name_f] = np.array(nums)
        mix_dict[mix_name] = np.array(nums)

    mix_dict["unknownMix"] = np.array([0]) 
    return mix_dict


def _equalize_classes(X, y, filenames, restricted_classes, _log_fn):
    '''
    undersampling to have all classes become balanced.
    The code may seem to be overcomplicated, but it's because it handles BOW-style y vectors, i.e entries where several bits are >0.
    '''

    # TODO: trier aussi les filenames

    counts = np.sum(y, axis=0)
    cap_N = np.min(counts) ## max number of samples per class: is chosen as the minimum number of samples per class, over all classes
    increasing_order = np.argsort(counts)
    X_out = []
    y_out = []
    files_out = []
    files = np.array(filenames)

    ## book-keeping:
    current_counts = np.zeros(len(restricted_classes))
    used_indices = set()  # Track indices already added to output
    for c in increasing_order:  ## taking the incresing order helps avoiding inbalanced final dataset.
        class_mask = y[:, c] > 0.0

        class_indices = np.where(class_mask)[0]
        # Remove already used indices
        available_indices = [idx for idx in class_indices if idx not in used_indices]
        if len(available_indices) == 0:
            _log_fn(f"\n\nWARNING:Class {c} already used up entirely ?: no available indices any more.\n\n")
            continue
            
        # Get samples for this class from available indices only
        available_mask = np.zeros(len(X), dtype=bool)
        available_mask[available_indices] = True
        # Take up to cap_N samples for this class
        class_y_values = y[available_mask, c]
        if current_counts[c] >= cap_N:
            _log_fn(f"\n\nWARNING:\nClass {c} already has {current_counts[c]} samples, skipping. \nThis may induce an imbalance dataset ! \n\n")
            continue
        mask = np.cumsum(class_y_values) <= cap_N-current_counts[c] ## key line: cumulative sum to select up to cap_N samples (minus those already in)
        selected_indices = np.where(available_mask)[0][mask]

        X_out.append(X[selected_indices])
        y_out.append(y[selected_indices])
        files_out.append(files[selected_indices])

        # Mark these indices as used
        used_indices.update(selected_indices)
        current_counts += np.sum(y[selected_indices, :], axis=0)
        _log_fn(f"class {c}: adding {len(y[selected_indices])} samples to the data, class counts is now: {np.round(current_counts,2)}")
    X_out = np.concatenate(X_out, axis=0)
    y_out = np.concatenate(y_out, axis=0)
    files_out = np.concatenate(files_out, axis=0)

    ## check uniformity, and possibly remove some samples (ideally the pure ones are easier to find to remove) until the classes are balanced:
    class_sums = y_out.sum(axis=0)
    max_class_sum = class_sums.max()
    min_class_sum = class_sums.min()
    emergency_exit=0
    while max_class_sum - min_class_sum > 10 :
        emergency_exit+=1
        c = int(np.argmax(class_sums)) ## ideally: choose at random between classes with excess number of : "samples"
        max_values = np.max(y_out, axis=1) ## check wether the sample is a true one-hot, for easy removal of samples from just one class
        class_mask = (y_out[:, c] > 0.0 ) & (max_values == 1)
        selected_indices = np.where(class_mask)[0]
        delta = int(max_class_sum - min_class_sum) ## round to aninteger number of samples.
        selected_indices = selected_indices[:delta]
        _log_fn(f"Classes are not balanced, removing {len(selected_indices)} samples (from class {c})")
        if len(selected_indices)==0:
            _log_fn("\n\nWarning: there are no samples to be removed (probably there is no pure stand data to remove from, or not enough, etc..)\nPerfect balancing will probably fail.\n\n")
        X_out = np.delete(X_out, selected_indices, axis=0)
        y_out = np.delete(y_out, selected_indices, axis=0)
        files_out = np.delete(files_out, selected_indices, axis=0)
        class_sums = y_out.sum(axis=0)
        max_class_sum = class_sums.max()
        min_class_sum = class_sums.min()
        _log_fn(f"Classes are now more balanced (hopefully): {class_sums} with min {min_class_sum} and max {max_class_sum}")
        if emergency_exit>5:
            _log_fn("Emergency exit reached.. exiting.")
            break

    return X_out, y_out, files_out


def _label_from_filename_func(filename, NUM_CLASSES, mix_dict, _log_fn, force_one_hot=True, dataset_choice="perfomix"):
    if "SCOOP" in dataset_choice:
        bac_match = re.search(r"bac(\d+)", filename)
        if not bac_match:
            _log_fn(f"Warning: Could not extract bac from {filename}")
            assert False
        bac_number = int(bac_match.group(1))
        if bac_number not in _scoop_class_by_bac:
            _log_fn(f"Warning: Bac {bac_number} not found in SCOOP class mapping")
            assert False
        label_from_filename = _scoop_class_by_bac[bac_number]
        if force_one_hot:
            label_from_filename = F.one_hot(
                torch.tensor(label_from_filename), num_classes=NUM_CLASSES
            ).float()
        return label_from_filename, False

    label_match = re.search(r"var\d{1,2}", filename)
    mix_match = re.search(r"mix\d{1,2}", filename)
    if label_match:
        mixed = False
        label_str = label_match.group(0)[3:]
        label_from_filename = (
            int(label_str) - 1
        )  ## use the [0,1,2....7] convention instead of 1,2,...,8
        if force_one_hot:
            label_from_filename = F.one_hot(
                torch.tensor(label_from_filename), num_classes=NUM_CLASSES
            ).float()
    elif mix_match:
        if mix_match:
            mixed = True
            mix_str = mix_match.group(0)
            if mix_str not in mix_dict:
                _log_fn(f"Warning: Mix string {mix_str} not found in mix_dict")
                mix_str= "unknownMix"
            
            labels = mix_dict[mix_str]
            Nlabels_true = len(labels)
            if force_one_hot:
                # Return a torch.float32 distribution, consistent with the `var` branch
                # which returns F.one_hot(...).float(). Required so the DataLoader can
                # collate batches that mix pure and partial labels, and so downstream
                # losses (soft_ce_weighted) receive a float tensor.
                label_from_filename = torch.zeros(NUM_CLASSES, dtype=torch.float32)
                label_from_filename[labels] = 1.0 / Nlabels_true
            else: 
                label_from_filename = labels[0]  ## taking the first of the list, naively..

    else:
        mixed=False
        _log_fn("DEBUG: 'var' not found and 'mix' not found either... ", filename)
        _log_fn(f"Warning: Could not extract label from {filename}")
        label_from_filename = -1
        if force_one_hot:
            label_from_filename = np.zeros(NUM_CLASSES) ## all zeros ?? But this is forbidden.
        assert False

    return label_from_filename, mixed


def load_datasets_microplot_split(
    dataset_path=None,
    _log_fn=None,
    limit_per_year=None,
    test_ratio=0.5,
    output_dir=None,
    restrict_classes=False,
    restricted_classes=None,
    yearChosen=2020,
    splitting_choice="muPlot-3muTrain-1muTest",
    NUM_CLASSES=8,
    fold_number=0,
    dataset_choice="perfomix",
    testOnMixedOnly=False,
    trainOnMixedAndPure=False,
    equalize_classes=True,
    NsamplesYear2 = None,
    testOnWholePureOnly=False,
    combineMixedAndPureInTest=True,
    clean_data=False,
    min_grain_area=None,
    max_grain_area=None,
):
    """
    takes care of making train/test split based on the micro plot tag.
       grain5323_x34y21-var8_8000_us_2x_2020-12-02T142436_corr.npz

    clean_data / min_grain_area / max_grain_area: TODO 5.5 "by size" cleaning
    (see rgb_grains/data/cleaning.py). Disabled by default to preserve the
    original (unfiltered) behavior; enable with clean_data=True.
    """

    SPLIT_DIR = output_dir / "splits"
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_CSV = SPLIT_DIR / "train.csv"
    TEST_CSV = SPLIT_DIR / "test.csv"
    def read_split_csv(csv_path):
        df = pd.read_csv(csv_path)
        return np.array(df["filepath"]), df

    assert os.path.exists(dataset_path), f"Dataset path not found! Make sure to mount drive correctly to: {dataset_path}"

    mix_dict=None
    if "perfomix" in dataset_choice:
        _log_fn("Using perfomix dataset")
        files = glob.glob(f"{dataset_path}/perfomix*_var1-8_processed/*.npz")
        if limit_per_year is not None:
            np.random.shuffle(files)
            files = files[:limit_per_year*8]
        _log_fn(f"Found {len(files)} total NPZ files (Pure Stand).")
        mixedStands_files = glob.glob(f"{dataset_path}/perfomix_*_mix_processed/*.npz")
        if limit_per_year is not None:
            np.random.shuffle(mixedStands_files) 
            mixedStands_files = mixedStands_files[:limit_per_year*8]
        _log_fn(f"Found {len(mixedStands_files)} total NPZ files (Mixed Stand).")
        mix_dict = _load_mix_dict()

        files = files + mixedStands_files
        del mixedStands_files
    elif "SCOOP" in dataset_choice :
        _log_fn("Using SCOOP dataset")
        files = glob.glob(f"{dataset_path}/SCOOP-R2022-bacs_processed/*.npz")
        if limit_per_year is not None:
            files_by_bac = {}
            for f in files:
                bac_match = re.search(r"bac(\d+)", os.path.basename(f))
                if bac_match:
                    files_by_bac.setdefault(bac_match.group(0), []).append(f)
            files = []
            for bac_files in files_by_bac.values():
                np.random.shuffle(bac_files)
                files.extend(bac_files[:limit_per_year])
        _log_fn(f"Found {len(files)} total NPZ files (Pure Stand, SCOOP).")
    else:
        _log_fn("\n\nDataset must be iether perfomix or SCOOP.\n\n")
        raise SystemExit
    # SCOOP-R2022-bacs_processed
    # perfomix_2019-2020_IE_HSI_var1-8_processed
    # perfomix_2020-2021_IE_HSI_var1-8_processed
    # perfomix_2019-2020_IE_HSI_mix_processed
    # perfomix_2020-2021_IE_HSI_mix_processed

    if clean_data:
        files = filter_excluded_files(
            files,
            dataset_choice=dataset_choice,
            min_area=min_grain_area,
            max_area=max_grain_area,
            enabled=True,
            log_fn=_log_fn,
        )

    rows = []
    rows_mixedStands = []
    for f in files:
        filename = os.path.splitext(os.path.basename(f))[0]
        # extract the microplot name (eg. _x40y20-var) from the filename grain7820_x40y20-var6_8000_us_2x_2020-12-02T134036_corr.npz:
        if "SCOOP" in dataset_choice:
            microplotsearch = re.search(r"bac\d+", filename)
        else:
            microplotsearch = re.search(r"x\d{2}y\d{2}", filename)
        if microplotsearch:
            microplotname = microplotsearch.group(0)
        else:
            _log_fn(f"Warning: Could not extract microplot name from {f}")
            microplotname = "unknown"


        label_from_filename, mixed = _label_from_filename_func(
            filename,
            NUM_CLASSES,
            mix_dict,
            _log_fn,
            force_one_hot=False,
            dataset_choice=dataset_choice,
        )

        if mixed == False:
            # Corrected regex to capture the year from filenames like _YYYY-MM-DDT...
            year_match = re.search(r"_(\d{4})-\d{2}-\d{2}T", filename)
            if year_match:
                year_str = year_match.group(1)
                # Assign to y1_files (2020) or y2_files (2021) based on year_str
                if not ((year_str == "2020") or (year_str == "2021") or ("SCOOP" in dataset_choice and year_str == "2022")):
                    _log_fn(f"Warning: year not matching in {f}")
                    year_str = "unknown"
            else:
                _log_fn(f"Warning: Could not extract year from {f}")
                year_str = "unknown"
        else: 
            year_str = "2026"

        if mixed==False:
            rows.append(
                {
                    "filename": filename,
                    "filepath": f,
                    "label": label_from_filename,
                    "microplot": microplotname,
                    "year": int(year_str),
                    "mixed": mixed
                }
            )
        else: 
            rows_mixedStands.append(
                {
                    "filename": filename,
                    "filepath": f,
                    "label": label_from_filename,
                    "microplot": microplotname,
                    "year": int(year_str),
                    "mixed": mixed
                }
            )
        

    df = pd.DataFrame(rows, columns=["filename", "filepath", "label", "microplot", "year", "mixed"])
    classes = np.sort(df["label"].unique())
    if restrict_classes == True:
        restricted_classes = restricted_classes
    else:
        restricted_classes = classes

    def _log_fn_split_stats(df, restricted_classes):
        for c in restricted_classes:
            ## TODO: _log_fn this to the logger instead of stdout.
            _log_fn(
                f"  Class {c}: {len(df[(df['label'] == c) & (df['split'] == 'train')])} train samples, from microplots \
                {np.unique(df[(df['label'] == c) & (df['split'] == 'train')]['microplot'].values)}, and from years: \
                {np.unique(df[(df['label'] == c) & (df['split'] == 'train')]['year'].values)}"
            )
            _log_fn(
                f"           {len(df[(df['label'] == c) & (df['split'] == 'test')])}  test samples, from microplots \
                {np.unique(df[(df['label'] == c) & (df['split'] == 'test')]['microplot'].values)}, and from years: \
                {np.unique(df[(df['label'] == c) & (df['split'] == 'test')]['year'].values)}"
            )

    if splitting_choice == "muPlot-2muTrain-2muTest":
        ## here I ahd forgotten to specify year explictly. Probability of random mathc between names of microp plots, for the same vairetiy over two years, is however minimal
        _log_fn(f"[*] Splitting : {splitting_choice}")
        df = pd.DataFrame(rows)
        if fold_number <2:
            for c in restricted_classes:
                ## extract the piece of the df that has this label and look at microplot names:
                for year in [2020, 2021]:
                    df_select = df[(df["label"] == c) & (df["year"] == year)]
                    microplot_names = np.sort(df_select["microplot"].unique())
                    assert len(microplot_names) > fold_number, f"Error: fold number {fold_number} is too large for microplot names {microplot_names}. Aborting this useless trial"
                    ## assign train tag to all but one microplot, and test tag to the last one (there are typically only two)
                    for m in microplot_names:  ## start by filling all of them, for this class
                        df.loc[(df["label"] == c) & (df["microplot"] == m) & (df["year"] == year), "split"] = "train"
                    df.loc[
                        (df["label"] == c) & (df["microplot"] == microplot_names[fold_number]) & (df["year"] == year), "split"
                    ] = "test"
        else:
            fold_number -= 2
            for c in restricted_classes:
                ## extract the piece of the df that has this label and look at microplot names:
                for year in [2020, 2021]:
                    df_select = df[(df["label"] == c) & (df["year"] == year)]
                    microplot_names = np.sort(df_select["microplot"].unique())
                    if year==2020:
                        microplot_names = microplot_names[::-1]
                    assert len(microplot_names) > fold_number, f"Error: fold number {fold_number} is too large for microplot names {microplot_names}. Aborting this useless trial"
                    ## assign train tag to all but one microplot, and test tag to the last one (there are typically only two)
                    for m in microplot_names:  ## start by filling all of them, for this class
                        df.loc[(df["label"] == c) & (df["microplot"] == m) & (df["year"] == year), "split"] = "train"
                    df.loc[
                        (df["label"] == c) & (df["microplot"] == microplot_names[fold_number]) & (df["year"] == year), "split"
                    ] = "test"

        _log_fn_split_stats(df, restricted_classes)
        ## 4 folds give:
        #   Class 0: 1748 train samples, from microplots                 ['x30y21' 'x73y14'], and from years:                 [2020 2021]
        #            1714  test samples, from microplots                 ['x33y24' 'x75y20'], and from years:                 [2020 2021]
        #   Class 0: 1783 train samples, from microplots                 ['x30y21' 'x75y20'], and from years:                 [2020 2021]
        #            1679  test samples, from microplots                 ['x33y24' 'x73y14'], and from years:                 [2020 2021]
        #   Class 0: 1679 train samples, from microplots                 ['x33y24' 'x73y14'], and from years:                 [2020 2021]
        #            1783  test samples, from microplots                 ['x30y21' 'x75y20'], and from years:                 [2020 2021]
        #   Class 0: 1714 train samples, from microplots                 ['x33y24' 'x75y20'], and from years:                 [2020 2021]
        #            1748  test samples, from microplots                 ['x30y21' 'x73y14'], and from years:                 [2020 2021]

    if splitting_choice == "muPlot-1muTrain-1muTest-crossYears":
        ## in this case, for a class we use a year as train and test, and for another class we use the other year as train and test
        _log_fn(f"[*] Splitting : {splitting_choice}")
        df = pd.DataFrame(rows)
        years_class_list=[2020, 2020, 2021, 2021, 2020, 2020, 2020, 2021]
        for c in restricted_classes:
            ## extract the piece of the df that has this label and look at microplot names:
            year = years_class_list[c]
            df_select = df[(df["label"] == c) & (df["year"] == year)]
            microplot_names = np.sort(df_select["microplot"].unique())
            assert len(microplot_names) > fold_number, f"Error: fold number {fold_number} is too large for microplot names {microplot_names}. Aborting this useless trial"
            ## assign train tag to all but one microplot, and test tag to the last one (there are typically only two)
            for m in microplot_names:  ## start by filling all of them, for this class
                df.loc[(df["label"] == c) & (df["microplot"] == m) & (df["year"] == year), "split"] = "train"
            df.loc[
                (df["label"] == c) & (df["microplot"] == microplot_names[fold_number]) & (df["year"] == year), "split"
            ] = "test"
        _log_fn_split_stats(df, restricted_classes)


    if splitting_choice == "muPlot-3muTrain-1muTest":
        _log_fn(
            f"[*] Splitting : {splitting_choice}:  by microplot, 3muTrain-1muTest, using a single microplot of Y2 as test, 3 microplots of as train (Y1,Y1,Y2)"
        )
        df = pd.DataFrame(rows)
        ## take all microplots of Y1 and most of Y2 for training, and the last microplot of Y2 for testing:
        for c in restricted_classes:
            ## extract the piece of the df that has this label and look at microplot names:
            df_c = df[df["label"] == c]
            microplot_names = np.sort(df_c["microplot"].unique())
            if c != 4:   
                assert len(microplot_names) > fold_number, f"Error: fold number {fold_number} is too large for microplot names {microplot_names}. Aborting this useless trial"
            ## assign train tag to all but one microplot, and test tag to the last one (there are typically only two)
            for m in microplot_names:  ## start by filling all of them, for this class
                df.loc[(df["label"] == c) & (df["microplot"] == m), "split"] = "train"
            
            if c != 4: 
                df.loc[
                    (df["label"] == c) & (df["microplot"] == microplot_names[fold_number]), "split"
                ] = "test"
            else:
                if fold_number < len(microplot_names):
                    df.loc[
                        (df["label"] == c) & (df["microplot"] == microplot_names[fold_number]), "split"
                    ] = "test"
                else:
                    _log_fn(f"[*] Warning: fold number {fold_number} is too large for microplot names {microplot_names} for class {c}. Skipping this class for test (This is because of the cursed microplot).")
        _log_fn_split_stats(df, restricted_classes)




    if splitting_choice == "muPlot-2muTrain-2muTest-yearGeneralization_year1only":
        _log_fn(
            f"[*] Splitting : {splitting_choice}:  by microplot, 2muTrain-2muTest-yearGeneralization_year1only, using 2 microplots of a year as train, and 2 of the other year as Test"
        )
        df = pd.DataFrame(rows)
        for c in restricted_classes:
            df_c = df[(df["label"] == c) & (df["year"] == yearChosen)] ## the training year is the chosen year (1 microplot)
            microplot_names = np.sort(df_c["microplot"].unique())
            assert len(microplot_names) > fold_number, f"Error: fold number {fold_number} is too large for microplot names {microplot_names}. Aborting this useless trial"
            df.loc[
                (df["label"] == c) & (df["microplot"] == microplot_names[fold_number]), "split"
            ] = "train"
        for c in restricted_classes:
            df_c = df[(df["label"] == c) & (df["year"] != yearChosen)]  ## the other year is for test (1 microplot)
            microplot_names = np.sort(df_c["microplot"].unique())
            assert len(microplot_names) > fold_number, f"Error: fold number {fold_number} is too large for microplot names {microplot_names}. Aborting this useless trial"
            df.loc[
                (df["label"] == c) & (df["microplot"] == microplot_names[fold_number]), "split"
            ] = "test"
        _log_fn_split_stats(df, restricted_classes)


    if splitting_choice == "muPlots_mvblNY2":
        _log_fn(            f"[*] Splitting : {splitting_choice}:  by microplot, taking 100% of year {yearChosen} for training, and a variable number of images from 1 microplot of the other year for training as well, and the last (of that scond year) microplot for testing"   )
        ## yearChosen: all goes to train, up to a budget NsamplesYear1
        df1 = pd.DataFrame(rows)
        NsamplesYear1 = 3300
        df1 = pd.DataFrame(rows)
        for c in restricted_classes:
            ## no micripolot distinction here:
            indices_to_add = df1[(df1["label"] == c) & (df1["year"] == yearChosen)].index
            _log_fn(f"We limit the number of instances per class for year 1, to NsamplesYear1={NsamplesYear1}")
            _log_fn(f"[*] Max size of sub-sample for class {c}, for year {yearChosen} : {len(indices_to_add)}")
            indices_to_add = indices_to_add[:NsamplesYear1]
            df1.loc[indices_to_add, "split"] = "train"

        ## the other year: some goes to train, some unused, some goes to test.
        df = pd.DataFrame(rows)
        df = df[df["year"] != yearChosen]
        for c in restricted_classes:
            ## extract the piece of the df that has this label and look at microplot names:
            df_c = df[df["label"] == c]
            microplot_names = np.sort(df_c["microplot"].unique())
            assert len(microplot_names) > fold_number, f"Error: fold number {fold_number} is too large for microplot names {microplot_names}. Aborting this useless trial"
            ## assign train tag to all but one microplot, and test tag to the last one (there are typically only two)
            assert fold_number <=1, "the logic of this loop is such that there should be exactly 2 microplots per class per year, fold 0 and fold 1."
            mutrain = microplot_names[fold_number]
            ## sub-sample NsamplesYear2 samples from the available ones to be actually train, and the rest to not be used (implictly):
            indices_to_add = df[(df["label"] == c) & (df["microplot"] == mutrain)].sample(n=NsamplesYear2, replace=False).index
            df.loc[indices_to_add, "split"] = "train"
            ## display the max size of that sub-sample:
            max_subsample_size = df.loc[
                (df["label"] == c) & (df["microplot"] == mutrain), "filepath"
            ].count()
            _log_fn(f"[*] Max size of sub-sample for class {c}, for the other year, and microplot {mutrain}: {max_subsample_size}")
            mutest = microplot_names[1-fold_number]
            df.loc[(df["label"] == c) & (df["microplot"] == mutest), "split"] = "test"

        df = pd.concat([df1, df])

        _log_fn_split_stats(df, restricted_classes)


    # 87% test bal acc:
    if splitting_choice == "muPlot_year1only":
        _log_fn(            f"[*] Splitting : {splitting_choice}:  by microplot, taking only year {yearChosen}"   )
        df = pd.DataFrame(rows)
        df = df[df["year"] == yearChosen]
        assert len(df) > 0, (
            f"No samples found for yearChosen={yearChosen} (dataset_choice={dataset_choice}). "
            f"Years present in this dataset: {sorted(pd.DataFrame(rows)['year'].unique())}."
        )
        for c in restricted_classes:
            ## extract the piece of the df that has this label and look at microplot names:
            df_c = df[df["label"] == c]
            microplot_names = np.sort(df_c["microplot"].unique())
            assert len(microplot_names) > fold_number, f"Error: fold number {fold_number} is too large for microplot names {microplot_names}. Aborting this useless trial"
            ## assign train tag to all but one microplot, and test tag to the last one (there are typically only two)
            for m in microplot_names:  ## start by filling all of them, for this class
                df.loc[(df["label"] == c) & (df["microplot"] == m), "split"] = "train"
            df.loc[
                (df["label"] == c) & (df["microplot"] == microplot_names[fold_number]), "split"
            ] = "test"
        _log_fn_split_stats(df, restricted_classes)

    if splitting_choice == "bacs_2train_1test":
        _log_fn(
            f"[*] Splitting : {splitting_choice}: SCOOP/BACS, 2 bacs train and 1 bac test per class, fold={fold_number}"
        )
        assert "SCOOP" in dataset_choice, "bacs_2train_1test is only valid with dataset_choice='SCOOP'"
        assert 0 <= fold_number < 3, "SCOOP/BACS has exactly 3 folds: 0, 1, 2"
        df = pd.DataFrame(rows)
        for c in restricted_classes:
            df_c = df[df["label"] == c]
            bac_names = np.sort(df_c["microplot"].unique())
            expected_bacs = [f"bac{bac}" for bac in _scoop_bacs_by_class[int(c)]]
            missing_bacs = sorted(set(expected_bacs) - set(bac_names))
            if missing_bacs:
                _log_fn(f"  Warning: class {c} is missing expected bacs: {missing_bacs}")
            assert len(bac_names) >= 2, f"Expected at least 2 bacs for class {c}, got {bac_names}"
            for bac_name in bac_names:
                df.loc[(df["label"] == c) & (df["microplot"] == bac_name), "split"] = "train"
            test_bac = bac_names[fold_number % len(bac_names)]
            df.loc[(df["label"] == c) & (df["microplot"] == test_bac), "split"] = "test"
            train_bacs = [str(bac_name) for bac_name in bac_names[bac_names != test_bac]]
            _log_fn(f"  Class {c}: test bac={test_bac}, train bacs={train_bacs}")
        _log_fn_split_stats(df, restricted_classes)

    # 92% test bal acc:
    if splitting_choice == "random_year1only":
        np.random.seed(fold_number)
        _log_fn(f"[*] Splitting : {splitting_choice}:  randomly, taking only one year={yearChosen} (i.e. 2 muPlots)")
        df = pd.DataFrame(rows)
        ## take only one year, split at random among microplots.
        df = df[df["year"] == yearChosen]
        assert len(df) > 0, (
            f"No samples found for yearChosen={yearChosen} (dataset_choice={dataset_choice}). "
            f"Years present in this dataset: {sorted(pd.DataFrame(rows)['year'].unique())}."
        )
        for c in restricted_classes:
            df_c = df[df["label"] == c]
            number_available = len(df_c)
            number_test = int(number_available * test_ratio)
            test_indices = np.random.choice(
                number_available, number_test, replace=False
            )
            train_indices = np.setdiff1d(np.arange(number_available), test_indices)
            df.loc[df_c.index[train_indices], "split"] = "train"
            df.loc[df_c.index[test_indices], "split"] = "test"
        _log_fn_split_stats(df, restricted_classes)


    if splitting_choice == "random_1muPlot":
        _log_fn(f"[*] Splitting : {splitting_choice}:  randomly, taking only ONE fold={fold_number} as train+test data")
        df = pd.DataFrame(rows)
        for c in restricted_classes:
            df_c = df[df["label"] == c]
            microplot_names = np.sort(df_c["microplot"].unique())
            assert len(microplot_names) > fold_number, f"Error: fold number {fold_number} is too large for microplot names {microplot_names}. Aborting this useless trial"
            df_select = df[ (df["label"] == c) & (df["microplot"] == microplot_names[fold_number]) ]
            number_available = len(df_select)
            number_test = int(number_available * test_ratio)
            test_indices = np.random.choice(
                number_available, number_test, replace=False
            )
            train_indices = np.setdiff1d(np.arange(number_available), test_indices)
            df.loc[df_select.index[train_indices], "split"] = "train"
            df.loc[df_select.index[test_indices], "split"] = "test"
        _log_fn_split_stats(df, restricted_classes)

    if splitting_choice == "random_3muPlot":
        _log_fn(f"[*] Splitting : {splitting_choice}:  randomly, taking THREE fold={fold_number} as train+test data")
        df = pd.DataFrame(rows)
        for c in restricted_classes:
            df_c = df[df["label"] == c]
            microplot_names = np.sort(df_c["microplot"].unique())
            assert len(microplot_names) > fold_number, f"Error: fold number {fold_number} is too large for microplot names {microplot_names}. Aborting this useless trial"
            df_select = df[ (df["label"] == c) & (df["microplot"] != microplot_names[fold_number]) ] ## exclude the fold_number microplot to obtain just 3 others
            assert len(microplot_names) ==4 , "the code logic assumes that there are 4 microplots per class, we exclude one and are left with 3."
            number_available = len(df_select)
            number_test = int(number_available * test_ratio)
            test_indices = np.random.choice(
                number_available, number_test, replace=False
            )
            train_indices = np.setdiff1d(np.arange(number_available), test_indices)
            df.loc[df_select.index[train_indices], "split"] = "train"
            df.loc[df_select.index[test_indices], "split"] = "test"
        _log_fn_split_stats(df, restricted_classes)

    if splitting_choice == "random_4muPlot":
        np.random.seed(fold_number)
        _log_fn(f"[*] Splitting : {splitting_choice}:  randomly, taking FOUR fold={fold_number} as train+test data")
        df = pd.DataFrame(rows)
        for c in restricted_classes:
            df_c = df[df["label"] == c]
            microplot_names = np.sort(df_c["microplot"].unique())
            df_select = df[ (df["label"] == c) ] ## exclude nothing
            number_available = len(df_select)
            number_test = int(number_available * test_ratio)
            test_indices = np.random.choice(
                number_available, number_test, replace=False
            )
            train_indices = np.setdiff1d(np.arange(number_available), test_indices)
            df.loc[df_select.index[train_indices], "split"] = "train"
            df.loc[df_select.index[test_indices], "split"] = "test"
        _log_fn_split_stats(df, restricted_classes)

    if splitting_choice == "inferenceMode_4muTrain-0muTest":
        _log_fn(f"[*] Splitting : {splitting_choice}: . This is meant to perform inference, on production (unlabeld) data, or for mixed Stand data (uncertain labels)")
        df = pd.DataFrame(rows)
        df["split"] = "train"
        assert fold_number == 0 , " there is no notion of fold number for this mode: don't waste resources at re-computing the same thing multiple times. Aborting."
        _log_fn_split_stats(df, restricted_classes)
        testOnMixedOnly=True  ## convenience flag, to read the predictions slightly more easily 


    df_mixedStand = pd.DataFrame(rows_mixedStands)
    df_pure = pd.DataFrame(rows)

    #####################
    ## very special case, where df is not the pure data, for once:
    if splitting_choice == "trainOnMixedOnly":
        _log_fn(f"[*] Splitting : {splitting_choice}: . This is meant to train on mixed stands only")
        df = pd.DataFrame(rows_mixedStands)
        df["split"] = "train"
        assert fold_number == 0 , " there is no notion of fold number for this mode: don't waste resources at re-computing the same thing multiple times. Aborting."
        _log_fn_split_stats(df, restricted_classes)

    #####################################################################
    ## df, coming from rows (pureStand data) is now split in train and test.
  
    ## split the df into train and test
    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]
    if trainOnMixedAndPure: ## flag that allows combining a splitting choice for df_pure and the **whole** df_mixed datasets.
        assert splitting_choice != "trainOnMixedOnly", "Incompatible choice. trainOnMixedAndPure should not be used with trainOnMixedOnly."
        assert splitting_choice != "inferenceMode_4muTrain-0muTest", "Incompatible choice. trainOnMixedAndPure should not be used with inferenceMode_4muTrain-0muTest: otherwise you have no test data left to test on... (and it's not the spirit of the Inference Mode)"
        ## we send all of the mixed stand data into the train set (!)
        df_mixedStand["split"] = "train" ## formally useless, but kept for consistency and debugging.
        df_train = pd.concat([df_train, df_mixedStand])
        ## remark: df_mixedStand will again be used as test set.. but it kind of makes sense, still, because we don't have the true label anyway there.
    
    if testOnMixedOnly : # splitting_choice == "inferenceMode_4muTrain-0muTest": ## TODO : extend this option to other modes too.
        assert trainOnMixedAndPure==False, "Incompatible choice. testOnMixedOnly should not be used with trainOnMixedAndPure: the mixed data is either in train, or in test, not in both."
        df_test = df_mixedStand
        ## remark: this mode is mainly for "inferenceMode_4muTrain-0muTest" or out of laziness to separate the mixed and pure predictions.. but it's ok because I managed to do it cleanly
    elif testOnWholePureOnly:
        ## ignore the mixed stand data (for test at least). This is mostly for convenince of reading the predicitons because we can separate the predictions after they are saved, so one may as well always compute them for everything.
        assert splitting_choice == "trainOnMixedOnly", "this is the only logical combination"
        df_test = df_pure

    elif combineMixedAndPureInTest:
        if splitting_choice != "trainOnMixedOnly":
            df_test = pd.concat([df_test, df_mixedStand])
        else:
            _log_fn("We are training only on the mixed data. We will test on ALL of the pure, and ALL of the mixed (for that one it's kind of ok to train and test since true labels are not known...)")
            df_test = pd.concat([df_pure, df_mixedStand])
            
    df_train.to_csv(TRAIN_CSV, index=False)
    df_test.to_csv(TEST_CSV, index=False)
    _log_fn(f"  Saved {len(df_train)} rows → {TRAIN_CSV}")
    _log_fn(f"  Saved {len(df_test)} rows → {TEST_CSV}")
    ## we then load, to make sure loading is consistent 
    train_files, df_train = read_split_csv(TRAIN_CSV)
    test_files, df_test = read_split_csv(TEST_CSV)


    ## remark: this is not fuilly consistent with what is below...
    
    _log_fn("\nTraining data:")
    df_train["split"] = "train"
    _log_fn_split_stats(df_train, restricted_classes)
    _log_fn("\nTest data:")
    df_test["split"] = "test"
    _log_fn_split_stats(df_test, restricted_classes)

    # ── Load arrays from file lists ──
    def load_bunch(file_list):
        X = np.zeros((len(file_list), 252, 252, 3), dtype=np.int16)
        Y = np.zeros((len(file_list), NUM_CLASSES), dtype=np.float32) ## partial labels incurr floats in the targets "one hot" (Bag Of Word really)
        for idx, f in enumerate(file_list):
            filename = os.path.splitext(os.path.basename(f))[0] ## avoid to read the folder name, which contains tags like mix* or var*
            data = np.load(f)
            X[idx] = data["x"]
            Y[idx], _ = _label_from_filename_func(
                filename,
                NUM_CLASSES,
                mix_dict,
                _log_fn,
                force_one_hot=True,
                dataset_choice=dataset_choice,
            )
        return X, Y

    X_y1_tr, Y_y1_tr = load_bunch(train_files)
    _log_fn(f"Loaded Train data... shape: X={X_y1_tr.shape}, y={Y_y1_tr.shape}")
    X_y1_te, Y_y1_te = load_bunch(test_files)
    _log_fn(f"Loading Test data... shape: X={X_y1_te.shape}, y={Y_y1_te.shape}")



    # Apply limit_per_year for fast debugging
    if limit_per_year is not None and limit_per_year > 0 and "SCOOP" not in dataset_choice:
        _log_fn(f"[*] Limiting total samples to {limit_per_year} (for debugging)")
        n_train = min(limit_per_year, len(X_y1_tr))
        n_test  = min(limit_per_year, len(X_y1_te))  # Keep decent size for test set
        indices_train = np.random.choice(len(X_y1_tr), n_train, replace=False)
        indices_test  = np.random.choice(len(X_y1_te), n_test, replace=False)
        X_y1_tr = X_y1_tr[indices_train]
        Y_y1_tr = Y_y1_tr[indices_train]
        train_files = train_files[indices_train]
        X_y1_te = X_y1_te[indices_test]
        Y_y1_te = Y_y1_te[indices_test]
        test_files = test_files[indices_test]
        _log_fn(f"[*] After limiting: Train={len(X_y1_tr)}, Test={len(X_y1_te)}")

    if equalize_classes:
        _log_fn(f"Equalize-classes: {equalize_classes}: we are balancing classes")
        X_y1_tr, Y_y1_tr, train_files = _equalize_classes(X_y1_tr, Y_y1_tr, train_files, restricted_classes, _log_fn) ## only the train needs to be equalized !!
    else:
        _log_fn(f"Equalize-classes: {equalize_classes}: we are NOT balancing classes")

    _log_fn(f"Balance of classes: train: {Y_y1_tr.sum(axis=0)}")
    _log_fn(f"Balance of classes: test: {Y_y1_te.sum(axis=0)}")
    _log_fn(f"Shapes: {X_y1_tr.shape}, {Y_y1_tr.shape}, {X_y1_te.shape}, {Y_y1_te.shape}")

    train_data = {"X": X_y1_tr, "y": Y_y1_tr , "ids": train_files}
    test_data = {"X": X_y1_te, "y": Y_y1_te, "ids": test_files}

    return train_data, test_data
