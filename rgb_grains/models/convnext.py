from __future__ import annotations
import os
import shutil
import math, os, time
import glob
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import json

from rgb_grains.data.dataset import *
from rgb_grains.utils.tools import *
from rgb_grains.viz.plots import plot_training_curve
# ## 2. ConvNeXt-Tiny Model Architecture

PIN_MEMORY = True
NUM_WORKERS = 0


def stratified_split(y, val_ratio, seed):
    '''
    imperfect straified split, in the case of partial labels, but ok, this is an edge case.
    '''
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    tr, va = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = max(1, int(round(len(idx) * val_ratio)))
        va.append(idx[:n])
        tr.append(idx[n:])
    tr = np.concatenate(tr)
    rng.shuffle(tr)
    return tr, np.concatenate(va)


"""
Grain variety classification – ConvNeXt-Tiny (pretrained ImageNet)

No torchvision required. Weights are loaded from:
  1. Local file  convnext_tiny_imagenet.pt  (alongside model.py if available)
  2. Direct urllib download from PyTorch CDN (~109 MB, cached to /tmp)
  3. Random init fallback (lower accuracy)

Training (single phase):
  - All layers trainable from epoch 1
  - Backbone LR 2e-4 | Head LR 1e-3 | 3-epoch linear warm-up | cosine decay
  - Mixup + CutMix (alpha=0.4), label smoothing 0.05, grad-clip 1.0
  - Head dropout 0.20, stochastic depth 0.10
  - SWA: collect from epoch 5, keep up to 10 snapshots, up to 20 epochs (time-guarded)

Inference:
  - Best + SWA state × 4 crops × D4 TTA  =  32 views per model state
"""


# ─────────────────────── ConvNeXt-Tiny (custom, no torchvision) ──────────────



class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = (
            torch.rand(
                (x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device
            )
            + keep
        ).floor_()
        return x * mask / keep


class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / (s + self.eps).sqrt()
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class CNBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, ls_init=1e-6):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.ln = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(4 * dim, dim, 1)
        self.ls = nn.Parameter(ls_init * torch.ones(1, dim, 1, 1))
        self.dp = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        h = self.dw(x)
        h = self.ln(h)
        h = self.pw1(h)
        h = self.act(h)
        h = self.pw2(h)
        return x + self.dp(self.ls * h)


class CNDown(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.ln = LayerNorm2d(ci)
        self.cv = nn.Conv2d(ci, co, 2, stride=2)

    def forward(self, x):
        return self.cv(self.ln(x))


_DIMS = [96, 192, 384, 768]
_DEPTHS = [3, 3, 9, 3]


# ─────────────────── weight loading (no torchvision) ───────────────────

_TV_SI = {1: 0, 3: 1, 5: 2, 7: 3}
_TV_DI = {2: 0, 4: 1, 6: 2}
_TV_SUB = {0: "dw", 2: "ln", 3: "pw1", 5: "pw2"}


def _map_tv(sd):
    """Remap torchvision ConvNeXt-Tiny keys to our model's names."""
    out = {}
    for k, v in sd.items():
        p = k.split(".")
        if p[0] == "classifier":
            if p[1] == "0":
                out["norm." + p[2]] = v
            continue
        if p[0] != "features":
            continue
        fi = int(p[1])
        if fi == 0:
            out["stem." + p[2] + "." + p[3]] = v
        elif fi in _TV_SI:
            si = _TV_SI[fi]
            bi = p[2]
            if p[3] == "layer_scale":
                out["stages.%d.%s.ls" % (si, bi)] = v.unsqueeze(0)
            elif p[3] == "block":
                sub = int(p[4])
                attr = p[5]
                name = _TV_SUB.get(sub)
                if name:
                    vv = v
                    if sub in (3, 5) and attr == "weight" and v.ndim == 2:
                        vv = v.unsqueeze(-1).unsqueeze(-1)
                    out["stages.%d.%s.%s.%s" % (si, bi, name, attr)] = vv
        elif fi in _TV_DI:
            di = _TV_DI[fi]
            pfx = "ln" if int(p[2]) == 0 else "cv"
            out["downs.%d.%s.%s" % (di, pfx, p[3])] = v
    return out


def _download_weights(url, dest):
    """Download file from url to dest with progress, return True on success."""
    import urllib.request

    try:
        print("[*] Downloading weights (~109 MB) …")
        t0 = time.time()

        def _prog(count, block, total):
            if total > 0 and count % 200 == 0:
                pct = min(100, count * block * 100 // total)
                print("    %.0f%%  %.0fs" % (pct, time.time() - t0), end="\r")

        urllib.request.urlretrieve(url, dest, reporthook=_prog)
        print("\n[*] Download complete (%.0fs)" % (time.time() - t0))
        return True
    except Exception as e:
        print("[!] Download failed:", e)
        return False


def _load_pretrained():
    """
    Load ImageNet ConvNeXt-Tiny weights without torchvision:
      1. Local file  (e.g. included alongside model.py in submission zip)
      2. urllib download from PyTorch CDN → cached at /tmp
      3. None  (random init fallback)
    """
    WEIGHT_FILE = "convnext_tiny_imagenet.pt"
    CDN_URL = "https://download.pytorch.org/models/convnext_tiny-983f1562.pth"
    CACHE_PATH = "./models/convnext_tiny_imagenet.pt"

    # 1 – local file search
    sd = "."
    for d in [
        sd,
        os.getcwd(),
        "pretrained_weights",
        "/app/submission",
        "/app/output",
        "/app/program",
        CACHE_PATH,
        "/tmp",
    ]:
        p = os.path.join(d, WEIGHT_FILE)
        if os.path.isfile(p):
            print(
                "[*] Weights (local):", p, "(%d MB)" % (os.path.getsize(p) // 1_000_000)
            )
            return _map_tv(torch.load(p, map_location="cpu", weights_only=True))

    # 2 – cached download
    if os.path.isfile(CACHE_PATH):
        print("[*] Weights (cached):", CACHE_PATH)
        return _map_tv(torch.load(CACHE_PATH, map_location="cpu", weights_only=True))

    # 2b – fresh download
    if _download_weights(CDN_URL, CACHE_PATH):
        try:
            return _map_tv(
                torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
            )
        except Exception as e:
            print("[!] Failed to load downloaded file:", e)

    print("[!] No pretrained weights – training from random init (lower accuracy)")
    return None


# ─────────────────────── training utilities ───────────────────────


def _autocast(en, device_type="cuda"):
    if not en:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type, enabled=en)
    except:
        return torch.cuda.amp.autocast(enabled=en)


def _scaler(en, device_type="cuda"):
    if not en:
        return None
    try:
        return torch.amp.GradScaler(device_type, enabled=en)
    except:
        return torch.cuda.amp.GradScaler(enabled=en)


class nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def mixup(x, y, alpha, nc):
    lam = max(float(np.random.beta(alpha, alpha)), 0.5)
    idx = torch.randperm(x.size(0), device=x.device)
    # y is already one-hot from dataset
    ya = y.float()
    yb = y[idx].float()
    return lam * x + (1 - lam) * x[idx], lam * ya + (1 - lam) * yb


def no_cutmix_neither_mixup(x, y, nc):
    # y is already one-hot from dataset
    return x, y.float()


def cutmix(x, y, alpha, nc):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape
    rh = int(H * math.sqrt(1 - lam))
    rw = int(W * math.sqrt(1 - lam))
    cy = int(torch.randint(H, (1,)))
    cx = int(torch.randint(W, (1,)))
    y1 = max(0, cy - rh // 2)
    y2 = min(H, cy + rh // 2)
    x1 = max(0, cx - rw // 2)
    x2 = min(W, cx + rw // 2)
    xm = x.clone()
    xm[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam2 = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    # y is already one-hot from dataset
    ya = y.float()
    yb = y[idx].float()
    return xm, lam2 * ya + (1 - lam2) * yb


def soft_ce(logits, targets, smooth=0.05):
    nc = logits.size(1)
    t = targets * (1 - smooth) + smooth / nc  ## label smoothing
    return -(t * F.log_softmax(logits, 1)).sum(1).mean()


def soft_ce_weighted(logits, targets, weights, smooth=0.05):
    """Weighted cross-entropy loss with per-class weights (inverse frequency).

    `targets` may be:
      - one-hot                       e.g. [0, 1, 0, 0, 0, 0]
      - partial-label distributions   e.g. [0.5, 0.5, 0, 0, 0, 0]
    The only requirement is that targets are non-negative and sum to 1 along dim=1.
    """
    nc = logits.size(1)
    # targets = targets.float()
    # # Renormalize defensively in case caller passes targets that don't quite sum to 1
    # s = targets.sum(1, keepdim=True).clamp_min(1e-8)
    # targets = targets / s
    # # Label smoothing that preserves soft-label semantics (still sums to 1)
    t = targets * (1 - smooth) + smooth / nc
    ce = -(t * F.log_softmax(logits, 1)).sum(1)
    if weights is not None:
        # Per-sample weight = expected class weight under the (soft) target distribution
        w = (targets * weights).sum(1)
        return (ce * w).mean()
    return ce.mean()


def avg_states(sds):
    out = {}
    for k in sds[0]:
        out[k] = (sum(s[k].float() for s in sds) / len(sds)).to(sds[0][k].dtype)
    return out


# @torch.no_grad()
# def val_acc(net, ld, device, amp, device_type="cuda"):
#     net.eval()
#     c = t = 0
#     for xb, yb in ld:
#         xb = xb.to(device, non_blocking=True)
#         yb = yb.to(device, non_blocking=True)
#         with _autocast(amp, device_type):
#             preds = net(xb).argmax(1)
#             if yb.ndim > 1:
#                 support = yb > 0
#                 c += support.gather(1, preds.unsqueeze(1)).sum().item()
#             else:
#                 c += (preds == yb).sum().item()
#         t += xb.size(0)
#     return c / max(t, 1)


@torch.no_grad()
def val_bal_acc_per_class(net, ld, device, amp, nc, device_type="cuda", strict=False):
    """Compute balanced accuracy and per-class recall."""
    net.eval()
    correct_per_class = np.zeros(nc)
    total_per_class = np.zeros(nc)
    # print(f"DEBUG: nc={nc}")
    for xb, yb in ld:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        yb_int = yb.argmax(1)
        with _autocast(amp, device_type):
            preds = net(xb).argmax(1)
        for c in np.arange(nc):
            mask = yb_int == c
            if yb.ndim > 1 and not strict:
                # yb is one-hot or soft label distribution [batch_size, nc]
                support = yb > 0  # boolean mask of allowed classes
                # For each sample in this class, check if prediction is in its support
                correct_per_class[c] += support[mask, preds[mask]].sum().item()
            else: ## strict case:
                # yb is already integer labels
                correct_per_class[c] += (preds[mask] == yb[mask]).sum().item()
            total_per_class[c] += mask.sum().item()
    recall_per_class = correct_per_class[total_per_class>0]/ total_per_class[total_per_class>0]
    bal_acc = recall_per_class.mean() if total_per_class.sum() > 0 else 0.0
    return bal_acc, recall_per_class


# @torch.no_grad()
# def val_bal_acc(net, ld, device, amp):
#     ## make this into balanced accuracy instead of accuracy:

#     net.eval(); c = t = 0
#     for xb, yb in ld:
#         xb = xb.to(device, non_blocking=True)
#         yb = yb.to(device, non_blocking=True)
#         with _autocast(amp):
#             c += (net(xb).argmax(1) == yb).sum().item()
#         t += xb.size(0)
#     return c / max(t, 1)


def _param_groups(net, lr_bb, lr_hd, wd):
    """Two-group split: backbone (lower LR) vs head (higher LR)."""
    bb, hd = [], []
    for n, p in net.named_parameters():
        if n.startswith("norm") or n.startswith("dropout") or n.startswith("head"):
            hd.append(p)
        else:
            bb.append(p)
    return [
        {"params": bb, "lr": lr_bb, "weight_decay": wd},
        {"params": hd, "lr": lr_hd, "weight_decay": 0.0},
    ]


def extract_features_from_dataset(model, train_y2, test_y2):

    device_lr = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp_lr = device_lr.type == "cuda"

    # net_lr = ConvNeXtTiny(num_classes=8, drop_path=0.0, head_drop=0.0)
    # net_lr.load_state_dict(torch.load(y1_model_path, map_location='cpu'), strict=False)
    # net_lr = net_lr.to(device_lr)
    model.eval()

    def extract_features(model, X, y, device, use_amp, batch_size=64, augment_train=0):
        """
        augment_train=False: average 32 views -> (N, 768)
        augment_train=True:  32 views as separate samples -> (32*N, 768) + tiled labels
        ConvNeXt: model.forward_features(xb) -> (B, 768)
        """
        TTA_CROPS = [160, 172, 184, 196]
        D4 = [
            (False, 0),
            (False, 1),
            (False, 2),
            (False, 3),
            (True, 0),
            (True, 1),
            (True, 2),
            (True, 3),
        ]
        N_VIEWS = len(TTA_CROPS) * len(D4)  # 32
        all_views = []
        for cs in TTA_CROPS:
            ds = GrainDataset_ConvNeXt(X, y, augment=False, crop_size=cs)
            nc = y.shape[1]
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
            for hflip, rot in D4:
                feats_v = []
                with torch.no_grad():
                    for xb, _ in dl:
                        xb = xb.to(device)
                        if hflip:
                            xb = xb.flip(3)
                        if rot:
                            xb = torch.rot90(xb, rot, [2, 3])
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            h = model.forward_features(xb)  # (B, 768)
                        feats_v.append(h.cpu().float().numpy())
                all_views.append(np.concatenate(feats_v, axis=0))
        if augment_train == 1:
            ## for training: augment data, get more "samples" (32 views per sample)
            # Features are view-major: [view0_all_samples, view1_all_samples, ...].
            # Labels must use the same ordering.
            y_aug = np.tile(y, (N_VIEWS, 1))
            return np.concatenate(all_views, axis=0), y_aug
            
        elif augment_train == 2:
            ## testing: average the input features, although we are before the logits.
            ## we could also concatenate them all and average predictions
            return np.mean(np.stack(all_views, axis=0), axis=0)
        elif augment_train == 3:
            ## concat + predict (later) + average outputs (logits) or just majority vote among the 32
            ## predictions (32 preds per test sample)
            return np.stack(all_views, axis=0)  # , np.tile(y,(N_VIEWS,1))

    # labels are already 0-indexed (0-7)
    y_train_lr = np.asarray(train_y2["y"]).astype(np.int64)
    y_test_lr = np.asarray(test_y2["y"]).astype(np.int64)
    ## TODO: see how partial labels fit with the logitistic regression.. they probably don't. Just preclude from doing that with mixed labels in train, and honestly, it makes sense.

    print(f"  Extracting train features x32 D4 views (n={len(y_train_lr)} -> 32x)...")
    X_train_lr, y_train_lr_aug = extract_features(
        model, train_y2["X"], y_train_lr, device_lr, use_amp_lr, augment_train=1
    )

    print(f"  Extracting test features (averaged 32 views, n={len(y_test_lr)})...")
    X_test_lr = extract_features(
        model, test_y2["X"], y_test_lr, device_lr, use_amp_lr, augment_train=2
    )

    print(
        f"  Extracting test features (**concatenate** 32 views, n={len(y_test_lr)})..."
    )
    X_test_lr_aug = extract_features(
        model, test_y2["X"], y_test_lr, device_lr, use_amp_lr, augment_train=3
    )

    # del model; torch.cuda.empty_cache()
    print(f"  Train features: {X_train_lr.shape}  (32x{len(y_train_lr)} aug samples)")
    print(f"  Test  features: {X_test_lr.shape}")

    return X_train_lr, y_train_lr_aug, X_test_lr, y_test_lr, X_test_lr_aug


# X_train_lr, y_train_lr_aug , X_test_lr, y_test_lr,X_test_lr_aug = extract_features_from_dataset(model, train_y2, test_y2)


def extract_features_train_only(model, train_data, n_views=32):
    """
    Extract features from training data only (no test set).
    
    Optimized for kickstart initialization where test evaluation is not needed.
    
    Parameters
    ----------
    model : nn.Module
        ConvNeXt model with forward_features method.
    train_data : dict
        Training data with keys 'X' (images) and 'y' (labels).
    n_views : int
        Number of augmentation views to use. Default 32 (4 crops x 8 D4 transforms).
        Set to 1 for single-view extraction (32x faster, but may reduce quality).
    
    Returns
    -------
    X_train : ndarray
        Features, shape (n_views * N, 768) if n_views > 1, else (N, 768).
    y_train : ndarray
        One-hot labels, shape (n_views * N, n_classes) if n_views > 1, else (N, n_classes).
    n_views : int
        Actual number of views used (for downstream use).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model.eval()
    
    y_raw = np.asarray(train_data["y"])
    
    # Handle both integer labels (1D) and one-hot/BOW labels (2D)
    if y_raw.ndim == 2:
        # Already one-hot or BOW format: (N, n_classes)
        y_onehot = y_raw.astype(np.float32)
        y_int = y_raw.argmax(axis=1)  # For dataset indexing (integer labels)
        n_classes = y_raw.shape[1]
    else:
        # Integer labels: convert to one-hot
        y_int = y_raw.astype(np.int64)
        n_classes = int(np.max(y_int)) + 1
        y_onehot = np.eye(n_classes, dtype=np.float32)[y_int]
    
    if n_views == 1:
        # Single-view extraction: fast path for speed
        print(f"  Extracting train features (single view, n={len(y_int)})...")
        X_train = _extract_features_single_view(
            model, train_data["X"], y_int, device, use_amp
        )
        return X_train, y_onehot, 1
    else:
        # Multi-view extraction: 32 views (4 crops x 8 D4 transforms)
        print(f"  Extracting train features x32 D4 views (n={len(y_int)} -> 32x)...")
        X_train, y_train_aug = _extract_features_multiview(
            model, train_data["X"], y_int, device, use_amp, augment=True,
            n_classes=n_classes
        )
        return X_train, y_train_aug, 32


def _extract_features_single_view(model, X, y, device, use_amp, batch_size=64, crop_size=184):
    """
    Extract features using a single center crop (no augmentation).
    Much faster than multi-view, suitable when speed is prioritized.
    """
    ds = GrainDataset_ConvNeXt(X, y, augment=False, crop_size=crop_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    feats = []
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                h = model.forward_features(xb)
            feats.append(h.cpu().float().numpy())
    
    return np.concatenate(feats, axis=0)


def _extract_features_multiview(model, X, y, device, use_amp, batch_size=64, augment=True, n_classes=None):
    """
    Extract features using 32 views (4 crop sizes x 8 D4 transforms).
    
    Parameters
    ----------
    y : ndarray
        Integer labels, shape (N,).
    n_classes : int, optional
        Total number of classes. If None, inferred from max(y)+1.
    
    If augment=True: returns (32*N, 768) features with tiled labels.
    If augment=False: returns (N, 768) averaged features.
    """
    TTA_CROPS = [160, 172, 184, 196]
    D4 = [
        (False, 0), (False, 1), (False, 2), (False, 3),
        (True, 0), (True, 1), (True, 2), (True, 3),
    ]
    N_VIEWS = len(TTA_CROPS) * len(D4)  # 32
    
    all_views = []
    for cs in TTA_CROPS:
        ds = GrainDataset_ConvNeXt(X, y, augment=False, crop_size=cs)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        for hflip, rot in D4:
            feats_v = []
            with torch.no_grad():
                for xb, _ in dl:
                    xb = xb.to(device)
                    if hflip:
                        xb = xb.flip(3)
                    if rot:
                        xb = torch.rot90(xb, rot, [2, 3])
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        h = model.forward_features(xb)
                    feats_v.append(h.cpu().float().numpy())
            all_views.append(np.concatenate(feats_v, axis=0))
    
    if augment:
        # Convert y to one-hot and tile for all views
        if n_classes is None:
            n_classes = int(np.max(y)) + 1
        y_onehot = np.eye(n_classes, dtype=np.float32)[y]
        # Features are arranged as: [view0_all_samples, view1_all_samples, ...]
        # So we need to tile (repeat the whole array N_VIEWS times), not repeat each row
        y_aug = np.tile(y_onehot, (N_VIEWS, 1))
        return np.concatenate(all_views, axis=0), y_aug
    else:
        return np.mean(np.stack(all_views, axis=0), axis=0)


class Model_ConvNeXt:
    def __init__(self, config=None, restricted_classes=None, base_dir="."):
        # Load configuration from dict, file path, or default config.json
        if config is None:
            # Default: load from config.json in the same directory as this module
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
            if os.path.isfile(config_path):
                print(f"Using config file found in {config_path}")
                with open(config_path, "r") as f:
                    config = json.load(f)
            else:
                print("Missing config file. Resorting to default, hard-coded values.")
                config = {}
        elif isinstance(config, str):
            # config is a path to a JSON file
            with open(config, "r") as f:
                config = json.load(f)
        # If config is already a dict, use it directly

        # Get output directory from config (default to "expe/"), relative to
        # base_dir (matches the OUTPUT_DIR computed by rgb_grains.train / .pipeline).
        expe_path = config.get("expe", "expe/")
        if not os.path.isabs(expe_path):
            self.expe = os.path.join(base_dir, "expe", expe_path)
        else:
            self.expe = expe_path
        os.makedirs(self.expe, exist_ok=True)
        print(f"[*] Output directory: {self.expe}")

        # Setup logging to file
        log_path = os.path.join(self.expe, "training.log")
        self.log_file = open(log_path, "a", encoding="utf-8")
        print(f"[*] Log file: {log_path}")

        # Helper method for logging to both stdout and file
        def _log(msg):
            print(msg)
            self.log_file.write(msg + "\n")
            self.log_file.flush()

        self._log_fn = _log

        self.restricted_classes = (
            restricted_classes
            if restricted_classes is not None
            else list(range(0, self.nc))
        )
        # Convert to 0-indexed for internal use
        # self.restricted_classes = [c - self.label_off for c in self.restricted_classes]
        self._log_fn(f"[*] Restricted classes (0-indexed): {self.restricted_classes}")

        # Apply configuration values (with defaults for any missing keys)
        self.seed = config.get("seed", 42)
        self.nc = config.get("nc", 8)
        self.label_off = 0  #  config.get("label_off", 0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            print(
                "[!] WARNING: CUDA not available. Running on CPU. Training will be very slow."
            )
        else:
            print("[*] Device:", self.device)

        self.crop = config.get("crop", 176)
        self.bs = config.get("bs", 48)
        self.val_ratio = config.get("val_ratio", 0.08)

        # TODO 5.3 resolution ablation: kernel=1 is a no-op (full resolution).
        self.downsample_kernel = config.get("downsample_kernel", 1)
        self.downsample_mode = config.get("downsample_mode", "mean")
        if self.downsample_kernel > 1:
            self._log_fn(
                f"[*] Resolution ablation: downsample_kernel={self.downsample_kernel} "
                f"mode={self.downsample_mode}"
            )

        self.time_budget = config.get("time_budget", 1020)
        self.predict_t = config.get("predict_t", 120)

        self.lr_bb = config.get("lr_bb", 2e-4)
        self.lr_hd = config.get("lr_hd", 1e-3)
        self.wd = config.get("wd", 0.01)
        self.warmup_ep = config.get("warmup_ep", 3)
        self.max_ep = config.get("max_ep", 20)
        self.alpha = config.get("alpha", 0.4)
        self.dp = config.get("dp", 0.10)
        self.head_drop = config.get("head_drop", 0.20)

        self.swa_start = config.get("swa_start", 5)
        self.swa_keep = config.get("swa_keep", 10)

        self.trainAugmentations = bool(config.get("trainAugmentations", True))
        if self.trainAugmentations == False:
            self._log_fn(
                "\n\nWarning ! Train augmentations turned off. Performance (accuracy) may be greatly reduced.\n\n"
            )
        self.cutMix_mixUp = bool(config.get("cutMix_mixUp", True))

        self.use_pretrained = bool(config.get("pretrained", True))
        init_mode = "pretrained" if self.use_pretrained else "random-init"
        self._log_fn(
            "[*] ConvNeXt-Tiny | %s | single-phase | %d ep | SWA"
            % (init_mode, self.max_ep)
        )
        seed_everything(self.seed)

        self.pt_state = _load_pretrained() if self.use_pretrained else None
        self.pretrained = self.pt_state is not None
        if not self.use_pretrained:
            self._log_fn("[*] Pretrained weights disabled; training from random initialization.")
        self.use_amp = self.device.type == "cuda"
        self.device_type = "cuda" if self.use_amp else "cpu"
        self._log_fn(f"Automatic Mixed Precision (amp): {self.use_amp}")
        self.all_models = []
        self.selected_model_name = None
        self.selected_state_dict = None
        self.selected_val_bal_acc = None
        self.train_bal_acc_32views = None
        self.train_recall_per_class_32views = None
        self.is_fitted = False

        self.net = ConvNeXtTiny(self.nc, self.dp, self.head_drop).to(self.device)

        # LP-FT (Linear-Probe then Fine-Tune) kickstart state.
        # When self.kickstart is True and self.W_raw/self.b_raw are populated,
        # fit() will overwrite the freshly-initialized head with these weights
        # (computed from a logistic regression on backbone features over the
        # training set, see compute_kickstart_head()).
        self.kickstart = False
        self.W_raw = None  # numpy (num_classes, feat_dim)
        self.b_raw = None  # numpy (num_classes,)
        self._backbone_loaded = False  # track whether pt_state has been pushed into self.net

    # ─── LP-FT / kickstart helpers ───────────────────────────────────────────
    def _apply_pretrained_backbone(self):
        """Push self.pt_state into self.net (idempotent).

        Used both by fit() and by compute_kickstart_head() so that backbone
        features can be extracted from the pretrained backbone BEFORE training
        starts. Head is left untouched here.
        """
        if self._backbone_loaded:
            return
        if self.pt_state:
            miss, unex = self.net.load_state_dict(self.pt_state, strict=False)
            self._log_fn(
                "[*] Backbone loaded  miss=%d  unex=%d" % (len(miss), len(unex))
            )
        self._backbone_loaded = True

    def compute_kickstart_head(self, train_data, cache_path=None, recompute=False, n_views=32):
        """Compute (W_raw, b_raw) for the linear head via logistic regression
        on backbone features extracted over the train set (LP-FT kickstart).

        - If ``cache_path`` is given and exists, weights are loaded from disk
          (unless ``recompute=True``).
        - Otherwise we extract features from train_data, fit logistic regression,
          and save the resulting (W_raw, b_raw) to ``cache_path`` if provided.

        Parameters
        ----------
        train_data : dict
            Training data with keys 'X' and 'y'.
        cache_path : str, optional
            Path to cache/load the computed weights.
        recompute : bool, optional
            If True, recompute even if cache exists.
        n_views : int, optional
            Number of augmentation views (default 32). Set to 1 for faster
            single-view extraction (32x speedup but may slightly reduce quality).

        Side effects: sets ``self.W_raw``, ``self.b_raw``, ``self.kickstart=True``.
        ``fit()`` will then inject these into ``self.net.head`` after the head
        has been (re-)initialized.

        NOTE: this requires the pretrained backbone to be loaded; we call
        ``_apply_pretrained_backbone()`` here so the call can happen before fit().
        """
        # local import to avoid a circular import at module top
        from rgb_grains.models.classifier_head import exactFit_classifier_return_W_b

        if cache_path is not None and os.path.isfile(cache_path) and not recompute:
            self._log_fn(f"[*] kickstart: loading W_raw/b_raw from {cache_path}")
            d = np.load(cache_path)
            self.W_raw = d["W_raw"]
            self.b_raw = d["b_raw"]
            self.kickstart = True
            return self.W_raw, self.b_raw

        view_desc = f"{n_views}-view" if n_views > 1 else "single-view"
        self._log_fn(f"[*] kickstart: computing W_raw/b_raw from train set "
                     f"(extract {view_desc} features + logistic regression)")
        self._apply_pretrained_backbone()

        t0 = time.time()
        # extract_features_train_only is defined in this same module
        X_train_lr, y_train_lr_aug, actual_views = \
            extract_features_train_only(self.net, train_data, n_views=n_views)
        self._log_fn(f"[*] kickstart: feature extraction done ({time.time()-t0:.1f}s)")

        t1 = time.time()
        W_raw, b_raw, acc_lr_train = exactFit_classifier_return_W_b(
            X_train_lr, y_train_lr_aug, scaler=False
        ) ## TODO: choose correct scaler=True/False value, or make it confugurable from CLI
        self._log_fn(
            f"[*] kickstart: LR fit done ({time.time()-t1:.1f}s)  "
            f"train_bal_acc={acc_lr_train:.4f}"
        )

        self.W_raw = np.asarray(W_raw, dtype=np.float32)
        self.b_raw = np.asarray(b_raw, dtype=np.float32)
        self.kickstart = True

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            np.savez(cache_path, W_raw=self.W_raw, b_raw=self.b_raw)
            self._log_fn(f"[*] kickstart: cached to {cache_path}")

        return self.W_raw, self.b_raw

    def _inject_kickstart_into_head(self, net):
        """Overwrite net.head.{weight,bias} with self.W_raw / self.b_raw.

        Validates shapes against the head's expected (num_classes, feat_dim).
        """
        if self.W_raw is None or self.b_raw is None:
            raise RuntimeError(
                "kickstart=True but W_raw/b_raw are not set. "
                "Call compute_kickstart_head() first."
            )
        W = torch.as_tensor(self.W_raw, dtype=net.head.weight.dtype)
        b = torch.as_tensor(self.b_raw, dtype=net.head.bias.dtype)
        if W.shape != net.head.weight.shape:
            raise ValueError(
                f"kickstart W_raw shape {tuple(W.shape)} != "
                f"head.weight shape {tuple(net.head.weight.shape)}"
            )
        if b.shape != net.head.bias.shape:
            raise ValueError(
                f"kickstart b_raw shape {tuple(b.shape)} != "
                f"head.bias shape {tuple(net.head.bias.shape)}"
            )
        with torch.no_grad():
            net.head.weight.copy_(W.to(net.head.weight.device))
            net.head.bias.copy_(b.to(net.head.bias.device))
        self._log_fn("[*] kickstart: head initialized from W_raw/b_raw (LP-FT)")

    def set_selected_model(self, name, state_dict, val_bal_acc=None):
        """Set the single model state used by saving and inference."""
        self.selected_model_name = name
        self.selected_state_dict = {
            key: value.detach().cpu().clone() for key, value in state_dict.items()
        }
        self.selected_val_bal_acc = val_bal_acc
        # Retained for compatibility with older reporting code.
        self.all_models = [(name, self.selected_state_dict)]
        self.net.load_state_dict(self.selected_state_dict)
        self.net.to(self.device)
        self.is_fitted = True

    def _train(self, net, trainset_loader, validset_loader, t0):
        self.train_loss = []
        self.train_acc = []
        # self.val_acc = []
        self.val_bal_acc = []
        self.val_recall_per_class = []
        device_name = (
            torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        )
        self._log_fn(f"Training on device: {device_name}")

        budget = self.time_budget - self.predict_t
        deadline = t0 + budget

        opt = torch.optim.AdamW(_param_groups(net, self.lr_bb, self.lr_hd, self.wd))
        spe = len(trainset_loader)
        ws = self.warmup_ep * spe
        ts = self.max_ep * spe

        def lr_fn(step):
            if step < ws:
                return max(1e-3, step / max(ws, 1))
            return max(
                0.01, 0.5 * (1 + math.cos(math.pi * (step - ws) / max(ts - ws, 1)))
            )

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
        scl = _scaler(self.use_amp, self.device_type)

        best_acc = -1.0
        best_sd = None
        best_epoch = None
        swa_snaps = []
        last_ept = 65.0
        arch_t0 = time.time()

        for ep in range(1, self.max_ep + 1):
            te = time.time()
            net.train()
            tot = cor = 0
            rl = 0.0

            for xb, yb in trainset_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with _autocast(self.use_amp, self.device_type):
                    if self.cutMix_mixUp:
                        xm, ym = (
                            mixup(xb, yb, self.alpha, self.nc)
                            if torch.rand(1)
                            < 0.5  ## proba 0.5 for mixup, and 0.5 for cutmix
                            else cutmix(xb, yb, self.alpha, self.nc)
                        )
                    else:  ## cutMix_mixUp disabled
                        xm, ym = no_cutmix_neither_mixup(xb, yb, self.nc)
                    lo = net(xm)
                    loss = soft_ce_weighted(lo, ym, self.class_weights)
                if self.use_amp and scl is not None:
                    scl.scale(loss).backward()
                    scl.unscale_(opt)
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    scl.step(opt)
                    scl.update()
                    sch.step()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                    sch.step()
                bs_ = xb.size(0)
                rl += loss.item() * bs_
                pred = lo.detach().argmax(1)
                ## old version:
                # yb_int = yb.argmax(1)
                # cor += (pred == yb_int).sum().item()

                # yb is a (possibly soft / partial) label distribution [batch_size, nc].
                # Count a prediction as "correct" if argmax(pred) is in the support of yb
                # (i.e. one of the candidate classes). Reduces to standard top-1 for one-hot.
                support = yb > 0
                cor += support.gather(1, pred.unsqueeze(1)).sum().item()
                tot += bs_

            # vacc = val_acc(net, validset_loader, self.device, self.use_amp, self.device_type)
            vbal_acc, recall_per_class = val_bal_acc_per_class(
                net, validset_loader, self.device, self.use_amp, self.nc, self.device_type
            )
            ept = time.time() - te
            last_ept = ept
            elapsed = time.time() - arch_t0
            remaining = deadline - time.time()

            if vbal_acc > best_acc:
                best_acc = vbal_acc
                best_epoch = ep
                best_sd = {k: v.cpu().clone() for k, v in net.state_dict().items()}

            if ep >= self.swa_start:
                swa_snaps.append(
                    {k: v.cpu().clone() for k, v in net.state_dict().items()}
                )
                if len(swa_snaps) > self.swa_keep:
                    swa_snaps.pop(0)

            self.train_loss.append(rl / tot)
            self.train_acc.append(cor / tot)
            # self.val_acc.append(vacc)
            self.val_bal_acc.append(vbal_acc)
            self.val_recall_per_class.append(recall_per_class)

            self._log_fn(
                "  E%02d loss=%.4f tr=%.4f val_bal=%.4f lr_bb=%.2e "
                "t=%.1fs G=%.0fs swa=%d"
                % (
                    ep,
                    rl / tot,
                    cor / tot,
                    vbal_acc,
                    opt.param_groups[0]["lr"],
                    ept,
                    time.time() - t0,
                    len(swa_snaps),
                )
            )

            # Plot training curve every 10 epochs
            if ep % 10 == 0:
                npz_path = os.path.join(self.expe, "training_metrics.npz")
                np.savez(
                    npz_path,
                    train_loss=np.array(self.train_loss),
                    train_acc=np.array(self.train_acc),
                    # val_acc=np.array(self.val_acc),
                    val_bal_acc=np.array(self.val_bal_acc),
                    val_recall_per_class=np.array(self.val_recall_per_class),
                )
                self._log_fn(f"[*] Training metrics saved to: {npz_path}")
                plot_training_curve(
                    npz_path, experiment_short_name="training"
                )
                self._log_fn(f"[*] Training curve saved.")

            if elapsed > budget - last_ept * 1.3:
                self._log_fn("  time-guard")
                break
            if remaining < last_ept * 1.5 + self.predict_t:
                self._log_fn("  time-guard")
                break

        # SWA
        swa_sd = None
        selected_name = f"best_val_epoch_{best_epoch}"
        selected_sd = best_sd
        selected_acc = best_acc
        if len(swa_snaps) >= 2:
            swa_sd = avg_states(swa_snaps)
            net.load_state_dict(swa_sd)
            net.to(self.device)
            # sv = val_acc(net, validset_loader, self.device, self.use_amp, self.device_type)
            vbal_acc, recall_per_class = val_bal_acc_per_class(
                net, validset_loader, self.device, self.use_amp, self.nc, self.device_type
            )
            sv = vbal_acc
            self._log_fn(
                "[*] SWA val=%.4f  best=%.4f  snaps=%d" % (sv, best_acc, len(swa_snaps))
            )
            if sv > best_acc:
                selected_name = "swa"
                selected_sd = swa_sd
                selected_acc = sv

        if selected_sd is None:
            raise RuntimeError("Training completed without producing a model checkpoint")
        self._log_fn(
            f"[*] Selected model: {selected_name}  val_bal={selected_acc:.4f}"
        )
        return selected_name, selected_sd, selected_acc


    def fit(self, train_data):
        ## TODO: add also the contorl over epochs for instance here
        t0 = time.time()
        X = train_data["X"]
        y = np.asarray(train_data["y"])  # Keep as float for partial labels

        # np.savez("ys_for_debug.npz", y=y)
        # print(f"y.shape={y.shape}")
        # y = np.load("ys_for_debug.npz")["y"]
        # print(f"y.shape={y.shape}")

        self._log_fn(
            f"[*] Data  n={X.shape[0]}  shape={X.shape[1:]}  y.shape={y.shape}"
        )
        ## in case of partial labels, we
        y_for_stratification = y.argmax(1)
        for i in range(len(y_for_stratification)):
            if (y[i] > 0).sum() > 1:
                y_for_stratification[i] = np.random.choice(np.where(y[i] > 0)[0])

        tr, va = stratified_split(y_for_stratification, self.val_ratio, self.seed)
        self._log_fn("[*] Train=%d  Val=%d" % (len(tr), len(va)))

        print(f"y.shape={y.shape}")
        # assert False
        unique_classes = np.arange(y.shape[1])
        # unique_classes, train_counts = np.unique(y[tr].argmax(1), return_counts=True)
        train_counts = y[tr].sum(0)
        val_counts = y[va].sum(0)
        # _, val_counts = np.unique(y[va].argmax(1), return_counts=True)
        self._log_fn("[*] Class distribution in train/val split:")
        for c, tc, vc in zip(unique_classes, train_counts, val_counts):
            self._log_fn(f"    Class {c}: train={tc:.1f} ({tc / len(tr) * 100:.0f}%), val={vc:.1f} ({vc / len(va) * 100:.0f}%)")

        # Compute class weights (inverse frequency) for weighted loss
        ## make this become broadcasted to length nc=8, the initial number of classes, not the restrained one,
        ## and assign weight 0 to classes that are not inside resticted_classes
        full_weights = np.zeros(self.nc)
        for i, c in enumerate(unique_classes):
            if c < self.nc and c in self.restricted_classes and train_counts[i] > 0:
                ## we use the cariant where weights are taken as sqrt(inv class frequency) to decrease the effect.
                full_weights[c] = (1.0 / (train_counts[i] / train_counts.sum())) # **0.5
                ## (classes with zero train samples keep weight 0: they never appear in the
                ## training loss, and 1/0 would otherwise poison the whole normalized vector with nan/inf)


        # Normalize so that sum of non-zero weights = nc
        nonzero_mask = full_weights > 0
        nonzero_sum = full_weights[nonzero_mask].sum()
        nonzero_count = nonzero_mask.sum()
        if nonzero_sum > 0:
            full_weights[nonzero_mask] = (
                full_weights[nonzero_mask] / nonzero_sum * nonzero_count
            )
        self.class_weights = torch.tensor(full_weights, dtype=torch.float32).to(
            self.device
        )
        self._log_fn(
            f"[*] Class weights for weighted loss: {self.class_weights.cpu().numpy()}"
        )

        tds = GrainDataset_ConvNeXt(
            X[tr],
            y[tr],
            augment=self.trainAugmentations,
            crop_size=self.crop,
            num_classes=self.nc,
            return_one_hot=True,
            downsample_kernel=self.downsample_kernel,
            downsample_mode=self.downsample_mode,
        )
        vds = GrainDataset_ConvNeXt(
            X[va],
            y[va],
            augment=False,
            crop_size=self.crop,
            num_classes=self.nc,
            return_one_hot=True,
            downsample_kernel=self.downsample_kernel,
            downsample_mode=self.downsample_mode,
        )
        trainset_loader = DataLoader(
            tds,
            self.bs,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            drop_last=len(tds) >= self.bs,
        )
        validset_loader = DataLoader(
            vds,
            self.bs * 2,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        net = self.net
        # Load pretrained backbone (idempotent — may have already been done by
        # compute_kickstart_head() if LP-FT kickstart is enabled).
        self._apply_pretrained_backbone()
        # Default head init: random truncated-normal weights, zero bias.
        nn.init.trunc_normal_(net.head.weight, std=0.02)
        nn.init.zeros_(net.head.bias)
        # LP-FT kickstart: overwrite the freshly-initialized head with the
        # logistic-regression solution computed by compute_kickstart_head().
        if self.kickstart:
            self._inject_kickstart_into_head(net)
        self._log_fn(
            "[*] %.1fM params  pretrained=%s"
            % (sum(p.numel() for p in net.parameters()) / 1e6, self.pretrained)
        )


        ## actual training is launched here: !!
        selected_name, selected_sd, selected_acc = self._train(
            net, trainset_loader, validset_loader, t0
        )
        self.set_selected_model(selected_name, selected_sd, selected_acc)

        ## plot the training curves: train loss etc as function of epochs:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ##using twin axes, left side for loss, right side for accuracies:
        ax1.semilogy(self.train_loss, label="Training Loss", color="tab:blue")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(
            self.train_acc, label="Training Accuracy (not balanced)", color="tab:orange"
        )
        ax2.plot(
            self.val_bal_acc, label="Validation Accuracy (balanced)", color="tab:green"
        )
        ax2.set_ylabel("Accuracy", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        plt.title("Training and Validation Loss")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
        # plt.show()
        plt.savefig(os.path.join(self.expe, "fine_tuning_monitoring.jpg"))
        plt.savefig(os.path.join(self.expe, "fine_tuning_monitoring.pdf"))
        plt.close()

        # Save training metrics to NPZ
        npz_path = os.path.join(self.expe, "training_metrics.npz")
        np.savez(
            npz_path,
            train_loss=np.array(self.train_loss),
            train_acc=np.array(self.train_acc),
            # val_acc=np.array(self.val_acc),
            val_bal_acc=np.array(self.val_bal_acc),
            val_recall_per_class=np.array(self.val_recall_per_class),
        )
        self._log_fn(f"[*] Training metrics saved to: {npz_path}")

        self._log_fn(
            "[*] Fit done  selected=%s  time=%.0fs"
            % (self.selected_model_name, time.time() - t0)
        )

    def predict_logits(self, test_data):
        """Return probabilities averaged over four crops and eight D4 views."""
        if not self.is_fitted:
            raise RuntimeError("fit() first")
        X = test_data["X"]
        n = X.shape[0]
        crops = [160, 172, 184, 196]

        self._log_fn(
            "[*] Predict probabilities  n=%d  selected=%s  crops=%s  D4(8)"
            % (n, self.selected_model_name, crops)
        )

        probs = torch.zeros(n, self.nc)
        if self.selected_state_dict is None:
            raise RuntimeError("No selected model state is available for inference")
        net = self.net
        net.load_state_dict(self.selected_state_dict)
        net.eval()
        for cs in crops:
            ds = GrainDataset_ConvNeXt(
                X,
                y=None,
                augment=False,
                crop_size=cs,
                downsample_kernel=self.downsample_kernel,
                downsample_mode=self.downsample_mode,
            )
            ld = DataLoader(
                ds,
                batch_size=64,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
            )
            off = 0
            with torch.no_grad():
                for xb in ld:
                    xb = xb.to(self.device, non_blocking=True)
                    bs_ = xb.size(0)
                    acc = torch.zeros(bs_, self.nc, device=self.device)
                    for hf in (False, True):
                        for rot in range(4):
                            xa = xb.flip(3) if hf else xb
                            if rot:
                                xa = torch.rot90(xa, rot, [2, 3])
                            with _autocast(self.use_amp, self.device_type):
                                acc += F.softmax(net(xa), dim=1)
                    probs[off : off + bs_] += acc.cpu()
                    off += bs_
        self._log_fn("  %s done" % self.selected_model_name)
        del net
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        ## end of for loop
        probs = probs / (len(crops) * 8)
        return probs.numpy()

    def evaluate_balanced_accuracy_32views(self, data, dataset_name="data"):
        """Evaluate the selected model with the deterministic 32-view protocol."""
        probabilities = self.predict_logits(data)
        predictions = probabilities.argmax(axis=1)
        balanced_accuracy, _, recall_per_class = val_bal_acc_per_class_offline(
            predictions, np.asarray(data["y"]), self.nc, strict=False
        )
        self._log_fn(
            f"[*] Selected-model {dataset_name} balanced accuracy "
            f"(32 views): {balanced_accuracy:.4f}"
        )
        self._log_fn(
            f"[*] Selected-model {dataset_name} recall per class "
            f"(32 views): {recall_per_class}"
        )
        return balanced_accuracy, recall_per_class, probabilities

    def predict(self, test_data):
        probs = self.predict_logits(test_data)
        preds = probs.argmax(1).astype(np.int64) + self.label_off
        return preds


class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=8, drop_path=0.10, head_drop=0.20):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, _DIMS[0], 4, stride=4), LayerNorm2d(_DIMS[0])
        )
        total = sum(_DEPTHS)
        dp = [drop_path * i / max(total - 1, 1) for i in range(total)]
        bi = 0
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        for si in range(4):
            stage = nn.Sequential(
                *[CNBlock(_DIMS[si], dp[bi + j]) for j in range(_DEPTHS[si])]
            )
            self.stages.append(stage)
            bi += _DEPTHS[si]
            if si < 3:
                self.downs.append(CNDown(_DIMS[si], _DIMS[si + 1]))
        self.norm = nn.LayerNorm(_DIMS[-1], eps=1e-6)
        self.dropout = nn.Dropout(p=head_drop)
        self.head = nn.Linear(_DIMS[-1], num_classes)

    def forward_features(self, x):
        x = self.stem(x)
        for i, st in enumerate(self.stages):
            x = st(x)
            if i < 3:
                x = self.downs[i](x)
        return self.norm(x.mean([2, 3]))

    def forward(self, x):
        return self.head(self.dropout(self.forward_features(x)))
