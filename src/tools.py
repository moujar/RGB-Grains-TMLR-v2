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
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
# import numpy as np
# import pandas as pd



def seed_everything(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


# def stratified_split(y, val_ratio, seed):
#     rng = np.random.RandomState(seed); y = np.asarray(y)
#     tr, va = [], []
#     for c in np.unique(y):
#         idx = np.where(y == c)[0]; rng.shuffle(idx)
#         n = max(1, int(round(len(idx) * val_ratio)))
#         va.append(idx[:n]); tr.append(idx[n:])
#     return np.concatenate(tr), np.concatenate(va)

## compare logits_test_data and true_y_test in the spirit of partial labelling:
# for each mix, count how many times the predicted class is the same as ONE OF the true class(es)
# and how many times it is different, i.e. compute the binary accuracy of being inside the expected class(es), and not.
## also plot, for each mix, the 8 expected ratios (the one hot of that mix), and the predicted probabilities, both as conting of predictions and as sum of logits (per class). Ideally it should be close to the "one hot" (several are hot):
def val_bal_acc_per_class_offline(preds, yb, nc, strict=False):
    """Compute balanced accuracy, per-class recall, and per-class precision.
    preds: predicted class indices
    yb: true class indices (one-hot encoded)
    """
    correct_per_class = np.zeros(nc)
    total_per_class = np.zeros(nc)
    yb_int = yb.argmax(1)

    if yb.ndim > 1 and not strict:
        support = yb > 0
        for c in np.arange(nc):
            mask = yb_int == c
            correct_per_class[c] = support[mask, preds[mask]].sum().item()
            total_per_class[c] = mask.sum().item()
        # Precision: for each predicted class, count how many predictions were correct
        correct_pred_per_class = np.zeros(nc)
        for i, pred in enumerate(preds):
            if support[i, pred]:
                correct_pred_per_class[pred] += 1

    else:
        correct_per_class      = np.array([(preds[yb == c] == c).sum().item() for c in range(nc)])
        total_per_class = np.array([(yb == c).sum().item() for c in range(nc)])
        correct_pred_per_class = np.array([(preds[preds == c] == c).sum().item() for c in range(nc)])

    total_per_class[total_per_class==0]+=0.41
    recall_per_class = correct_per_class/total_per_class # correct_per_class[total_per_class>0]/ total_per_class[total_per_class>0]
    bal_acc = recall_per_class.mean() if total_per_class.sum() > 0.42 else 0.0

    pred_per_class = np.array([(preds == c).sum().item() for c in range(nc)])
    precision = correct_pred_per_class / np.where(pred_per_class > 0, pred_per_class, 1)
    precision[pred_per_class == 0] = 0.0
    return bal_acc, precision, recall_per_class



def soft_class_metrics(logits, true_y):
    """
    Fully soft per-class precision, recall, and F1.
    Both the true label and the prediction are kept probabilistic (no argmax).
 
    For each class c:
      - soft_tp[c]  = sum_i  w[i,c] * p[i,c]          (soft true positives)
      - sum_true[c] = sum_i  w[i,c]                    (soft support)
      - sum_pred[c] = sum_i  p[i,c]                    (total predicted mass)
      - Recall[c]    = soft_tp[c] / sum_true[c]
      - Precision[c] = soft_tp[c] / sum_pred[c]
 
    where w[i,c] = true_y[i,c] / |S_i|  (uniform prior over candidate set)
    and   p[i,c] = softmax(logits[i])[c]
 
    Parameters
    ----------
    logits : np.ndarray, shape (N, nc)
    true_y : np.ndarray, shape (N, nc)  — BoW / multi-hot
 
    Returns
    -------
    dict with keys 'precision', 'recall', 'f1', each an array of length nc.
    """
    # # Softmax probabilities
    # shifted = logits - logits.max(axis=1, keepdims=True)   # numerical stability
    # exp_l   = np.exp(shifted)
    # probs   = exp_l / exp_l.sum(axis=1, keepdims=True)     # (N, nc)
 
    ## sum the logits over the support classes and aggregate results such as to cmpute per class recall, precision:
    # logits = logits_test_data[keep_mixed]
    # true_y = true_y_test[keep_mixed]
    # logits = logits_test_data[keep_pure]
    # true_y = true_y_test[keep_pure]
    nc = true_y.shape[1]

    # Uniform weights over candidate sets
    weights  = true_y / true_y.sum(axis=1, keepdims=True)  # (N, nc)
    # weights = true_y
 
    soft_tp  = (weights * logits).sum(axis=0)               # (nc,)
    sum_true = weights.sum(axis=0)                         # (nc,)
    sum_pred = logits.sum(axis=0)                           # (nc,)
 
    precision = np.where(sum_pred > 0, soft_tp / sum_pred, 0.0)
    recall    = np.where(sum_true > 0, soft_tp / sum_true, 0.0)
    f1        = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0
    )
    return precision, recall, f1 

def soft_confusion_matrix(y_pred_test, true_y):
    """
    Compute the soft (expected) confusion matrix for PLL test data,
    using a uniform prior over each instance's candidate label set.

    Parameters
    ----------
    y_pred_test : np.ndarray, shape (N, 1)
        argmax of the logits
    true_y : np.ndarray, shape (N, nc)  — BoW / multi-hot encoding
        true_y[i, k] = 1 if class k is a candidate for instance i, 0 otherwise.
        Rows with a single 1 are fully observed; rows with multiple 1s are partial.

    Returns
    -------
    C_soft : np.ndarray, shape (nc, nc)
        C_soft[k, l] = expected number of true-class-k instances predicted as l,
                       under a uniform prior over each candidate set.
    """
    nc = true_y.shape[1]

    # Uniform weight over candidate set: w[i,k] = true_y[i,k] / |S_i|
    weights = true_y / true_y.sum(axis=1, keepdims=True)   # (N, nc)

    # One-hot encode hard predictions
    # y_pred_test = np.argmax(logits, axis=1)                   # (N,)
    one_hot_pred = np.eye(nc)[y_pred_test]                    # (N, nc)

    # C_soft[k, l] = sum_i  w[i,k] * 1[pred_i == l]
    return weights.T @ one_hot_pred                         # (nc, nc)



def soft_class_metrics_CM(C_soft):
    """
    Derive per-class soft precision, recall, and F1 from the soft confusion matrix.
    Follows the standard per-class definitions applied to the soft counts.

    Returns a dict of arrays of length n_classes.
    """
    tp = np.diag(C_soft)
    col_sum = C_soft.sum(axis=0)   # all instances predicted as class k
    row_sum = C_soft.sum(axis=1)   # all instances whose (soft) true class is k

# LA JE CROIS QUE CEST PAS BIEN NORMALISE...

    precision = np.where(col_sum > 0, tp / col_sum, 0.0)
    recall    = np.where(row_sum > 0, tp / row_sum, 0.0)
    f1        = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0
    )
    return precision, recall, f1


    # precision_pure_soft, recall_pure_soft, f1_pure_soft  = soft_class_metrics_CM(cm_pure)
    # print("from CM: soft prec, recall, f1\n", precision_pure_soft, recall_pure_soft, f1_pure_soft)

    # precision_mixed_soft, recall_mixed_soft, f1_mixed_soft  = soft_class_metrics_CM(cm_mixed)
    # print("from CM: soft prec, recall, f1\n", precision_mixed_soft, recall_mixed_soft, f1_mixed_soft)
