import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from  module_plots import plot_transfer_learning_learningCurve
from tools import val_bal_acc_per_class_offline


def _check_view_major_labels(y_train_lr_aug, n_views):
    n_train = y_train_lr_aug.shape[0] // n_views
    if y_train_lr_aug.shape[0] != n_views * n_train:
        raise ValueError("Training label count must be divisible by the number of views")
    y_views = y_train_lr_aug.reshape((n_views, n_train, y_train_lr_aug.shape[1]))
    if not np.allclose(y_views, y_views[0][None, :, :]):
        raise ValueError("Feature views and labels are not aligned")
    return y_views[0]


def _predict_proba_view_average(clf, scaler, X_views, num_classes):
    """Average logistic-regression probabilities over view-major features."""
    n_views, n_samples, n_features = X_views.shape
    probabilities = np.zeros((n_views, n_samples, num_classes), dtype=np.float32)
    for view in range(n_views):
        X_view = X_views[view].reshape(n_samples, n_features)
        if scaler is not None:
            X_view = scaler.transform(X_view)
        view_probabilities = probabilities[view]
        view_probabilities[:, clf.classes_] = clf.predict_proba(X_view)
    return probabilities.mean(axis=0)


def frozen_feature_logreg_grid_search(
    X_train_lr,
    y_train_lr_aug,
    X_test_lr,
    y_test_lr,
    X_test_lr_aug,
    OUTPUT_DIR,
    c_grid=None,
    n_splits=5,
    scaler=True,
    random_state=42,
):
    """Tune logistic-regression C on frozen ConvNeXt features and test once.

    Cross-validation is performed at original-grain level. All 32 views of a
    grain stay in the same fold, avoiding augmented-view leakage.
    """
    if c_grid is None:
        c_grid = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    else:
        c_grid = np.asarray(c_grid, dtype=float)

    n_views = X_test_lr_aug.shape[0]
    n_train = X_train_lr.shape[0] // n_views
    n_features = X_train_lr.shape[-1]
    num_classes = y_train_lr_aug.shape[1]
    if X_train_lr.shape[0] != n_views * n_train:
        raise ValueError("Training feature count must be divisible by the number of views")

    X_train_views = X_train_lr.reshape((n_views, n_train, n_features))
    y_train_base = _check_view_major_labels(y_train_lr_aug, n_views)
    y_train_int = y_train_base.argmax(axis=1)
    pure_train_mask = (y_train_base > 0).sum(axis=1) == 1
    if pure_train_mask.sum() < 2:
        raise ValueError("Frozen-feature grid search needs at least two pure-label train samples")

    train_indices = np.where(pure_train_mask)[0]
    y_cv = y_train_int[train_indices]
    present_classes, class_counts = np.unique(y_cv, return_counts=True)
    max_splits = int(class_counts.min()) if len(class_counts) else 0
    actual_splits = min(n_splits, max_splits)

    cv_mean_scores = []
    cv_std_scores = []
    if actual_splits >= 2 and len(present_classes) >= 2:
        splitter = StratifiedKFold(
            n_splits=actual_splits, shuffle=True, random_state=random_state
        )
        for C in c_grid:
            fold_scores = []
            for fold_train_rel, fold_val_rel in splitter.split(train_indices, y_cv):
                fold_train_idx = train_indices[fold_train_rel]
                fold_val_idx = train_indices[fold_val_rel]
                X_fold_train = X_train_views[:, fold_train_idx, :].reshape(
                    n_views * len(fold_train_idx), n_features
                )
                y_fold_train = np.tile(y_train_int[fold_train_idx], n_views)

                scaler_lr = StandardScaler() if scaler else None
                if scaler_lr is not None:
                    X_fold_train = scaler_lr.fit_transform(X_fold_train)

                clf = LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    C=float(C),
                    random_state=random_state,
                )
                clf.fit(X_fold_train, y_fold_train)

                val_prob = _predict_proba_view_average(
                    clf, scaler_lr, X_train_views[:, fold_val_idx, :], num_classes
                )
                val_pred = val_prob.argmax(axis=1)
                fold_score, _, _ = val_bal_acc_per_class_offline(
                    val_pred, y_train_base[fold_val_idx], num_classes, strict=False
                )
                fold_scores.append(fold_score)
            cv_mean_scores.append(float(np.mean(fold_scores)))
            cv_std_scores.append(float(np.std(fold_scores)))
    else:
        cv_mean_scores = [np.nan for _ in c_grid]
        cv_std_scores = [np.nan for _ in c_grid]

    if np.all(np.isnan(cv_mean_scores)):
        best_C = 1.0 if 1.0 in c_grid else float(c_grid[len(c_grid) // 2])
    else:
        best_C = float(c_grid[int(np.nanargmax(cv_mean_scores))])

    X_train_pure = X_train_views[:, train_indices, :].reshape(
        n_views * len(train_indices), n_features
    )
    y_train_pure = np.tile(y_train_int[train_indices], n_views)
    scaler_final = StandardScaler() if scaler else None
    if scaler_final is not None:
        X_train_pure = scaler_final.fit_transform(X_train_pure)

    final_clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        C=best_C,
        random_state=random_state,
    )
    final_clf.fit(X_train_pure, y_train_pure)

    test_prob_avg_features = np.zeros((X_test_lr.shape[0], num_classes), dtype=np.float32)
    X_test_sc = scaler_final.transform(X_test_lr) if scaler_final is not None else X_test_lr
    test_prob_avg_features[:, final_clf.classes_] = final_clf.predict_proba(X_test_sc)
    test_pred_avg_features = test_prob_avg_features.argmax(axis=1)
    test_bal_acc_avg_features, test_precision, test_recall = val_bal_acc_per_class_offline(
        test_pred_avg_features, y_test_lr, num_classes, strict=False
    )

    test_prob_view_average = _predict_proba_view_average(
        final_clf, scaler_final, X_test_lr_aug, num_classes
    )
    test_pred_view_average = test_prob_view_average.argmax(axis=1)
    test_bal_acc_view_average, _, test_recall_view_average = val_bal_acc_per_class_offline(
        test_pred_view_average, y_test_lr, num_classes, strict=False
    )

    result = {
        "c_grid": c_grid,
        "cv_mean_scores": np.asarray(cv_mean_scores, dtype=float),
        "cv_std_scores": np.asarray(cv_std_scores, dtype=float),
        "best_C": best_C,
        "n_splits": actual_splits,
        "train_sample_count": len(train_indices),
        "test_bal_acc_avg_features": float(test_bal_acc_avg_features),
        "test_bal_acc_view_average": float(test_bal_acc_view_average),
        "test_precision_avg_features": np.asarray(test_precision, dtype=float),
        "test_recall_avg_features": np.asarray(test_recall, dtype=float),
        "test_recall_view_average": np.asarray(test_recall_view_average, dtype=float),
        "test_prob_avg_features": test_prob_avg_features,
        "test_prob_view_average": test_prob_view_average,
    }

    np.savez(
        OUTPUT_DIR / "frozen_feature_logreg_grid_search.npz",
        **result,
    )
    print(
        "[*] Frozen-feature LR best C=%s  cv_bal=%.4f  test_bal=%.4f  "
        "test_bal_viewavg=%.4f"
        % (
            best_C,
            np.nanmax(result["cv_mean_scores"]) if not np.all(np.isnan(result["cv_mean_scores"])) else np.nan,
            test_bal_acc_avg_features,
            test_bal_acc_view_average,
        )
    )
    return result

def exactFit_classifier_and_score(Ntrain_cap, X_train_lr, y_train_lr_aug , X_test_lr, y_test_lr, X_test_lr_aug, \
    scaler= True):
    NVIEWS = X_test_lr_aug.shape[0]
    Nfeatures = X_test_lr_aug.shape[-1]
    num_classes = y_train_lr_aug.shape[1]
    Ntrain_full = X_train_lr.shape[0]//NVIEWS
    if X_train_lr.shape[0] != NVIEWS * Ntrain_full:
        raise ValueError("Training feature count must be divisible by the number of views")
    if Ntrain_cap==0 or Ntrain_cap > Ntrain_full:
        Ntrain_cap = Ntrain_full
    X_train_lr_views   = X_train_lr.reshape((NVIEWS,Ntrain_full,Nfeatures))
    y_train_lr_aug_int = y_train_lr_aug.argmax(1)

    y_train_lr_reduced = y_train_lr_aug_int.reshape((NVIEWS,Ntrain_full))
    if not np.all(y_train_lr_reduced == y_train_lr_reduced[0]):
        raise ValueError("Feature views and labels are not aligned")

    base_labels = y_train_lr_reduced[0]
    present_classes = np.unique(base_labels)
    Ntrain_cap = max(Ntrain_cap, len(present_classes))
    if Ntrain_cap > Ntrain_full:
        raise ValueError("Not enough training samples to include every present class")

    # Guarantee one sample per present class, then fill the remaining budget.
    rng = np.random.RandomState(42)
    required = np.array(
        [rng.choice(np.where(base_labels == class_id)[0]) for class_id in present_classes]
    )
    available = np.setdiff1d(np.arange(Ntrain_full), required)
    extra_count = Ntrain_cap - len(required)
    extra = rng.choice(available, size=extra_count, replace=False)
    selection = np.concatenate([required, extra])

    X_train_lr_reduced = X_train_lr_views[:,selection, :]
    y_train_lr_reduced = y_train_lr_reduced[:, selection]
    X_train_lr_reduced = X_train_lr_reduced.reshape((NVIEWS*Ntrain_cap,Nfeatures))
    y_train_lr_reduced = y_train_lr_reduced.reshape((NVIEWS*Ntrain_cap))
    print(f"classes present in train set: {np.unique(y_train_lr_reduced)}")
    print(f"classes present in train set: {np.unique(y_train_lr_aug_int)}")
    # y_train_lr_aug

    y_test_lr_int  = y_test_lr.argmax(1)

    if scaler:
        scaler_lr = StandardScaler()
        scaler_lr.fit(X_train_lr_reduced)
        X_train_sc = scaler_lr.transform(X_train_lr_reduced)  # scale all 32*N
        X_test_sc  = scaler_lr.transform(X_test_lr)
        X_test_sc_views = np.zeros(X_test_lr_aug.shape)
        for view in range(NVIEWS):  
            X_test_sc_views[view]  = scaler_lr.transform(X_test_lr_aug[view])

        # X_train_sc - (X_train_lr_reduced-scaler_lr.mean_)/scaler_lr.scale_ --> is basically 0.
    else:
        X_train_sc = X_train_lr_reduced  # scale all 32*N
        X_test_sc  = X_test_lr
        X_test_sc_views = X_test_lr_aug



    # print('  Fitting LogisticRegression (lbfgs, max_iter=1000)...')
    lr_clf = LogisticRegression(class_weight="balanced", max_iter=1000, solver='lbfgs', C=1.0, random_state=42)
    lr_clf.fit(X_train_sc, y_train_lr_reduced)

    ## train 
    acc_lr_train = balanced_accuracy_score(y_train_lr_reduced, lr_clf.predict(X_train_sc))

    ## test (averaged input features, which are the backbone's representations (over NVIEWS views))
    preds_lr_y2 = lr_clf.predict(X_test_sc)
    acc_lr_test = balanced_accuracy_score(y_test_lr_int, preds_lr_y2)

    ## test (averaging predictions)
    probs_lr_y2_views = np.zeros((NVIEWS, preds_lr_y2.shape[0], num_classes))
    for view in range(NVIEWS):
        view_probabilities = probs_lr_y2_views[view]
        view_probabilities[:, lr_clf.classes_] = lr_clf.predict_proba(X_test_sc_views[view])
    preds_lr_y2_views = probs_lr_y2_views.mean(0).argmax(1)
    acc_lr_test_views  = balanced_accuracy_score(y_test_lr_int, preds_lr_y2_views)
    
    if scaler:
        sigma = scaler_lr.scale_          # shape (d,)
        mu    = scaler_lr.mean_           # shape (d,)
        W_raw = lr_clf.coef_ / sigma           # (num_classes, d)
        b_raw = lr_clf.intercept_ - (lr_clf.coef_ / sigma) @ mu
    else:
        W_raw = lr_clf.coef_
        b_raw = lr_clf.intercept_

    return Ntrain_cap, acc_lr_train, acc_lr_test, acc_lr_test_views, W_raw, b_raw
# acc_lr_train, acc_lr_test = exactFit_classifier_and_score(X_train_lr, y_train_lr_aug, X_test_lr, y_test_lr)


def exactFit_classifier_return_W_b(X_train_lr, y_train_lr_aug, scaler=False):
    """
    Simplified logistic regression fit for kickstart initialization.
    
    Only trains on the provided data and returns W_raw, b_raw.
    No test evaluation, no learning curve logic.
    
    Parameters
    ----------
    X_train_lr : ndarray
        Training features, shape (n_views * N_train, n_features) if n_views > 1,
        or (N_train, n_features) if n_views == 1.
    y_train_lr_aug : ndarray
        One-hot encoded labels, shape (n_views * N_train, n_classes) if n_views > 1,
        or (N_train, n_classes) if n_views == 1.
    n_views : int
        Number of augmentation views used (default 32).
    scaler : bool
        Whether to apply StandardScaler (default False, as kickstart doesn't use it).
    
    Returns
    -------
    W_raw : ndarray
        Logistic regression weights, shape (n_classes, n_features).
    b_raw : ndarray
        Logistic regression biases, shape (n_classes,).
    acc_train : float
        Balanced accuracy on training data (for logging only).
    """
    y_train_int = y_train_lr_aug.argmax(axis=1)
    n_classes = y_train_lr_aug.shape[1]
    
    if scaler:
        scaler_lr = StandardScaler()
        X_train_sc = scaler_lr.fit_transform(X_train_lr)
    else:
        X_train_sc = X_train_lr
    
    lr_clf = LogisticRegression(
        class_weight="balanced", max_iter=1000, solver='lbfgs', C=1.0, random_state=42
    )
    # assert False

    lr_clf.fit(X_train_sc, y_train_int)
    
    acc_train = balanced_accuracy_score(y_train_int, lr_clf.predict(X_train_sc))
    print(f"[*] exactFit_classifier_return_W_b: train_bal_acc={acc_train:.4f}")
    
    if scaler:
        sigma = scaler_lr.scale_
        mu = scaler_lr.mean_
        W_raw = lr_clf.coef_ / sigma
        b_raw = lr_clf.intercept_ - (lr_clf.coef_ / sigma) @ mu
    else:
        W_raw = lr_clf.coef_
        b_raw = lr_clf.intercept_
    
    return W_raw, b_raw, acc_train


def learning_curve_exactFit(X_train_lr, y_train_lr_aug , X_test_lr, y_test_lr, X_test_lr_aug, OUTPUT_DIR):
    NVIEWS=X_test_lr_aug.shape[0]
    Ntrain_full = X_train_lr.shape[0]//NVIEWS
    base=2
    n_present_classes = len(np.unique(y_train_lr_aug.argmax(axis=1)))
    Ncap_range = []
    ncap = n_present_classes
    while ncap < Ntrain_full:
        Ncap_range.append(ncap)
        ncap *= base
    Ncap_range.append(Ntrain_full)
    Ncaps = []
    ## we record the accuracies into list, append to them, then convert to numpy:
    acc_lr_train_list = []
    acc_lr_test_list = []
    acc_lr_test_views_list = [] 
    for Ncap in Ncap_range:
        t0 = time.time()
        Ncap, acc_lr_train, acc_lr_test, acc_lr_test_views, W_raw, b_raw \
        = exactFit_classifier_and_score(Ncap, X_train_lr, y_train_lr_aug , X_test_lr, y_test_lr, X_test_lr_aug)
        Ncaps.append(Ncap)
        acc_lr_train_list.append(acc_lr_train)
        acc_lr_test_list.append(acc_lr_test)
        acc_lr_test_views_list.append(acc_lr_test_views)
        print(f' Ncap={Ncap} ,  Train acc: {acc_lr_train:.4f} , Test acc : {acc_lr_test:.4f} , Test acc (views): {acc_lr_test_views:.4f} , Time taken for this Ncap: {time.time()-t0:.2f} s')
    Ncaps = np.array(Ncaps)
    acc_lr_train_list = np.array(acc_lr_train_list)
    acc_lr_test_list = np.array(acc_lr_test_list)
    acc_lr_test_views_list = np.array(acc_lr_test_views_list)
    np.savez(OUTPUT_DIR / 'transfer_learning.npz', Ncaps=Ncaps, acc_lr_train_list=acc_lr_train_list, acc_lr_test_list=acc_lr_test_list, acc_lr_test_views_list=acc_lr_test_views_list)

    plot_transfer_learning_learningCurve(Ncaps, acc_lr_train_list, acc_lr_test_list, acc_lr_test_views_list, OUTPUT_DIR)

    return Ncaps, acc_lr_train_list, acc_lr_test_list, acc_lr_test_views_list, W_raw, b_raw 

# # Comparison bar chart (6 bars)
# labels_cmp = ['Y1->Y2\n(Direct)', 'Y1->Y2\n(Head FT)',
#               'Y1->Y2\n LogReg\n768', 'Y2 Native\n(Upper Bound)']
# accs_cmp   = [acc_direct, acc_ft_y2, acc_lr_test, acc_native]
# colors_cmp = ['#E53935', '#1565C0', '#7B1FA2', '#2E7D32']
# edge_cmp   = ['#B71C1C', '#0D47A1', '#4A148C', '#1B5E20']

# fig, ax = plt.subplots(figsize=(12, 5))
# bars = ax.bar(labels_cmp, accs_cmp, color=colors_cmp, edgecolor=edge_cmp,
#               linewidth=1.2, alpha=0.87, width=0.55)
# for bar, acc in zip(bars, accs_cmp):
#     ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.010,
#             f'{acc:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
# ax.set_ylabel('Accuracy', fontsize=12)
# ax.set_ylim(0, 1.15)
# ax.set_title(
#     'Y1->Y2 Transfer: Comparison of Adaptation Strategies\nConvNeXt-Tiny — Year 2 Test Set',
#     fontsize=13, fontweight='bold')
# ax.grid(axis='y', alpha=0.35); ax.set_axisbelow(True)
# plt.tight_layout()
# plt.savefig(BASE_DIR / 'head_finetune_comparison.png', dpi=300, bbox_inches='tight')
# plt.show()
