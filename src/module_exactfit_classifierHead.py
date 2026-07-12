import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer
import matplotlib.pyplot as plt

def exactFit_classifier_and_score(Ntrain_cap, X_train_lr, y_train_lr_aug , X_test_lr, y_test_lr, X_test_lr_aug):
    NVIEWS = X_test_lr_aug.shape[0]
    Nfeatures = X_test_lr_aug.shape[-1]
    NCLASSES=8
    Ntrain_full = X_train_lr.shape[0]//NVIEWS
    if Ntrain_cap==0 or Ntrain_cap > Ntrain_full:
        Ntrain_cap = Ntrain_full
    if Ntrain_cap <= NCLASSES :
        Ntrain_cap = NCLASSES ## number of classes 
    X_train_lr_views   = X_train_lr.reshape((NVIEWS,Ntrain_full,Nfeatures))
    y_train_lr_aug = y_train_lr_aug.argmax(1) ## TO BE DEBUGGED, after passing into onehot vectors for y. 

    y_train_lr_reduced = y_train_lr_aug.reshape((NVIEWS,Ntrain_full))
    selection = np.random.permutation(Ntrain_full)[:Ntrain_cap]
    if Ntrain_cap <= NCLASSES :
        ## make a selection based on y_train_lr_reduced, so that there is one sample of each class:
        selection = np.zeros(Ntrain_cap, dtype=int)
        for i in range(Ntrain_cap):
            selection[i] = np.where(y_train_lr_reduced[0]==i)[0][0]
        # y_train_lr_reduced[0,selection] == array([0, 1, 2, 3, 4, 5, 6, 7])
    while len(np.unique(y_train_lr_reduced[0,selection])) < NCLASSES :
        selection = np.random.permutation(Ntrain_full)[:Ntrain_cap]
    X_train_lr_reduced = X_train_lr_views[:,selection, :]
    y_train_lr_reduced = y_train_lr_reduced[:, selection]
    X_train_lr_reduced = X_train_lr_reduced.reshape((NVIEWS*Ntrain_cap,Nfeatures))
    y_train_lr_reduced = y_train_lr_reduced.reshape((NVIEWS*Ntrain_cap))
    # print(f"classes present in train set: {np.unique(y_train_lr_reduced)}")
    # print(f"classes present in train set: {np.unique(y_train_lr_aug)}")
    # y_train_lr_aug

    scaler_lr = StandardScaler()
    scaler_lr.fit(X_train_lr_reduced)
    X_train_sc = scaler_lr.transform(X_train_lr_reduced)  # scale all 32*N
    X_test_sc  = scaler_lr.transform(X_test_lr)
    X_test_sc_views = np.zeros(X_test_lr_aug.shape)
    for view in range(NVIEWS):  
        X_test_sc_views[view]  = scaler_lr.transform(X_test_lr_aug[view])

    # print('  Fitting LogisticRegression (lbfgs, max_iter=1000)...')
    lr_clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0, random_state=42)
    lr_clf.fit(X_train_sc, y_train_lr_reduced)

    ## train 
    acc_lr_train = balanced_accuracy_score(y_train_lr_reduced, lr_clf.predict(X_train_sc))

    ## test (averaged input features, which are the backbone's representations (over NVIEWS views))
    preds_lr_y2  = lr_clf.predict(X_test_sc) + 1          # back to 1-indexed (1–8)
    acc_lr_test  = balanced_accuracy_score(y_test_lr, lr_clf.predict(X_test_sc))

    ## test (averaging predictions)
    probs_lr_y2_views = np.zeros((NVIEWS, preds_lr_y2.shape[0], 8))
    for view in range(NVIEWS):  
        probs_lr_y2_views[view] = lr_clf.predict_proba(X_test_sc_views[view]) + 1          # back to 1-indexed (1–8)
    # probs_lr_y2_views  = probs_lr_y2_views.mean(0)
    preds_lr_y2_views = (probs_lr_y2_views.mean(0)).argmax(1)
    acc_lr_test_views  = balanced_accuracy_score(y_test_lr, preds_lr_y2_views)
    
    return Ntrain_cap, acc_lr_train, acc_lr_test, acc_lr_test_views
# acc_lr_train, acc_lr_test = exactFit_classifier_and_score(X_train_lr, y_train_lr_aug, X_test_lr, y_test_lr)


def learning_curve_exactFit(X_train_lr, y_train_lr_aug , X_test_lr, y_test_lr, X_test_lr_aug, OUTPUT_DIR):

    NVIEWS=32
    Ntrain_full = X_train_lr.shape[0]//NVIEWS
    base=2
    NCLASSES=8
    Ncap_range = [NCLASSES*base**k for k in range(0,int((np.log(Ntrain_full/NCLASSES)+1)/np.log(base))+1)]
    Ncaps = []
    ## we record the accuracies into list, append to them, then convert to numpy:
    acc_lr_train_list = []
    acc_lr_test_list = []
    acc_lr_test_views_list = [] 
    for Ncap in Ncap_range:
        if  Ncap==1:
            Ncap=2
        t0 = time.time()
        Ncap, acc_lr_train, acc_lr_test, acc_lr_test_views \
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

    return Ncaps, acc_lr_train_list, acc_lr_test_list, acc_lr_test_views_list


# ─────────────────────────── 5.1 "kickstart loop" recycling ───────────────────────────
# Recycle the frozen-feature extraction (32 views, default 3train1test splitting_choice,
# called from the main script): save the extracted features to disk once, then run a
# quick CV "grid" search on the logistic-regression regularization C, and report the
# held-out test set result -- the "best thing we get from frozen features" (missing in v1).

def save_frozen_features(path, X_train_lr, y_train_lr_aug, X_test_lr, y_test_lr, X_test_lr_aug):
    """Save the once-extracted (32-view) frozen features to disk, so the CV grid
    search over C below can be re-run cheaply without re-extracting features."""
    np.savez(
        path,
        X_train_lr=X_train_lr, y_train_lr_aug=y_train_lr_aug,
        X_test_lr=X_test_lr, y_test_lr=y_test_lr, X_test_lr_aug=X_test_lr_aug,
    )
    print(f"[*] Frozen features saved to: {path}")


def load_frozen_features(path):
    flow = np.load(path)
    return (flow["X_train_lr"], flow["y_train_lr_aug"], flow["X_test_lr"],
            flow["y_test_lr"], flow["X_test_lr_aug"])


def frozen_feature_gridsearch_C(
    X_train_lr, y_train_lr_aug, X_test_lr, y_test_lr,
    c_grid=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0), cv=5, seed=42,
):
    """CV grid search over the logistic-regression C, on the frozen (32-view) train
    features, then report the *held-out* test set's balanced accuracy at the best C.

    Returns a dict with best_C, cv_mean_scores/cv_std_scores (one pair per c_grid
    entry, balanced accuracy), and test_bal_acc (at best_C, on X_test_lr/y_test_lr).
    """
    scaler = StandardScaler().fit(X_train_lr)
    X_train_sc = scaler.transform(X_train_lr)
    X_test_sc = scaler.transform(X_test_lr)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    bal_acc_scorer = make_scorer(balanced_accuracy_score)

    cv_mean_scores, cv_std_scores = [], []
    for C in c_grid:
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=C, random_state=seed)
        scores = cross_val_score(clf, X_train_sc, y_train_lr_aug, cv=skf, scoring=bal_acc_scorer)
        cv_mean_scores.append(scores.mean())
        cv_std_scores.append(scores.std())
        print(f"  C={C:<8g} CV bal_acc={scores.mean():.4f} +/- {scores.std():.4f}")

    best_idx = int(np.argmax(cv_mean_scores))
    best_C = c_grid[best_idx]

    ## refit on the full (frozen-feature) train set at best_C, then evaluate on the held-out test set:
    final_clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=best_C, random_state=seed)
    final_clf.fit(X_train_sc, y_train_lr_aug)
    test_bal_acc = balanced_accuracy_score(y_test_lr, final_clf.predict(X_test_sc))
    print(f"[*] Best C={best_C} (CV bal_acc={cv_mean_scores[best_idx]:.4f}) -> test bal_acc={test_bal_acc:.4f}")

    return {
        "c_grid": np.asarray(c_grid, dtype=float),
        "cv_mean_scores": np.asarray(cv_mean_scores),
        "cv_std_scores": np.asarray(cv_std_scores),
        "best_C": best_C,
        "test_bal_acc": test_bal_acc,
        "classifier": final_clf,
    }

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
