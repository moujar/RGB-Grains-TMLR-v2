import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from  module_plots import plot_transfer_learning_learningCurve

def exactFit_classifier_and_score(Ntrain_cap, X_train_lr, y_train_lr_aug , X_test_lr, y_test_lr, X_test_lr_aug, \
    scaler= True):
    NVIEWS = X_test_lr_aug.shape[0]
    Nfeatures = X_test_lr_aug.shape[-1]
    NCLASSES=8
    Ntrain_full = X_train_lr.shape[0]//NVIEWS
    if Ntrain_cap==0 or Ntrain_cap > Ntrain_full:
        Ntrain_cap = Ntrain_full
    if Ntrain_cap <= NCLASSES :
        Ntrain_cap = NCLASSES ## number of classes 
    X_train_lr_views   = X_train_lr.reshape((NVIEWS,Ntrain_full,Nfeatures))
    y_train_lr_aug_int = y_train_lr_aug.argmax(1) ## TO BE DEBUGGED, after passing into onehot vectors for y. 

    y_train_lr_reduced = y_train_lr_aug_int.reshape((NVIEWS,Ntrain_full))
    selection = np.random.permutation(Ntrain_full)[:Ntrain_cap]
    if Ntrain_cap <= NCLASSES :
        ## make a selection based on y_train_lr_reduced, so that there is one sample of each class:
        selection = np.zeros(Ntrain_cap, dtype=int)
        for i in range(Ntrain_cap):
            selection[i] = np.where(y_train_lr_reduced[0]==i)[0][0]
        # y_train_lr_reduced[0,selection] == array([0, 1, 2, 3, 4, 5, 6, 7])
    retries = 0 
    while len(np.unique(y_train_lr_reduced[0,selection])) < NCLASSES and retries < 100:
        selection = np.random.permutation(Ntrain_full)[:Ntrain_cap]
        retries += 1
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
    preds_lr_y2  = lr_clf.predict(X_test_sc) + 1          # back to 1-indexed (1–8)
    acc_lr_test  = balanced_accuracy_score(y_test_lr_int, lr_clf.predict(X_test_sc))

    ## test (averaging predictions)
    probs_lr_y2_views = np.zeros((NVIEWS, preds_lr_y2.shape[0], 8))
    for view in range(NVIEWS):  
        probs_lr_y2_views[view] = lr_clf.predict_proba(X_test_sc_views[view]) + 1          # back to 1-indexed (1–8)
    # probs_lr_y2_views  = probs_lr_y2_views.mean(0)
    preds_lr_y2_views = (probs_lr_y2_views.mean(0)).argmax(1)
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
