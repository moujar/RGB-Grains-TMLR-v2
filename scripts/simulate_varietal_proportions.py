"""Bootstrap simulation of binary mixture-ratio estimation error across all
variety pairs, from a saved predictions .npz (see rgb_grains.train).

Usage:
    python scripts/simulate_varietal_proportions.py --predictions-npz expe/<run>/predictions.npz
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import argparse
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--boot', type=int, default=100)
parser.add_argument('--predictions-npz', required=True,
                     help="Path to a predictions .npz with keys logits_y1_on_y1, true_y1_test "
                          "(pure-stand, single-domain test predictions).")
parser.add_argument('--output-dir', default='calib_outputs', help="Where to save the calibration plots.")
args = parser.parse_args()
N_BOOTSTRAP = args.boot

RATIOS      = np.linspace(0.05,0.95,19)
rng_mix     = np.random.RandomState(42)

GREEN_Y1 = '#2E7D32'
GREEN_LIGHT = '#81C784'

DEBUG_BOOST = 1 ## to get more samples ?

flow = np.load(args.predictions_npz)
probs_y1 = flow["logits_y1_on_y1"]
y_true_1idx = flow["true_y1_test"]
classes     = sorted(np.unique(y_true_1idx))

# ── All C(8,2) = 28 pairs ────────────────────────────────────────────
PAIRS   = list(combinations(classes, 2))
N_PAIRS = len(PAIRS)
print(f'Analysing all {N_PAIRS} variety pairs  (C(8,2))')

pair_results = {}
biases = []


N_pools = []
for VAR_I, VAR_J in PAIRS:
    ## select grains that are i or j from the bag of grains (no cheating)
    idx_i  = np.where(y_true_1idx == VAR_I)[0]
    idx_j  = np.where(y_true_1idx == VAR_J)[0]
    ## pool is equal and symmetric for simplicity
    N_pool = min(len(idx_i), len(idx_j))
    N_pools.append(N_pool)
    results = []
    for ratio in RATIOS:
        n_i = max(1, int(round(N_pool*DEBUG_BOOST * ratio)))
        n_j = max(1, int(round(N_pool*DEBUG_BOOST * (1.0 - ratio))))
        true_ratio = n_i/(n_i+n_j) ## computed on the non-masked samples ! It's important, otherwise we cheat.
        boot_pred_ratios_i = []
        for _ in range(N_BOOTSTRAP):
            ## uild bootstrapped sample:
            samp_i = rng_mix.choice(idx_i, size=n_i, replace=True)
            samp_j = rng_mix.choice(idx_j, size=n_j, replace=True)
            samp   = np.concatenate([samp_i, samp_j])
            assert len(np.unique(y_true_1idx[samp_i])) == 1, "All samples should be from the same variety"
            assert len(np.unique(y_true_1idx[samp_j])) == 1, "All samples should be from the same variety"
            assert abs(true_ratio - samp_i.shape[0]/(samp_i.shape[0]+ samp_j.shape[0])) < 0.01 , f"ratio shall be exactly at the prescribed value, we get {true_ratio} vs {samp_i.shape[0]/(samp_i.shape[0]+ samp_j.shape[0])}"

            ## extract predictions probabilites
            p_i   = probs_y1[samp, VAR_I - 1]
            p_j   = probs_y1[samp, VAR_J - 1]

            ## Naive, robust approach:
            pred_ratio_var_i = ( p_i >= p_j).mean() ## as soon as p_i >= p_j, we predict variety i, regardless of the p_i value
            ## Claude's alternative suggestion (worse results)
            ## they sum to 1 :)

            ## recording
            boot_pred_ratios_i.append(pred_ratio_var_i)

        boot_arr_i = np.array(boot_pred_ratios_i)
        ci_lo,   ci_hi   = np.percentile(boot_arr_i, [2.5, 97.5])
        results.append((true_ratio, \
            np.median(boot_arr_i), \
                boot_arr_i.std(), \
                    ci_lo, ci_hi,
                         boot_arr_i.min(), \
                            boot_arr_i.max()))
    pair_results[(VAR_I, VAR_J)] = results
N_pools = np.array(N_pools)

os.makedirs(args.output_dir, exist_ok=True)


for ax_idx, (pair, results) in enumerate(pair_results.items()):
    plt.figure(1, [5,3])
    VAR_I, VAR_J = pair
    ratios_arr, means, stds, ci_lo_arr, ci_hi_arr, \
        mini, maxi = \
        map(np.array, zip(*results))

    plt.plot([0.0, 1.0], [0., 0.], '--', color='gray', lw=1.5, label='Perfect prediction')
    plt.plot(ratios_arr, means-ratios_arr , 'o-', color=GREEN_Y1, lw=2.0, markersize=5,
            label=f'Var {VAR_I} pred. ratio - true ratio')
    plt.fill_between(ratios_arr, ci_lo_arr-ratios_arr, ci_hi_arr-ratios_arr, alpha=0.18, color=GREEN_LIGHT)
    plt.scatter(ratios_arr, mini-ratios_arr, marker='.', color=GREEN_Y1)
    plt.scatter(ratios_arr, maxi-ratios_arr, marker='.', color=GREEN_Y1)
    plt.xticks(ratios_arr[::2])
    plt.yticks([-0.04, -0.02, 0, 0.02, 0.04])
    plt.xticks([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    plt.xlim([0.05, 0.95])
    plt.ylim([-0.05, 0.05])
    plt.xlabel('True fraction')
    plt.title(f'V{VAR_I} vs V{VAR_J} - pool size:{N_pools[ax_idx]}', pad=2)
    plt.grid(alpha=0.3);

    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/mixing_ratio_robustness_Nb={N_BOOTSTRAP}_varI={VAR_I}_varJ={VAR_J}.jpg', dpi=300, bbox_inches='tight')
    plt.savefig(f'{args.output_dir}/mixing_ratio_robustness_Nb={N_BOOTSTRAP}_varI={VAR_I}_varJ={VAR_J}.pdf', dpi=300, bbox_inches='tight')
    plt.close()

## averaged medians of all the 28 combinations: we plot the average of the means-ratios_arr and corresponding CI and min/max:
mean_medians = []
mean_std =[]
mean_ci_lo = []
mean_ci_hi = []
mean_mini = []
mean_maxi = []
for ax_idx, (pair, results) in enumerate(pair_results.items()):
    VAR_I, VAR_J = pair
    ratios_arr, means, stds, ci_lo_arr, ci_hi_arr, mini, maxi =  map(np.array, zip(*results))
    mean_medians.append(means)
    mean_std.append(stds)
    mean_ci_lo.append(ci_lo_arr)
    mean_ci_hi.append(ci_hi_arr)
    mean_mini.append(mini)
    mean_maxi.append(maxi)
mean_medians = np.array(mean_medians)
mean_std = np.array(mean_std)
mean_ci_lo = np.array(mean_ci_lo)
mean_ci_hi = np.array(mean_ci_hi)
mean_mini = np.array(mean_mini)
mean_maxi = np.array(mean_maxi)
mean_medians = np.mean(mean_medians, axis=0)
mean_std = np.mean(mean_std, axis=0)
mean_ci_lo = np.mean(mean_ci_lo, axis=0)
mean_ci_hi = np.mean(mean_ci_hi, axis=0)
mean_mini = np.mean(mean_mini, axis=0)
mean_maxi = np.mean(mean_maxi, axis=0)

plt.figure(2, [5,3])
plt.plot([0.0, 1.0], [0., 0.], '--', color='gray', lw=1.5, label='Perfect prediction')
plt.plot(ratios_arr, mean_medians-ratios_arr , 'o-', color=GREEN_Y1, lw=2.0, markersize=5,
        label=f'Var {VAR_I} pred. ratio - true ratio')
plt.fill_between(ratios_arr, mean_ci_lo-ratios_arr, mean_ci_hi-ratios_arr, alpha=0.18, color=GREEN_LIGHT)
plt.scatter(ratios_arr, mean_mini-ratios_arr, marker='.', color=GREEN_Y1)
plt.scatter(ratios_arr, mean_maxi-ratios_arr, marker='.', color=GREEN_Y1)
plt.xticks(ratios_arr[::2])
plt.yticks([-0.04, -0.02, 0, 0.02, 0.04])
plt.xticks([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
plt.xlim([0.05, 0.95])
plt.ylim([-0.05, 0.05])
plt.xlabel('True fraction')
plt.title(f'All pairs average', pad=2)
plt.grid(alpha=0.3);

plt.tight_layout()
plt.savefig(f'{args.output_dir}/mixing_ratio_robustness_Nb={N_BOOTSTRAP}_28-var-avg.jpg', dpi=300, bbox_inches='tight')
plt.savefig(f'{args.output_dir}/mixing_ratio_robustness_Nb={N_BOOTSTRAP}_28-var-avg.pdf', dpi=300, bbox_inches='tight')
plt.close()


## huge plot summairzing all pairs ion one pdf or jpg:

# ── Figure: aspect=1 (square) + unified % notation on both axes ──────
N_COLS = 7; N_ROWS = 4
fig, axes_grid = plt.subplots(N_ROWS, N_COLS,
                              figsize=(N_COLS * 3.2, N_ROWS * 3.4),
                              sharex=True, sharey=True)
axes_flat = axes_grid.flatten()
pct_labels = [f'{r:.0%}' for r in RATIOS]  # same label list for both axes

for ax_idx, (pair, results) in enumerate(pair_results.items()):
    ax = axes_flat[ax_idx]
    VAR_I, VAR_J = pair
    ratios_arr, means, stds, ci_lo_arr, ci_hi_arr, \
        mini, maxi = \
        map(np.array, zip(*results))

    ax.plot([0.1, 0.9], [0., 0.], '--', color='gray', lw=1.5, label='Perfect prediction')
    ax.plot(ratios_arr, means-ratios_arr , 'o-', color=GREEN_Y1, lw=2.0, markersize=5,
            label=f'Var {VAR_I} pred. ratio - true ratio')

    ax.fill_between(ratios_arr, ci_lo_arr-ratios_arr, ci_hi_arr-ratios_arr, alpha=0.18, color=GREEN_LIGHT)
    ax.scatter(ratios_arr, mini-ratios_arr, marker='.', color=GREEN_Y1)
    ax.scatter(ratios_arr, maxi-ratios_arr, marker='.', color=GREEN_Y1)

    # Unified % notation on both axes + square aspect ratio
    ax.set_xticks(ratios_arr[::2])
    ax.set_xticklabels(pct_labels[::2], fontsize=6, rotation=45)
    ## set yticks labels to be -4%,-2%, 0%, 2%, 4%:
    ax.set_yticks([-0.04, -0.02, 0, 0.02, 0.04])
    ax.set_yticklabels(['-4%','-2%','0%','2%','4%'], fontsize=6)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(-0.05, 0.05)
    ax.set_xlabel('True fraction', fontsize=7)
    ax.set_title(f'V{VAR_I} vs V{VAR_J} -- pool size:{N_pools[ax_idx]}', fontsize=8, fontweight='bold', pad=2)
    if ax_idx == 0:
        ax.legend(fontsize=6, loc='upper left')
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)

for ax_idx in range(N_PAIRS, N_ROWS * N_COLS):
    axes_flat[ax_idx].set_visible(False)
fig.text(0.5,  0.01, 'True fraction of Variety i/j',
         ha='center', fontsize=10)
fig.text(0.005, 0.5, 'Error in predicted fraction', va='center', rotation='vertical', fontsize=10)
fig.suptitle(   f'Mixing Ratio Error — {N_BOOTSTRAP} bootstraps\n',  fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{args.output_dir}/mixing_ratio_robustness_Nb={N_BOOTSTRAP}.jpg', dpi=300, bbox_inches='tight')
plt.savefig(f'{args.output_dir}/mixing_ratio_robustness_Nb={N_BOOTSTRAP}.pdf', dpi=300, bbox_inches='tight')
plt.show()
