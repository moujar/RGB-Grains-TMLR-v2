import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from pathlib import Path


def calibration_plot(logits_test_data, true_y_test, OUTPUT_DIR, experiment_short_name=""):
    # ── Compute probabilities on Year 1 test set ─────────────────────────────
    y_true_0 = true_y_test.argmax(1)
    confidence = logits_test_data.max(axis=1)
    predicted = logits_test_data.argmax(axis=1)
    ## support flexible PLL style correctness:
    # correct_per_class[c] = support[mask, preds[mask]].sum().item()
    # total_per_class[c] = mask.sum().item()
    support = true_y_test > 0
    mask = np.arange(len(y_true_0))
    correct = support[mask, predicted[mask]] # .sum().item()
    # correct = (predicted == y_true_0).astype(float)

    # ── Calibration binning — non-overlapping fixed-width bins ────────────────
    # Use 10 equal-width bins [0.0, 0.1), [0.1, 0.2), ... [0.9, 1.0]
    # Each bar sits exactly in its own range — no overlap possible.
    N_BINS = 20
    bin_size = 1.0 / N_BINS
    bin_edges = np.linspace(0.0, 1.0, N_BINS + 1)
    bin_left = bin_edges[:-1]  # left edge of each bin (bar x-position)
    bin_acc, bin_conf, bin_count = [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Include right endpoint only for last bin
        if hi == 1.0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        n_in = mask.sum()
        if n_in > 0:
            bin_acc.append(correct[mask].mean())
            bin_conf.append(confidence[mask].mean())
            bin_count.append(int(n_in))
        else:
            bin_acc.append(None)
            bin_conf.append((lo + hi) / 2)
            bin_count.append(0)

    ece = sum(
        abs(a - c) * n / len(y_true_0)
        for a, c, n in zip(bin_acc, bin_conf, bin_count)
        if a is not None
    )

    # ── Plot ─────────────────────────────────────────────────────────────────
    # Green = Y1→Y1 test set
    GREEN_Y1 = "#2E7D32"
    GREEN_LIGHT = "#81C784"

    # ── Figure 1  ─────────
    fig1, ax = plt.subplots(1, 1, figsize=(5, 5))  # square figure
    bar_width = bin_size * 1.0  # full-width bars
    for lo, acc in zip(bin_left, bin_acc):
        if acc is not None:
            ax.bar(
                lo,
                acc,
                width=bar_width,
                align="edge",
                color=GREEN_LIGHT,
                edgecolor=GREEN_Y1,
                linewidth=1.2,
                alpha=0.85,
                label="Empirical accuracy" if lo == bin_left[0] else "",
            )
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax.set_xlabel("Model confidence (top-1)", fontsize=11)
    ax.set_ylabel("Fraction of correct predictions", fontsize=11)
    expe_title = f" {experiment_short_name}" if experiment_short_name else ""
    ax.set_title(f"Reliability Diagram{expe_title}  (ECE = {ece:.4f})")
    ax.set_xticks(bin_edges[::4])
    # ax.set_xticklabels([f'{v:.1f}' for v in bin_edges[::4]], fontsize=8)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")  # square plot
    ax.grid(alpha=0.4)
    ax.set_axisbelow(True)
    fig1.tight_layout()
    fig1.savefig(
        OUTPUT_DIR / f"calibration_reliability_{experiment_short_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig1.savefig(
        OUTPUT_DIR / f"calibration_reliability_{experiment_short_name}.pdf",
        bbox_inches="tight",
    )
    # plt.show()
    plt.close()
    print(f"[*] Saved calibration_reliability.png + .pdf  (main text figure)")

    # ── Figure 2 (appendix): Confidence histogram ────────────────────────
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    ax2.hist(confidence, bins=20, color="#E57373", alpha=0.80, edgecolor="white")
    ax2.set_xlabel("Model confidence", fontsize=11)
    ax2.set_ylabel("Number of samples", fontsize=11)
    ax2.set_title(f"Confidence Distribution{expe_title} — ConvNeXt-Tiny (Y1 test)")
    ax2.grid(alpha=0.4)
    ax2.set_axisbelow(True)
    # fig2.suptitle('ConvNeXt-Tiny Calibration  |  Year 1 → Year 1',                fontsize=11, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig(
        OUTPUT_DIR / f"calibration_histogram_{experiment_short_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig2.savefig(
        OUTPUT_DIR / f"calibration_histogram_{experiment_short_name}.pdf",
        bbox_inches="tight",
    )
    # plt.show()
    plt.close()
    print(f"[*] Saved calibration_histogram.png + .pdf  (appendix figure)")

    print(f"ECE        = {ece:.4f}")
    print(f"Mean conf  = {confidence.mean():.4f}")
    print(f"Accuracy   = {correct.mean():.4f}")


def plot_training_curve(
    training_metrics_path="expe/noAug-epoch200/training_metrics.npz",
    prefix="",
    experiment_short_name="",
):
    """Plot and save training curve during training (every 10 epochs)."""
    expe = os.path.dirname(training_metrics_path)
    flow = np.load(training_metrics_path)
    train_loss = flow["train_loss"]
    train_acc = flow["train_acc"]
    # val_acc = flow["val_acc"]
    val_bal_acc = flow["val_bal_acc"] # , val_acc)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.semilogy(train_loss, ls="-.", label="Training Loss", color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(
        train_acc, ls="-.", label="Training Accuracy (not balanced)", color="tab:orange"
    )
    ax2.plot(val_bal_acc, ls="-", label="Validation Accuracy (balanced)", color="tab:green")
    # if len(val_bal_acc) > 0 and np.any(val_bal_acc > 0):
    #     ax2.plot(
    #         val_bal_acc,
    #         ls="--",
    #         label="Validation Balanced Accuracy",
    #         color="darkgreen",
    #     )
    ax2.set_ylabel("Accuracy", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.set_ylim([0.7, 1.0])
    ax2.axhline(y=0.945, color="k", linestyle="--", label="reference bal acc: 0.945")

    expe_title = f" {experiment_short_name}" if experiment_short_name else ""
    plt.title(f"Training and Validation Loss\n{expe_title}")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xlim([0, len(train_loss)])
    ax2.set_xlim([0, len(train_loss)])

    # Save to both JPG and PDF
    jpg_path = os.path.join(expe, f"{prefix}training_curve_{experiment_short_name}.jpg")
    pdf_path = os.path.join(expe, f"{prefix}training_curve_{experiment_short_name}.pdf")
    svg_path = os.path.join(expe, f"{prefix}training_curve_{experiment_short_name}.svg")
    plt.savefig(jpg_path, dpi=300)
    plt.savefig(pdf_path)
    plt.savefig(svg_path)
    plt.close()
    print(f"[*] Training curve saved to: {jpg_path} (+pdf+svg)")


# training_metrics_path = "expe/transferlearning/training_metrics.npz"
# training_metrics_path = "expe/noAug-noCutMix-noMixUp-epoch50-nworker=5/training_metrics.npz"
# training_metrics_path = "expe/noAug-epoch200/training_metrics.npz"
# plot_training_curve(training_metrics_path)


# def _plot_training_curve(self, prefix=""):
#     """Plot and save training curve during training (every 10 epochs)."""
#     if len(self.train_loss) == 0:
#         return

#     fig, ax1 = plt.subplots(figsize=(10, 5))
#     ax1.semilogy(self.train_loss, label='Training Loss', color='tab:blue')
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Loss', color='tab:blue')
#     ax1.tick_params(axis='y', labelcolor='tab:blue')

#     ax2 = ax1.twinx()
#     ax2.plot(self.train_acc, label='Training Accuracy (not balanced)', color='tab:orange')
#     ax2.plot(self.val_acc, label='Validation Balanced Accuracy', color='tab:green')
#     ax2.set_ylabel('Accuracy', color='tab:orange')
#     ax2.tick_params(axis='y', labelcolor='tab:orange')
#     ## manage to have the second y-axis, and only this one, in a given range:
#     ax2.set_ylim([0.7,1.0])
#     ax2.axhline(y=0.945, color='k', linestyle='--', label='reference bal acc: 0.945')


#     plt.title('Training and Validation Loss')
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

#     # Save to both JPG and PDF
#     jpg_path = os.path.join(self.expe, f"{prefix}training_curve.jpg")
#     pdf_path = os.path.join(self.expe, f"{prefix}training_curve.pdf")
#     plt.savefig(jpg_path, dpi=150)
#     plt.savefig(pdf_path)
#     plt.close()
#     self._log_fn(f"[*] Training curve saved to: {jpg_path}")


def plot_transfer_learning_learningCurve(
    Ncaps, acc_lr_train_list, acc_lr_test_list, acc_lr_test_views_list, OUTPUT_DIR
):
    plt.figure(figsize=(6, 4))
    plt.semilogx(
        Ncaps,
        acc_lr_train_list,
        color="tab:blue",
        ls="-",
        marker="o",
        label="Train Balanced Accuracy",
    )
    plt.semilogx(
        Ncaps,
        acc_lr_test_list,
        c="tab:green",
        ls="-",
        marker="x",
        label="Test Balanced Accuracy",
    )
    # plt.semilogx(Ncaps, acc_lr_test_list, c='tab:green', ls='-', marker='x', label='Test Accuracy (averaging input features)')
    # plt.semilogx(Ncaps, acc_lr_test_views_list, c='darkgreen', ls="--", marker="+", label='Test Accuracy (averaging over logReg logits)')
    plt.axhline(y=0.945, color="k", linestyle="--", label="reference bal acc: 0.945")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Accuracy")
    # plt.xlim([1,Ncaps[-1]])
    plt.ylim([0.0, 1.0])
    # plt.title('Accuracy vs Number of Training Samples')
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "transfer_learning.jpg", dpi=300)
    plt.savefig(OUTPUT_DIR / "transfer_learning.pdf")
    plt.savefig(OUTPUT_DIR / "transfer_learning.svg")
    plt.close()


# flow = np.load("expe/transferlearning/transfer_learning.npz")
# Ncaps = flow['Ncaps']
# acc_lr_train_list = flow['acc_lr_train_list']
# acc_lr_test_list = flow['acc_lr_test_list']
# acc_lr_test_views_list = flow['acc_lr_test_views_list']
# plot_transfer_learning_learningCurve(Ncaps, acc_lr_train_list, acc_lr_test_list, acc_lr_test_views_list, OUTPUT_DIR)


### Confusion Matrices Viewer
def plot_cm(y_true, y_pred, title, suffix, OUTPUT_DIR):
    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    cm = confusion_matrix(y_true, y_pred)
    ## also print the per-class recall
    recall = np.diag(cm) / np.sum(cm, axis=1)
    print(f"Recall per class: {recall}")
    ## also print the per-class precision
    precision = np.diag(cm) / np.sum(cm, axis=0)
    print(f"Precision per class: {precision}")

    sns.heatmap(cm, annot=True, fmt="g", cmap="Blues", ax=ax, cbar=False)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    # Models rows, 3 Evaluation Columns
    # fig, axes = plt.subplots(len(models_y1), 3, figsize=(18, 5 * len(models_y1)))
    # if len(models_y1) == 1: axes = [axes]

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"Confusion_Matrices_{suffix}.png", dpi=300)
    plt.savefig(OUTPUT_DIR / f"Confusion_Matrices_{suffix}.pdf")
    # plt.show()
    plt.close()



def plot_soft_confusion_matrix(C_soft, suffix, OUTPUT_DIR, class_names=None, title="Soft confusion matrix (uniform PLL prior)"):
    """
    Plot the soft confusion matrix as a heatmap with row-normalised values
    overlaid on the raw soft counts.
    """
    n = C_soft.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    row_sums = C_soft.sum(axis=1, keepdims=True)
    C_norm = np.where(row_sums > 0, C_soft / row_sums, 0.0)   # row-normalised

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, data, fmt, label in zip(
        axes,
        [C_soft,  C_norm],
        [".1f",   ".2f"],
        ["Raw soft counts\n(expected #instances)", "Row-normalised\n(soft recall per class)"]
    ):
        im = ax.imshow(data, cmap="Blues", aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(n)); ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticks(range(n)); ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted label", fontsize=10)
        ax.set_ylabel("(Soft) true label", fontsize=10)
        ax.set_title(f"{title}\n{label}", fontsize=10)

        thresh = data.max() / 2.0
        for r in range(n):
            for c in range(n):
                val = data[r, c]
                if val > 0:
                    ax.text(c, r, f"{val:{fmt}}", ha="center", va="center",
                            fontsize=7, color="white" if val > thresh else "black")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"Confusion_Matrices_{suffix}.png", dpi=300)
    plt.savefig(OUTPUT_DIR / f"Confusion_Matrices_{suffix}.pdf")
    # plt.show()
    plt.close()
    # return fig

