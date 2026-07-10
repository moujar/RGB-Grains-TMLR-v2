import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from pathlib import Path


def plot_augmentation_examples(
    train_data,
    output_dir,
    crop_size=176,
    n_grains=5,
    n_augmentations=9,
    seed=42,
):
    """Plot original center crops and independent random augmentations."""
    from dataset import GrainDataset_ConvNeXt, IMGNET_MEAN, IMGNET_STD

    X = train_data["X"]
    y = np.asarray(train_data["y"])
    if len(X) == 0:
        raise ValueError("Cannot plot augmentations from an empty training set")

    rng = np.random.RandomState(seed)
    pure_indices = np.where((y > 0).sum(axis=1) == 1)[0]
    selected = []
    if len(pure_indices) > 0:
        pure_classes = y[pure_indices].argmax(axis=1)
        for class_id in rng.permutation(np.unique(pure_classes)):
            candidates = pure_indices[pure_classes == class_id]
            selected.append(int(rng.choice(candidates)))
            if len(selected) == n_grains:
                break

    remaining = np.setdiff1d(np.arange(len(X)), np.asarray(selected, dtype=int))
    if len(selected) < n_grains and len(remaining) > 0:
        extra_count = min(n_grains - len(selected), len(remaining))
        selected.extend(rng.choice(remaining, size=extra_count, replace=False).tolist())

    n_grains = len(selected)
    X_selected = X[selected]
    y_selected = y[selected]
    original_ds = GrainDataset_ConvNeXt(
        X_selected, y=None, augment=False, crop_size=crop_size
    )
    augmented_ds = GrainDataset_ConvNeXt(
        X_selected, y=None, augment=True, crop_size=crop_size
    )
    mean = np.asarray(IMGNET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(IMGNET_STD, dtype=np.float32).reshape(3, 1, 1)

    def display_image(tensor):
        image = tensor.detach().cpu().numpy() * std + mean
        return np.clip(image.transpose(1, 2, 0), 0.0, 1.0)

    rows = n_augmentations + 1
    fig, axes = plt.subplots(
        rows, n_grains, figsize=(2.1 * n_grains, 2.1 * rows), squeeze=False
    )
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for col, class_targets in enumerate(y_selected):
            class_text = ",".join(map(str, np.where(class_targets > 0)[0]))
            axes[0, col].imshow(display_image(original_ds[col]))
            axes[0, col].set_title(f"Original\nclass {class_text}", fontsize=9)
            for row in range(1, rows):
                axes[row, col].imshow(display_image(augmented_ds[col]))

    for row in range(rows):
        axes[row, 0].set_ylabel(
            "Original" if row == 0 else f"Augmentation {row}", fontsize=8
        )
        for col in range(n_grains):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.suptitle("Training Data Augmentation Examples", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    output_dir = Path(output_dir)
    jpg_path = output_dir / "augmentation_examples.jpg"
    pdf_path = output_dir / "augmentation_examples.pdf"
    fig.savefig(jpg_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return jpg_path


def plot_failed_predictions(
    test_data,
    prediction_scores,
    output_dir,
    max_examples=25,
    columns=5,
):
    """Plot the most confident incorrect predictions using original grains."""
    from dataset import CH_SCALE

    X = test_data["X"]
    true_y = np.asarray(test_data["y"])
    ids = np.asarray(test_data.get("ids", [""] * len(X)))
    scores = np.asarray(prediction_scores)
    predicted = scores.argmax(axis=1)
    confidence = scores.max(axis=1)

    if not (len(X) == len(true_y) == len(scores)):
        raise ValueError("Test images, labels, and predictions must have equal lengths")

    support = true_y > 0
    correct = support[np.arange(len(predicted)), predicted]
    failed_indices = np.where(~correct)[0]
    if len(failed_indices) == 0:
        return None

    failed_indices = failed_indices[
        np.argsort(confidence[failed_indices])[::-1]
    ][:max_examples]
    rows = int(np.ceil(len(failed_indices) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(3.2 * columns, 3.5 * rows), squeeze=False
    )
    channel_scale = np.asarray(CH_SCALE, dtype=np.float32).reshape(1, 1, 3)

    for ax, index in zip(axes.flat, failed_indices):
        image = np.asarray(X[index], dtype=np.float32)
        image = np.clip(image / channel_scale, 0.0, 1.0)
        allowed = ",".join(map(str, np.where(support[index])[0]))
        sample_name = Path(str(ids[index])).stem if index < len(ids) else ""
        if len(sample_name) > 28:
            sample_name = sample_name[:25] + "..."
        ax.imshow(image)
        ax.set_title(
            f"true: {allowed} | pred: {predicted[index]}\n"
            f"confidence: {confidence[index]:.3f}\n{sample_name}",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes.flat[len(failed_indices):]:
        ax.axis("off")

    fig.suptitle(
        f"Most Confident Failed Predictions ({len(failed_indices)} shown)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.0)
    output_dir = Path(output_dir)
    jpg_path = output_dir / "failed_predictions.jpg"
    pdf_path = output_dir / "failed_predictions.pdf"
    fig.savefig(jpg_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return jpg_path


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
    fig1.savefig(OUTPUT_DIR / "calibration_reliability.png", dpi=300, bbox_inches="tight")
    fig1.savefig(OUTPUT_DIR / "calibration_reliability.pdf", bbox_inches="tight")
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
    fig2.savefig(OUTPUT_DIR / "calibration_histogram.png", dpi=300, bbox_inches="tight")
    fig2.savefig(OUTPUT_DIR / "calibration_histogram.pdf", bbox_inches="tight")
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
def _confusion_metrics(cm):
    diagonal = np.diag(cm)
    recall = np.divide(
        diagonal, cm.sum(axis=1), out=np.zeros_like(diagonal, dtype=float),
        where=cm.sum(axis=1) > 0,
    )
    precision = np.divide(
        diagonal, cm.sum(axis=0), out=np.zeros_like(diagonal, dtype=float),
        where=cm.sum(axis=0) > 0,
    )
    return precision, recall


def _plot_confusion_panel(
    fig,
    subplot_spec,
    matrix,
    metrics_matrix,
    class_names,
    title,
    value_format,
    true_label="True label",
):
    """Draw a confusion matrix with recall on the right and precision below."""
    precision, recall = _confusion_metrics(metrics_matrix)
    grid = subplot_spec.subgridspec(
        2, 2, width_ratios=(8, 1.25), height_ratios=(8, 1.25),
        wspace=0.05, hspace=0.05,
    )
    ax_main = fig.add_subplot(grid[0, 0])
    ax_recall = fig.add_subplot(grid[0, 1])
    ax_precision = fig.add_subplot(grid[1, 0])
    ax_empty = fig.add_subplot(grid[1, 1])
    ax_empty.axis("off")

    sns.heatmap(
        matrix, annot=True, fmt=value_format, cmap="Blues", cbar=False,
        xticklabels=False, yticklabels=class_names, ax=ax_main,
    )
    ax_main.set_title(title, fontsize=11, pad=10)
    ax_main.set_ylabel(true_label)
    ax_main.set_xlabel("")

    sns.heatmap(
        recall[:, None], annot=True, fmt=".2f", cmap="Greens", cbar=False,
        vmin=0.0, vmax=1.0, xticklabels=False, yticklabels=False,
        ax=ax_recall,
    )
    ax_recall.set_title("Recall", fontsize=10, pad=10)
    ax_recall.set_ylabel("")

    sns.heatmap(
        precision[None, :], annot=True, fmt=".2f", cmap="Greens", cbar=False,
        vmin=0.0, vmax=1.0, xticklabels=class_names,
        yticklabels=["Precision"], ax=ax_precision,
    )
    ax_precision.set_xlabel("Predicted label")
    ax_precision.tick_params(axis="y", rotation=0)
    return precision, recall


def plot_cm(y_true, y_pred, title, suffix, OUTPUT_DIR, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    class_names = [str(i) for i in range(cm.shape[0])]
    fig = plt.figure(figsize=(7, 7))
    outer_grid = fig.add_gridspec(1, 1)
    precision, recall = _plot_confusion_panel(
        fig, outer_grid[0], cm, cm, class_names, title, "g"
    )
    print(f"Recall per class: {recall}")
    print(f"Precision per class: {precision}")

    fig.savefig(OUTPUT_DIR / f"Confusion_Matrices_{suffix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"Confusion_Matrices_{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)



def plot_soft_confusion_matrix(C_soft, suffix, OUTPUT_DIR, class_names=None, title="Soft confusion matrix (uniform PLL prior)"):
    """
    Plot the soft confusion matrix as a heatmap with row-normalised values
    overlaid on the raw soft counts.
    """
    n = C_soft.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    row_sums = C_soft.sum(axis=1, keepdims=True)
    C_norm = np.divide(
        C_soft, row_sums, out=np.zeros_like(C_soft, dtype=float), where=row_sums > 0
    )

    fig = plt.figure(figsize=(16, 7))
    outer_grid = fig.add_gridspec(1, 2, wspace=0.25)

    for subplot_spec, data, fmt, label in zip(
        outer_grid,
        [C_soft,  C_norm],
        [".1f", ".2f"],
        ["Raw soft counts\n(expected #instances)", "Row-normalised\n(soft recall per class)"]
    ):
        _plot_confusion_panel(
            fig,
            subplot_spec,
            data,
            C_soft,
            class_names,
            f"{title}\n{label}",
            fmt,
            true_label="(Soft) true label",
        )

    fig.savefig(OUTPUT_DIR / f"Confusion_Matrices_{suffix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"Confusion_Matrices_{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)

