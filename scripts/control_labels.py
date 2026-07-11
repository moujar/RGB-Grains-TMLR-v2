#!/usr/bin/env python3
"""Sanity-check mixed-stand labels against a saved predictions.npz.

For each mix (and each microplot it appears in), plots the true variety
proportions against several proxies derived from the model's predicted
logits. Useful for catching mislabeled mixtures.

Usage:
    python scripts/control_labels.py --predictions-npz expe/<run>/predictions.npz --name-tag <run>
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import re

from rgb_grains.data.dataset import _load_mix_dict


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions-npz", required=True, help="Path to a predictions.npz saved by rgb_grains.train")
    p.add_argument("--name-tag", default="run", help="Short tag used in output filenames")
    p.add_argument("--output-dir", default="comparo_labels", help="Where to save the comparison plots")
    p.add_argument("--mix-csv", default=None, help="Defaults to the bundled rgb_grains/data/perfomix_mixtures.csv")
    p.add_argument("--num-classes", type=int, default=8)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mix_dict = _load_mix_dict(args.mix_csv)
    num_classes = args.num_classes

    test_data = np.load(args.predictions_npz)
    test_data_ids = test_data["ids"]
    logits_test_data = test_data["logits_test_data"]

    per_class_TP = np.zeros(num_classes, dtype=int)
    per_class_FP = np.zeros(num_classes, dtype=int)

    bins = np.linspace(0, 1, 101)
    for mixName in mix_dict.keys():
        masking_per_mix = np.array([mixName in test_data_ids[k] for k in range(len(test_data_ids))])
        if masking_per_mix.sum() == 0:
            continue
        ids = test_data_ids[masking_per_mix]
        microplots_list = []
        for id_ in ids:
            microplotname = "x00y00"
            microplotsearch = re.search(r"x\d{2}y\d{2}", id_)
            if microplotsearch:
                microplotname = microplotsearch.group(0)
            microplots_list.append(microplotname)
        microplots = np.unique(microplots_list)

        for microplot in microplots:
            masking_per_microplot = np.array([microplot in test_data_ids[k] for k in range(len(test_data_ids))])
            masking_per_mix_and_microplot = masking_per_mix * masking_per_microplot
            if masking_per_mix_and_microplot.sum() == 0:
                continue
            restricted_logits = logits_test_data[masking_per_mix_and_microplot]

            restricted_preds = restricted_logits.argmax(axis=1)

            integral_085_1_list = []
            restricted_preds_list = []
            restricted_logits_cumulated = []
            preds_per_class_above_threshold = []
            threshold = 0.91
            for c in range(num_classes):
                integral_085_1 = np.sum((restricted_logits[:, c] * (bins[1] - bins[0]))[85:])
                integral_085_1_list.append(integral_085_1)
                restricted_preds_list.append(np.sum(restricted_preds == c))
                restricted_logits_cumulated.append(np.sum(restricted_logits[:, c]))
                preds_per_class_above_threshold.append(np.sum(restricted_logits[:, c] > threshold))

            true_proportions = np.array([c in mix_dict[mixName] for c in range(num_classes)], dtype=float)
            true_proportions /= true_proportions.sum()
            integral_085_1_list = np.array(integral_085_1_list, dtype=float)
            integral_085_1_list /= integral_085_1_list.sum()
            restricted_preds_list = np.array(restricted_preds_list, dtype=float)
            restricted_preds_list /= restricted_preds_list.sum()
            restricted_logits_cumulated = np.array(restricted_logits_cumulated, dtype=float)
            restricted_logits_cumulated /= restricted_logits_cumulated.sum()
            preds_per_class_above_threshold = np.array(preds_per_class_above_threshold, dtype=float)
            preds_per_class_above_threshold /= np.sum(preds_per_class_above_threshold)

            expected_classes = mix_dict[mixName]
            TP = FP = 0
            for pred in restricted_preds:
                if pred in expected_classes:
                    TP += 1
                else:
                    FP += 1
            accuracy = TP / (TP + FP)

            for pred_class in restricted_preds:
                if pred_class in expected_classes:
                    per_class_TP[pred_class] += 1
                else:
                    per_class_FP[pred_class] += 1

            suffix = f"_acc={accuracy:.2f}_mix={mixName}_{microplot}_N={len(restricted_preds)}"
            plt.figure()
            plt.title(f"True proportions for mix {mixName}")
            plt.bar(range(num_classes), true_proportions, edgecolor="black", color="white")
            plt.plot(range(num_classes), integral_085_1_list, "--", label="logits integrated in [0.85, 1]")
            plt.plot(range(num_classes), restricted_preds_list, "-", label="predictions")
            plt.plot(range(num_classes), restricted_logits_cumulated, ":", label="cumulated logits")
            plt.plot(range(num_classes), preds_per_class_above_threshold, "-.", label=f"confident predictions (above {threshold})")
            plt.legend()
            plt.savefig(output_dir / f"{args.name_tag}_comparison_true_vs_preds_{suffix}.jpg")
            plt.close()

            projected_prediction = expected_classes[restricted_logits[:, expected_classes].argmax(1)]
            projected_proportions = np.bincount(projected_prediction, minlength=num_classes) / len(projected_prediction)
            print(f"{mixName}, {microplot}, N={len(restricted_logits)}: acc: {accuracy:.2f} proj. props: {np.round(projected_proportions, 2)}")

            plt.figure()
            plt.title(f"Projected proportions for mix {mixName}")
            plt.bar(range(num_classes), true_proportions, edgecolor="black", color=None, lw=2)
            plt.plot(range(num_classes), projected_proportions, color="green", lw=2)
            plt.savefig(output_dir / f"{args.name_tag}_comparison_projected_proportions_{suffix}.jpg")
            plt.close()

    print("\n" + "=" * 60)
    print("PER-CLASS ACCURACY (summed over all mixes)")
    print("=" * 60)

    per_class_accuracy = np.zeros(num_classes)
    per_class_total = np.zeros(num_classes)
    for c in range(num_classes):
        total_for_class = per_class_TP[c] + per_class_FP[c]
        if total_for_class > 0:
            per_class_accuracy[c] = per_class_TP[c] / total_for_class
            per_class_total[c] = total_for_class

    print(f"{'Class':<6} {'TP':<6} {'FP':<6} {'Total':<8} {'Accuracy':<10}")
    print("-" * 40)
    for c in range(num_classes):
        print(f"{c:<6} {per_class_TP[c]:<6} {per_class_FP[c]:<6} {per_class_total[c]:<8} {per_class_accuracy[c]:<10.3f}")
    print("-" * 40)
    denom = per_class_TP.sum() + per_class_FP.sum()
    overall_acc = per_class_TP.sum() / denom if denom > 0 else float("nan")
    print(f"Overall: {per_class_TP.sum():<6} {per_class_FP.sum():<6} {per_class_total.sum():<8} {overall_acc:<10.3f}")

    plt.figure(figsize=(10, 6))
    plt.bar(range(num_classes), per_class_accuracy, color="skyblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title(f"Per-Class Accuracy (summed over all mixes)\n{args.name_tag}")
    plt.xticks(range(num_classes))
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    for i, acc in enumerate(per_class_accuracy):
        plt.text(i, acc + 0.02, f"{acc:.3f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    out_path = output_dir / f"{args.name_tag}_per_class_accuracy_overall.jpg"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nPer-class accuracy plot saved to: {out_path}")


if __name__ == "__main__":
    main()
