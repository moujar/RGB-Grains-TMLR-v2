#!/usr/bin/env python3
"""Aggregate overall_perf_summary.json (one JSON line per run of rgb_grains.train)
grouped by splitting-choice and restrict-classes, printing mean +/- std of
train/val/test accuracy per configuration.

Usage:
    python scripts/read_perf_summary.py --summary-json overall_perf_summary.json
"""
import argparse
from pathlib import Path
import json
import numpy as np

DEFAULT_SPLIT_CHOICES = [
    "muPlot-1muTrain-1muTest-crossYears",
    "muPlot-2muTrain-2muTest-yearGeneralization_year1only",
    "muPlot_year1only",
    "muPlot-2muTrain-2muTest",
    "muPlot-3muTrain-1muTest",
    "random_1muPlot",
    "random_year1only",
    "random_3muPlot",
    "random_4muPlot",
    "bacs_2train_1test",
]
RESTRICT_LABELS = ["MultiClass", "Restricted classes"]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-json", type=str, default="overall_perf_summary.json",
                    help="Path to the JSONL performance summary produced by rgb_grains.train.")
    p.add_argument("--split-choices", nargs="+", default=DEFAULT_SPLIT_CHOICES,
                    help="Substrings of experiment_short_name to group by.")
    p.add_argument("-v", "--verbose", action="store_true", help="Print per-run details.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary_path = Path(args.summary_json)
    summary_list = []
    if summary_path.exists():
        with open(summary_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    summary_list.append(json.loads(line))
    else:
        print(f"No summary file found at {summary_path}")
        return

    for split_choice in args.split_choices:
        print(f"\n\nsplitChoice: {split_choice}")
        for restrict in (0, 1):
            print(f"\n     {RESTRICT_LABELS[restrict]}")
            train_accs, val_bal_accs, test_accs = [], [], []
            last_summary = None
            for summary in summary_list:
                name = summary.get("experiment_short_name", "")
                if split_choice in name and f"rstCls={restrict}" in name:
                    if args.verbose:
                        print(f"        {name}")
                    train_accs.append(summary.get("train_acc"))
                    val_bal_accs.append(summary.get("selected_val_bal_acc"))
                    test_accs.append(summary.get("test_bal_acc"))
                    last_summary = summary
            if last_summary is not None:
                print(f"            experiment_short_name:{last_summary['experiment_short_name']}  Nfolds={len(train_accs)}")
                if args.verbose:
                    print(f"train_accs:{train_accs}")
                    print(f"selected_val_bal_accs:{val_bal_accs}")
                    print(f"test_bal_accs:{test_accs}\n")
                print(f"            train_acc:               {np.nanmean(train_accs):.3f} +/- {np.nanstd(train_accs):.4f}")
                print(f"            selected_val_bal_acc:    {np.nanmean(val_bal_accs):.3f} +/- {np.nanstd(val_bal_accs):.4f}")
                print(f"            test_bal_acc:             {np.nanmean(test_accs):.3f} +/- {np.nanstd(test_accs):.4f}")


if __name__ == "__main__":
    main()
