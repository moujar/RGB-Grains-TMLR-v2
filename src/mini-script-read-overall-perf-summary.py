from pathlib import Path
import json
import numpy as np

VERBOSE_CHECKS = False
# VERBOSE_CHECKS = True

BASE_DIR = Path(".")
summary_json_path = BASE_DIR / "overall_perf_summary-7classes.json"
summary_list = []
if summary_json_path.exists():
    with open(summary_json_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                summary_list.append(json.loads(line))

splitChoices=[]
splitChoices.append("muPlot-1muTrain-1muTest-crossYears")
splitChoices.append("2muTrain-2muTest-yearGeneralization")
splitChoices.append("muPlot_year1only")
splitChoices.append("muPlot-2muTrain-2muTest_year")
splitChoices.append("muPlot-3muTrain-1muTest_year")
splitChoices.append("random_1muPlot")
splitChoices.append("random_year1only")
splitChoices.append("random_3muPlot")
splitChoices.append("random_4muPlot")

restricted_classes = [0,1]
# {
#   "experiment_short_name": "std50_muPlot-3muTrain-1muTest_year=all_restrClass=1_fold2",
#   "split_number": 2,
#   "num_epochs": 50,
#   "year": "all",
#   "num_classes": 8,
#   "test_acc": 0.78,
#   "val_acc": 0.782051282051282,
#   "train_acc": 0.859375,
#   "per_class_recall": [
#     0.8333333333333334,
#     0.7307692307692307
#   ],
#   "per_class_precision": [
#     0.7407407407407407,
#     0.8260869565217391
#   ],
#   "restricted_classes": [
#     5,
#     7
#   ]
# }
RestrictStringExplanation = ["MultiClass", "Binary Classif"]

for splitChoice in splitChoices:
    print(f"\n\nsplitChoice: {splitChoice}")
    for restrict in restricted_classes:

        print(f"\n     {RestrictStringExplanation[restrict]}")
        # std50_muPlot-3muTrain-1muTest_year=all_restrClass=1_fold2
        train_accs=[]
        val_accs=[]
        test_accs=[]
        for summary in summary_list:
            if splitChoice in summary["experiment_short_name"] \
                and f"restrClass={restrict}" in summary["experiment_short_name"]:
                    if VERBOSE_CHECKS:
                        print(f"        {summary['experiment_short_name']}")
                    train_accs.append(summary["train_acc"])
                    val_accs.append(summary["val_acc"])
                    test_accs.append(summary["test_acc"])
                    last_summary_retained = summary
        if len(train_accs) > 0:
                    
                        # print(f"{summary['experiment_short_name']} train_acc:{summary['train_acc']} val_acc:{summary['val_acc']} test_acc:{summary['test_acc']}")
            print(f"            experiment_short_name:{last_summary_retained['experiment_short_name'][:-6]}  Nfolds={len(train_accs)}")
            if VERBOSE_CHECKS: 
                print(f"train_accs:{train_accs}")
                print(f"val_accs:{val_accs}")
                print(f"test_accs:{test_accs}\n")
            ## use only 3 digits after the decimal point:
            print(f"            train_accs: {np.mean(train_accs):.3f} +/- {np.std(train_accs):.4f}")
            print(f"            val_accs mean:{np.mean(val_accs):.3f} +/- {np.std(val_accs):.4f}")
            print(f"            test_accs mean:{np.mean(test_accs):.3f} +/- {np.std(test_accs):.4f}")
