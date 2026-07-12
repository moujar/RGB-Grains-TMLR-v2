import numpy as np
import matplotlib.pyplot as plt
import argparse
import re

from dataset import _variety_to_num
from dataset import _load_mix_dict

'''
This script is used to control the labels of the dataset.
It plots the logits for each mix and compares them to the true proportions.
This is useful to check the mix are indeed what we expect them to be.... 
'''

# input_file = "logits_mix=mix1.npz"
input_file = "expe/newBds-50_muPlot-3muTrain-1muTest_year=all_restrClass=0_fold3/ConvNeXt-Tiny_predictions_newBds-50_muPlot-3muTrain-1muTest_year=all_restrClass=0_fold3.npz"
name_tag="3muTrain-1muTest_year=all_restrClass=0_fold3"

# input_file = "expe/newBds-50_inferenceMode_4muTrain-0muTest_year=all_restrClass=0_fold0/ConvNeXt-Tiny_predictions_newBds-50_inferenceMode_4muTrain-0muTest_year=all_restrClass=0_fold0.npz"
# name_tag="inferenceMode_4muTrain-0muTest"



mix_dict = _load_mix_dict()

num_classes = 8

test_data = np.load(input_file)
test_data_ids = test_data["ids"]
logits_test_data = test_data["logits_test_data"]
true_y_test = test_data["true_y_test"]

per_class_TP = np.zeros(num_classes, dtype=int)
per_class_FP = np.zeros(num_classes, dtype=int)

bins = np.linspace(0,1,101)
for i, mixName in enumerate(mix_dict.keys()):
    masking_per_mix = np.array([mixName in test_data_ids[k] for k in range(len(test_data_ids))])
    if masking_per_mix.sum() == 0 :
        continue
    ids = test_data_ids[masking_per_mix]
    ## detect which microplot are in these ids: search with a regexp x82y19 and similar things in the ids:
    microplots_list = []
    for id in ids:
        microplotname="x00y00"
        microplotsearch = re.search(r"x\d{2}y\d{2}", id)
        if microplotsearch:
            microplotname = microplotsearch.group(0)
        microplots_list.append(microplotname)
    microplots = np.unique(microplots_list)
    # print(f"Microplots in mix {mixName}: {microplots}")

    for microplot in microplots:
        masking_per_microplot =   np.array([microplot in test_data_ids[k] for k in range(len(test_data_ids))])
        masking_per_mix_and_microplot = masking_per_mix * masking_per_microplot  
        if masking_per_mix_and_microplot.sum() == 0 :
            continue 
        else:
            restricted_logits = logits_test_data[masking_per_mix_and_microplot]
    
        # ## a control plot, too detial, not very useful:
        
        # plt.figure()
        # plt.title(f"Logits for mix {mixName}")
        # plt.xlim(0.85, 1)
        # plt.ylim(0, 10)
        # for c in range(num_classes):
        #     counts, bs = np.histogram(restricted_logits[:,c], bins=bins, density=True)
        #     plt.plot(bs[1:], counts, label=f"class {c}")
        # print(f"Mix is: {mixName}")
        # plt.legend()
        # plt.xlabel("logit value")
        # plt.ylabel("density")
        # plt.savefig(f"../comparo_labels/{name_tag}_logits_mix={mixName}.png") 
        # # plt.show() 
        # plt.close()

        restricted_preds = restricted_logits.argmax(axis=1)

        ## compute integral of density curves between 0.85 and 1:
        integral_085_1_list=[]
        preds_per_class = []
        preds_per_class_above_threshold = []
        threshold = 0.91
        restricted_logits_cumulated =[]
        restricted_preds_list = []
        for c in range(num_classes):
            integral_085_1 = np.sum( (restricted_logits[:,c] * (bins[1] - bins[0]))[85:] )
            integral_085_1_list.append(integral_085_1)
            restricted_preds_list.append(np.sum(restricted_preds == c))
            restricted_logits_cumulated.append(np.sum(restricted_logits[:,c]))
            preds_per_class_above_threshold.append(np.sum(restricted_logits[:,c] > threshold))

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

        ## make a binary accuracy metric:
        expected_classes = mix_dict[mixName]
        TP = 0 
        FP = 0
        for sample in range(len(restricted_preds)):
            if restricted_preds[sample] in expected_classes:
                ## if the predicted class is in one of the expected ones, that's a true (positive).
                TP +=1
            else:
                ## if pred class is not in expected ones, that's a false (negative).
                FP +=1
        accuracy = TP / (TP + FP)
        # print(f"Accuracy for mix {mixName}: {accuracy:.2f}")

        ## Accumulate per-class TP and FP across mixes
        # Count per-class true positives and false negatives
        for sample_idx in range(len(restricted_preds)):
            pred_class = restricted_preds[sample_idx]
            if pred_class in expected_classes:
                # True positive: predicted class is one of the expected classes
                per_class_TP[pred_class] += 1
            else:
                # False negative: predicted class is not in expected classes
                # Count FP for each expected class that was missed
                per_class_FP[pred_class] += 1

        # for c in range(num_classes):  
        #     print(f"Class {c}: {integral_085_1_list[c]} integral between 0.85 and 1: ")
        #     print(f"Class {c}: {restricted_preds_list[c]} predictions are of class {c}")
        #     print(f"Class {c}: {restricted_logits_cumulated[c]} cumulated logit")
        #     print(f"Class {c}: {preds_per_class_above_threshold[c]} very confident (logit>{threshold}) predictions are of class {c}")
        #     print(f"Class {c}: true proportion: {true_proportions[c]}")

        suffix = f"_acc={accuracy:.2f}_mix={mixName}_{microplot}_N={len(restricted_preds)}"
        # print(f"True proportions for mix {mixName}: {true_proportions}")
        ##TODO: plot a sort of bar chart with the true proportions in bold black bars:
        plt.figure()
        plt.title(f"True proportions for mix {mixName}")
        plt.bar(range(num_classes), true_proportions, edgecolor='black', color='white')
        plt.plot(range(num_classes), integral_085_1_list, '--', label='logits integrated in [0.85, 1]')
        plt.plot(range(num_classes), restricted_preds_list, '-', label='predictions')
        plt.plot(range(num_classes), restricted_logits_cumulated, ':', label='cumulated logits')
        plt.plot(range(num_classes), preds_per_class_above_threshold, '-.', label=f'confident predictions (above {threshold})')
        plt.legend()
        plt.savefig(f"../comparo_labels/{name_tag}_comparison_true_vs_preds_{suffix}.jpg") 
        # plt.show()
        plt.close()


        ## estimated proportions: we project only to the expected classes.
        projected_prediction = expected_classes[restricted_logits[:,expected_classes].argmax(1)]
        ## compare the expected proportion with the projected proportion
        projected_proportions = np.bincount(projected_prediction, minlength=num_classes) / len(projected_prediction)
        # print(f"Projected proportions for mix {mixName}: {projected_proportions}")
        
        print(f"{mixName}, {microplot}, N={len(restricted_logits)}: acc: {accuracy:.2f} proj. props: {np.round(projected_proportions, 2)}")

        ## bar chart: 
        plt.figure()
        plt.title(f"Projected proportions for mix {mixName}")
        plt.bar(range(num_classes), true_proportions, edgecolor='black', color=None, lw=2)
        plt.plot(range(num_classes), projected_proportions,  color='green', lw=2)
        plt.savefig(f"../comparo_labels/{name_tag}_comparison_projected_proportions_{suffix}.jpg") 
        # plt.show()
        plt.close()

        # ## parting between class 1 and 7 (almost tie):
        # print(f"Logits comaprison: Class 1 > Class 7: {np.sum(restricted_logits[:,1] > restricted_logits[:,7])} predictions")
        # print(f"Logits comaprison: Class 1 < Class 7: {np.sum(restricted_logits[:,1] < restricted_logits[:,7])} predictions")
        # ## same comparison presented as confidence matrix for all pairs:
        # conf_matrix = np.zeros((num_classes, num_classes))
        # for i in range(num_classes):
        #     for j in range(num_classes):
        #         conf_matrix[i,j] = np.sum(restricted_logits[:,i] > restricted_logits[:,j])
        #     ## diagonal represents when the logit is the alrgest:
        #     conf_matrix[i,i] = np.sum(restricted_logits[:,i] >= restricted_logits.max(axis=1))
        # print(f"Confidence matrix: \n{conf_matrix}")
        # plt.figure()
        # plt.imshow(conf_matrix)
        # plt.colorbar()
        # plt.xlabel("class i")
        # plt.ylabel("class j (less likely)")
        # plt.savefig(f"confidence_matrix_mix={mixName}.png")
        # plt.show()

        # ## confidence levels:
        # plt.figure()
        # plt.hist(restricted_logits.max(axis=1), bins=50)
        # # plt.savefig(f"confidence_levels_mix={mixName}.png")
        # plt.show()

## Calculate and display per-class accuracy across all mixes
print("\n" + "="*60)
print("PER-CLASS ACCURACY (summed over all mixes)")
print("="*60)

# Calculate per-class accuracy: TP / (TP + FP) for each class
per_class_accuracy = np.zeros(num_classes)
per_class_total = np.zeros(num_classes)

for c in range(num_classes):
    total_for_class = per_class_TP[c] + per_class_FP[c]
    if total_for_class > 0:
        per_class_accuracy[c] = per_class_TP[c] / total_for_class
        per_class_total[c] = total_for_class
    else:
        per_class_accuracy[c] = 0.0
        per_class_total[c] = 0

# Display results
print(f"{'Class':<6} {'TP':<6} {'FP':<6} {'Total':<8} {'Accuracy':<10}")
print("-" * 40)
for c in range(num_classes):
    print(f"{c:<6} {per_class_TP[c]:<6} {per_class_FP[c]:<6} {per_class_total[c]:<8} {per_class_accuracy[c]:<10.3f}")

print("-" * 40)
print(f"Overall: {per_class_TP.sum():<6} {per_class_FP.sum():<6} {per_class_total.sum():<8} {per_class_TP.sum()/(per_class_TP.sum()+per_class_FP.sum()):<10.3f}")

# Create bar plot of per-class accuracy
plt.figure(figsize=(10, 6))
plt.bar(range(num_classes), per_class_accuracy, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.title(f'Per-Class Accuracy (summed over all mixes)\n{name_tag}')
plt.xticks(range(num_classes))
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, acc in enumerate(per_class_accuracy):
    plt.text(i, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(f"../comparo_labels/{name_tag}_per_class_accuracy_overall.jpg", dpi=150)
# plt.show()
plt.close()

print(f"\nPer-class accuracy plot saved to: ../comparo_labels/{name_tag}_per_class_accuracy_overall.jpg")
