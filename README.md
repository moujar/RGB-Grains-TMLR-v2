5. TODOs:
To Oudoum & the M1 team, or everyone in general, there is a number of things I would like to have, based on my newer version of the model's code:

5.1 Easy fixes:
(urgent) code wise: clean up the best model recording. There is some confusion between (1) best val acc, (2) best val acc + SWA, (3) last model. At some point the code was averaing logits of (1) and (2). I am not sure I cleaned this up properly. In the end, only the logits of one of them should be used, prfereably the ones of best val acc+SWA since that point, or simply, best bal acc (since SWA did not seem to play a major role in our case)
(easy) display some examples of inputs after the data augmentation (imshow ~100 augmented grains)
(easy) display some examples of failed predictions (imshow the original grains)
(easy) confusion matrices: also display per-class recall and per-class precision on the sides (one horizontally, one vertically, make the sensible choice)
(relatively easy) add computation of train balanced accuracy on the best model, using minimal augmentations (32 views) only, to assess overfitting (this bal acc should be higher than the validation one, hopefully not a lot)
5.2 Mengtsu et al dataset
(big work, worth it) adapt the code pipeline to apply to the data of Mengtsu et al. I downloaded their files (had to click them 1 by 1). I will share that next week.
build appropriate dataset mode (or new dataset class, as you prefer)
run the model on it (I can do the long runs)
report results
This is quite a nice added value for resubmission to TMLR, esp. if we can show that fine-tuning+data-aug allow us to improve over the results of Mengtsu et al, or do as well at much smaller compute cost. I expect it will.
5.2.bis [optional but desirable] Martin's dataset (SCOOP BACS)
(relatively easy) add a mode to handle that data. Split by microplot, again. There are 3 bacs by species. As Martin said:
Les grains viennent de 4 variétés, qui sont répétées dans 3 bacs chacun (équivalent à 3 micro-parcelles différentes), toujours en culture pure.
EL4X-199: Bacs: 14, 18 et 43
EL4X-35: Bacs: 17, 71 et 74
EL4X-482: Bacs: 41,42, 50
GQ4X-83: Bacs: 7, 57, 92
so, it's K=4 class classification, one has to split by bac (~microplot).
take 2 bacs for train and 1 for test. You can cross validate on 3 folds (choice of which bac is the test one). Ignore the combinatorics of choosing the test bac for each species (like I did for the perfomix data)
this is basically what is written in the case if splitting_choice == "muPlot_year1only":
one has only to edit the beggining of load_datasets_microplot_split() and use that mode, "muPlot_year1only", maybe hacking yearChosen so it works too. There is no mixed data in that dataset, but it's of, df_mixed can be empty and things will work fine
5.3 [optional but desirable] Downsampling image resolution
(relatively easy) deliberately reduce the resolution of input images, during the dataloading (with either meanpool or maxpool, using increasing 2x2, 3x3, 4x4, etc pooling kernel sizes), to test how much accuracy depends on resolution (related to Phuoc's task)
implement a couple of downresolution (meanpool/maxpool, and kernel size as argument, so it can be controlled when runnning the script)
run the fine-tuning+test (either collab, or I can do it if you coded well) for each downresolution (starting from no downresolution at all, for control)
if the downresolution tests still provide good accuracy, then it's worth asking Timothée to take nice resolution RGB pictures with some decent (not HSI) camera.
after we manage to segment these RGB grains (low res or prefereably higher resolution), esp. if the downresolution tests are positive:
run the model on those (adapt dataloader to this new "dataset"). Not too much hope to get good results there.
5.4 [very optional] Factoring the code better
(Optional, can become big work): refactoring the code to make it much cleaner & professional: less code duplication, only one model (convnext), no hard coding data augs when they are available by torch, etc.
5.5 [very optional] Data cleaning
(Optional, can become big work): data cleaning: there is sometimes a bit of dust or other artifacts of data collection, that are seen as grains by watershed, and then used for train or test. We should get rid of those. (not a lot, really). Ideally, with some clean way to pre-process the data, (area of non-zero values, etc) or by hand (feeding an exclusion list):
(relatively easy) by size (areas of acitve pixels less than a threshold -> put in the exclusion list/ or directly moved from the _processed folder to a _excluded folder)
BACS: a rather simple criterion is when np.sum(img>0.01) < 110000, then it's so small we can remove it with little loss (some half grains will be lost, it's ok)
perfomix: to be studied.
(easy but time consuming) Or manually with a simple tagging (user deletes anomalous jpgs, the collects the jpg paths, that are then converted to npz paths, which would feed an exclusion list)
(bigger work) Or by OD (Outlier Detection) -> remove all outliers (no labels used)