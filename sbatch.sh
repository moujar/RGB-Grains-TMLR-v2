#!/bin/sh
#SBATCH -v
#SBATCH -N 1
#SBATCH -c 5
#SBATCH --mem-per-cpu 30G
#SBATCH --gres=gpu:1

# EDIT ME: activate your own environment (see README.md "Installation").
# e.g. source /path/to/your/venv/bin/activate
source "$HOME/venvs/rgb-grains/bin/activate"

# splitChoice="muPlot_year1only"
# splitChoice="random_year1only"
# splitChoice="muPlot-2muTrain-2muTest"
# splitChoice="muPlot-3muTrain-1muTest"
# yearChosen=2020
# restClass=1
# restClass=0

for restClass in 0 1  ; do
    for yearChosen in 2020 2021; do
        for splitChoice in "muPlot_year1only" "random_year1only"; do
            echo "run  python -m rgb_grains.train --splitting-choice $splitChoice --yearChosen $yearChosen --restrict-classes $restClass"
            time python3 -m rgb_grains.train --config configs/serious.json --splitting-choice "$splitChoice" --yearChosen $yearChosen --restrict-classes $restClass
        done
    done
    for splitChoice in "muPlot-2muTrain-2muTest" "muPlot-3muTrain-1muTest"; do
        yearChosen="all"
        echo "run  python -m rgb_grains.train --splitting-choice $splitChoice --yearChosen $yearChosen --restrict-classes $restClass"
        time python3 -m rgb_grains.train --config configs/serious.json --splitting-choice "$splitChoice" --yearChosen $yearChosen --restrict-classes $restClass
    done
done

exit
