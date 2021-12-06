#!/bin/bash

#rsync -av --delete --exclude="__pycache__" ~/Desktop/code/Machine_learning/ltsm_stocks/ simon@G5.lan:/home/simon/ltsm_stocks/

rsync -av --delete G5:/home/simon/ltsm_stocks/results/ ~/Desktop/code/Machine_learning/ltsm_stocks/results/
rsync -av --delete --exclude={"results/*","__pycache__","output"} ~/Desktop/code/Machine_learning/ltsm_stocks/ G5:/home/simon/ltsm_stocks/

echo -e '\nCopying to Raspberry Pi:\n'
#rsync -av --delete --exclude="__pycache__" ~/Desktop/code/Machine_learning/ltsm_stocks/ rpi4:/mnt/hdd/ltsm_stocks/

#echo -e '\nCopying to G5:\n'
#rsync -av --delete --exclude="__pycache__" ~/Desktop/code/Machine_learning/ltsm_stocks/ simon@G5.lan:/home/simon/ltsm_stocks/
