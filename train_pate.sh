#!/bin/bash

python train_pate.py --train-images-folder '/datadrive/coco/train2017' --prepared-train-labels '/datadrive/coco/prepared_train_annotation.pkl' --val-labels '/datadrive/coco/val_subset.json' --val-images-folder '/datadrive/coco/val2017' --checkpoint-path mobilenet_sgd_68.848.pth.tar --from-mobilenet --batch-size 160 --epoch-count 200 --experiment-name exp_pate --checkpoint-path '/home/morvayb/pose_estim/lightweight-human-pose-estimation.pytorch/mobilenet_sgd_68.848.pth.tar' --num-teachers 10
