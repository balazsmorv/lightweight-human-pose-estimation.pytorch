#!/bin/bash

python train_pate.py --train-images-folder '/home/oem/Dokumentumok/AlphaPose/data/coco/train2017' --prepared-train-labels '/home/oem/Dokumentumok/pose_estimation/lightweight-human-pose-estimation.pytorch/prepared_train_annotation.pkl' --val-labels '/home/oem/Dokumentumok/pose_estimation/lightweight-human-pose-estimation.pytorch/val_subset.json' --val-images-folder '/home/oem/Dokumentumok/AlphaPose/data/coco/val2017' --checkpoint-path mobilenet_sgd_68.848.pth.tar --from-mobilenet --batch-size 160 --epoch-count 200 --experiment-name exp_pate_DELETE --checkpoint-path '/home' --num-teachers 10