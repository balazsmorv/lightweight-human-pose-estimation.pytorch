#!/bin/bash

python train.py --train-images-folder '/home/oem/Dokumentumok/AlphaPose/data/coco/train2017' --prepared-train-labels prepared_train_annotation.pkl --val-labels val_subset.json --val-images-folder '/home/oem/Dokumentumok/AlphaPose/data/coco/val2017' --checkpoint-path '/home/oem/Dokumentumok/lightweight-human-pose-estimation.pytorch/mobilenet_sgd_68.848.pth.tar' --from-mobilenet --batch-size 20 --epoch-count 2 --experiment-name exp2
