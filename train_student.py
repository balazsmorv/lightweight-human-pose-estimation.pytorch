import argparse
import cv2
import os
import numpy as np
import torch
import pickle
import multiprocessing
import json
import shutil

from datasets.coco import CocoTrainDataset
from modules.keypoints import extract_keypoints, group_keypoints
from pate import perform_analysis_torch
from modules.pose import Pose
import demo
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
from train_pate import get_data_loaders
from torchvision import transforms
from datasets.transformations import ConvertKeypoints, Scale, Rotate, CropPad, Flip
from forward import run_demo


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader


models = []

epsilon = 0.2


def aggregated_teacher(models, dataloader, epsilon=0.2):
    """Aggregates teacher predictions for the student training data"""
    for i, model in enumerate(models):
        print(f'Run demo for model {i}')

        # num_pictures
        out = run_demo(model, '/home/oem/Dokumentumok/pose_estimation/lightweight-human-pose-estimation.pytorch/student_train_images', i)
        out_file = open(f"teacher_{i}.json", "w")
        json.dump(out, out_file, indent=6)

    ## itt kene a zajt hozzaadni


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints-folder', type=str, required=True, help='path to teacher checkpoint file folder')
    parser.add_argument('--prepared-train-labels', type=str, required=True,
                        help='path to the file with prepared annotations')
    parser.add_argument('--train-images-folder', type=str, required=True, help='path to COCO train images folder')
    args = parser.parse_args()

    print("GPU count: ", torch.cuda.device_count())
    print("Current GPU in use: ", torch.cuda.current_device())

    for filename in os.listdir(args.checkpoints_folder):
        f = os.path.join(args.checkpoints_folder, filename)
        if os.path.isfile(f):
            net = PoseEstimationWithMobileNet()
            checkpoint = torch.load(f)
            load_state(net, checkpoint)
            net = net.cuda()
            models.append(net)

    print(f'Found {len(models)} teacher model.')

    stride = 8
    sigma = 7
    path_thickness = 1
    train_set = CocoTrainDataset(args.prepared_train_labels,
                                 args.train_images_folder,
                                 stride, sigma, path_thickness,
                                 transform=transforms.Compose([
                                     ConvertKeypoints(),
                                     Scale(),
                                     Rotate(pad=(128, 128, 128)),
                                     CropPad(pad=(128, 128, 128)),
                                     Flip()]))

    train_loaders, student_train_loader, student_test_loader = get_data_loaders(train_set, 10, 160, 8)

    aggregated_teacher(models, student_train_loader)


