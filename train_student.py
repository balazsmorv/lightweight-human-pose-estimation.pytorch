import argparse
import cv2
import os
import numpy as np
import torch
import pickle
import multiprocessing
import json
import shutil
import numpy
import scipy.cluster.hierarchy as hcluster

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
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader


models = []

epsilon = 0.2


# Make a training json for the student model. Aggregate the keypoints from the teachers
def make_student_train_json():
    # open the teacher prediction jsons
    jsons = []
    for i in range(10):
        with open(f'teacher_{i}.json', 'r') as openfile:
            json_object = json.load(openfile)
            jsons.append(json_object)
            openfile.close()

    for i in range(len(jsons[0]["images_preds"])): # for every image
        filename = jsons[0]["images_preds"][i]["filename"]

        max_num_poses = 0

        for j, teacher_json in enumerate(jsons): # for every teacher
            preds = teacher_json["images_preds"][i]["preds"]
            num_poses = len(preds)
            if num_poses > max_num_poses:
                max_num_poses = num_poses

        predictions = torch.zeros((max_num_poses, 10, 3*18), dtype=torch.int)
        categorized_predictions = torch.zeros((max_num_poses, 10, 3*18), dtype=torch.int)

        for j, teacher_json in enumerate(jsons): # for every teacher
            for k, pred in enumerate(teacher_json["images_preds"][i]["preds"]):
                predictions[k][j] = torch.tensor(pred["kpts"], dtype=torch.int)
        print(f'predictions for {filename}: {predictions.shape}')







def aggregated_teacher(models, dataloader, epsilon=0.2):
    """Aggregates teacher predictions for the student training data"""
    for i, model in enumerate(models):
        print(f'Run demo for model {i}')

        # num_pictures
        out = run_demo(model, '/home/oem/Dokumentumok/pose_estimation/lightweight-human-pose-estimation.pytorch/student_train_images', i)
        out_file = open(f"teacher_{i}.json", "w")
        json.dump(out, out_file, indent=6)




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

    #aggregated_teacher(models, student_train_loader)
    make_student_train_json()

