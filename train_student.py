import argparse
import cv2
import os
import numpy as np
import torch

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


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader


models = []

def predict(model, dataloader):
    outputs = []
    model.eval()

    for batch_data in dataloader:
        image = batch_data['image']
        image = np.squeeze(image.numpy())
        image = np.moveaxis(image, 0, -1)
        heatmaps, pafs, scale, pad = demo.infer_fast(model, image, 256, 8, 4, False)
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(18):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * 8 / 4 - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * 8 / 4 - pad[0]) / scale

        num_keypoints = Pose.num_kpts
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                    print(f'Found keypoint {kpt_id}: {int(all_keypoints[int(pose_entries[n][kpt_id]), 0])}')
                    print(f'Found keypoint {kpt_id}: {int(all_keypoints[int(pose_entries[n][kpt_id]), 1])}')
                else:
                    print(f'No keypoint found for image')
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        for pose in current_poses:
            print(pose.keypoints)



    return outputs


epsilon = 0.2


def aggregated_teacher(models, dataloader, epsilon=0.2):
    """Aggregates teacher predictions for the student training data"""
    preds = torch.zeros((len(models), len(dataloader), 18, 2), dtype=torch.long)
    for i, model in enumerate(models):
        results = predict(model, dataloader)  # num_images x num_poses x 18 x 2
        preds[i] = results

    print('preds = ', preds)

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


