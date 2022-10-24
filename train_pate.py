import argparse
import cv2
import os

import numpy as np
import torch
from torch.nn import DataParallel
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from datasets.coco import CocoTrainDataset
from datasets.transformations import ConvertKeypoints, Scale, Rotate, CropPad, Flip
from modules.get_parameters import get_parameters_conv, get_parameters_bn, get_parameters_conv_depthwise
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.loss import l2_loss
from modules.load_state import load_state, load_from_mobilenet
from val import evaluate
from modules.keypoints import extract_keypoints, group_keypoints
from pate import perform_analysis_torch
import demo

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader

def get_data_loaders(train_data, num_teachers, batch_size, num_workers):
    """ Function to create data loaders for the Teacher classifier """
    teacher_loaders = []
    #data_size = len(train_data) // (num_teachers + 2)
    data_size = 100

    for i in range(num_teachers):
        indices = list(range(i * data_size, (i + 1) * data_size))
        subset_data = Subset(train_data, indices)
        print("data subset size = ", len(subset_data))
        loader = torch.utils.data.DataLoader(subset_data, batch_size=batch_size, num_workers=num_workers)
        teacher_loaders.append(loader)

    student_train_indices = list(range(num_teachers * data_size, (num_teachers + 1) * data_size))
    subset = Subset(train_data, student_train_indices)
    student_train_loader = torch.utils.data.DataLoader(subset, batch_size=1, num_workers=num_workers)

    student_test_indices = list(range((num_teachers + 1) * data_size, (num_teachers + 2) * data_size))
    subset = Subset(train_data, student_test_indices)
    student_test_loader = torch.utils.data.DataLoader(subset, batch_size=1, num_workers=num_workers)

    return teacher_loaders, student_train_loader, student_test_loader


def train(prepared_train_labels, train_images_folder, num_refinement_stages, base_lr, batch_size, batches_per_iter,
          num_workers, checkpoint_path, weights_only, from_mobilenet, checkpoints_folder, log_after,
          val_labels, val_images_folder, val_output_name, checkpoint_after, val_after, writer, num_teachers):

    stride = 8
    sigma = 7
    path_thickness = 1
    train_set = CocoTrainDataset(prepared_train_labels, train_images_folder,
                               stride, sigma, path_thickness,
                               transform=transforms.Compose([
                                   ConvertKeypoints(),
                                   Scale(),
                                   Rotate(pad=(128, 128, 128)),
                                   CropPad(pad=(128, 128, 128)),
                                   Flip()]))


    train_loaders, student_train_loader, student_test_loader = get_data_loaders(train_set, num_teachers, batch_size, num_workers)
    writer.add_scalar('Batch size', batch_size)

    models = []
    for i in range(num_teachers):
        print("teacher ", i)
        net = PoseEstimationWithMobileNet(num_refinement_stages)
        train_loader = train_loaders[i]

        optimizer = optim.Adam([
            {'params': get_parameters_conv(net.model, 'weight')},
            {'params': get_parameters_conv_depthwise(net.model, 'weight'), 'weight_decay': 0},
            {'params': get_parameters_bn(net.model, 'weight'), 'weight_decay': 0},
            {'params': get_parameters_bn(net.model, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
            {'params': get_parameters_conv(net.cpm, 'weight'), 'lr': base_lr},
            {'params': get_parameters_conv(net.cpm, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
            {'params': get_parameters_conv_depthwise(net.cpm, 'weight'), 'weight_decay': 0},
            {'params': get_parameters_conv(net.initial_stage, 'weight'), 'lr': base_lr},
            {'params': get_parameters_conv(net.initial_stage, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
            {'params': get_parameters_conv(net.refinement_stages, 'weight'), 'lr': base_lr * 4},
            {'params': get_parameters_conv(net.refinement_stages, 'bias'), 'lr': base_lr * 8, 'weight_decay': 0},
            {'params': get_parameters_bn(net.refinement_stages, 'weight'), 'weight_decay': 0},
            {'params': get_parameters_bn(net.refinement_stages, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        ], lr=base_lr, weight_decay=5e-4)


        num_iter = 0
        current_epoch = 0
        drop_after_epoch = [100, 200, 260]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=drop_after_epoch, gamma=0.333)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)

            if from_mobilenet:
                load_from_mobilenet(net, checkpoint)
            else:
                load_state(net, checkpoint)
                if not weights_only:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    num_iter = checkpoint['iter']
                    current_epoch = checkpoint['current_epoch']

        net = DataParallel(net).cuda()
        net.train()
        for epochId in range(current_epoch, 2):
            print("epoch ", epochId)
            scheduler.step()
            total_losses = [0, 0] * (num_refinement_stages + 1)  # heatmaps loss, paf loss per stage
            batch_per_iter_idx = 0
            for batch_data in train_loader:
                if batch_per_iter_idx == 0:
                    optimizer.zero_grad()

                images = batch_data['image'].cuda()
                keypoint_masks = batch_data['keypoint_mask'].cuda()
                paf_masks = batch_data['paf_mask'].cuda()
                keypoint_maps = batch_data['keypoint_maps'].cuda()
                paf_maps = batch_data['paf_maps'].cuda()

                stages_output = net(images)

                losses = []
                for loss_idx in range(len(total_losses) // 2):
                    losses.append(l2_loss(stages_output[loss_idx * 2], keypoint_maps, keypoint_masks, images.shape[0]))
                    writer.add_scalar('Loss/keypoint_map',
                                      l2_loss(stages_output[loss_idx * 2], keypoint_maps, keypoint_masks, images.shape[0]),
                                      num_iter)
                    losses.append(l2_loss(stages_output[loss_idx * 2 + 1], paf_maps, paf_masks, images.shape[0]))
                    writer.add_scalar('Loss/paf_map',
                                      l2_loss(stages_output[loss_idx * 2 + 1], paf_maps, paf_masks, images.shape[0]),
                                      num_iter)
                    total_losses[loss_idx * 2] += losses[-2].item() / batches_per_iter
                    total_losses[loss_idx * 2 + 1] += losses[-1].item() / batches_per_iter



                loss = losses[0]
                for loss_idx in range(1, len(losses)):
                    loss += losses[loss_idx]
                loss /= batches_per_iter
                loss.backward()
                batch_per_iter_idx += 1
                if batch_per_iter_idx == batches_per_iter:
                    optimizer.step()
                    batch_per_iter_idx = 0
                    num_iter += 1
                else:
                    continue

                print("iter = ", num_iter)
                if num_iter % log_after == 0:
                    print('Iter: {}'.format(num_iter))
                    for loss_idx in range(len(total_losses) // 2):
                        print('\n'.join(['stage{}_pafs_loss:     {}', 'stage{}_heatmaps_loss: {}']).format(
                            loss_idx + 1, total_losses[loss_idx * 2 + 1] / log_after,
                            loss_idx + 1, total_losses[loss_idx * 2] / log_after))
                    for loss_idx in range(len(total_losses)):
                        total_losses[loss_idx] = 0
                if num_iter % checkpoint_after == 0:
                    snapshot_name = '{}/checkpoint_iter_{}.pth'.format(checkpoints_folder, num_iter)
                    torch.save({'state_dict': net.module.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'iter': num_iter,
                                'current_epoch': epochId},
                               snapshot_name)
                if num_iter % val_after == 0:
                    print('Validation of teacher model ', str(i))
                    evaluate(val_labels, val_output_name, val_images_folder, net)
                    net.train()
        print("finished training model ", i)
        models.append(net)

    aggregated_teacher(models, student_train_loader, epsilon)


    writer.close()


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
        print("all_keypoints_by_type: ", all_keypoints_by_type)
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        print("All keypoints = ", all_keypoints)
        outputs.append(torch.from_numpy(all_keypoints))

    return outputs

epsilon = 0.2
def aggregated_teacher(models, dataloader, epsilon = 0.2):
    """Aggregates teacher predictions for the student training data"""
    preds = torch.zeros((len(models), len(dataloader), 18, 2), dtype=torch.long)
    for i, model in enumerate(models):
        results = predict(model, dataloader) # num_images x 18 x 2
        preds[i] = results

    print('preds = ', preds)

    ## itt kene a zajt hozzaadni


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared-train-labels', type=str, required=True,
                        help='path to the file with prepared annotations')
    parser.add_argument('--train-images-folder', type=str, required=True, help='path to COCO train images folder')
    parser.add_argument('--num-refinement-stages', type=int, default=1, help='number of refinement stages')
    parser.add_argument('--base-lr', type=float, default=4e-5, help='initial learning rate')
    parser.add_argument('--batch-size', type=int, default=80, help='batch size')
    parser.add_argument('--batches-per-iter', type=int, default=1, help='number of batches to accumulate gradient from')
    parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint to continue training from')
    parser.add_argument('--from-mobilenet', action='store_true',
                        help='load weights from mobilenet feature extractor')
    parser.add_argument('--weights-only', action='store_true',
                        help='just initialize layers with pre-trained weights and start training from the beginning')
    parser.add_argument('--experiment-name', type=str, default='default',
                        help='experiment name to create folder for checkpoints')
    parser.add_argument('--log-after', type=int, default=100, help='number of iterations to print train loss')

    parser.add_argument('--val-labels', type=str, required=True, help='path to json with keypoints val labels')
    parser.add_argument('--val-images-folder', type=str, required=True, help='path to COCO val images folder')
    parser.add_argument('--val-output-name', type=str, default='detections.json',
                        help='name of output json file with detected keypoints')
    parser.add_argument('--checkpoint-after', type=int, default=5000,
                        help='number of iterations to save checkpoint')
    parser.add_argument('--val-after', type=int, default=5000,
                        help='number of iterations to run validation')
    parser.add_argument('--epoch-count', type=int, default=200, required=False,
                        help='Number of epochs to train for')
    parser.add_argument('--num-teachers', type=int, default=2, required=False, help='Number of teacher models')
    args = parser.parse_args()

    checkpoints_folder = '{}_checkpoints'.format(args.experiment_name)
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    writer = SummaryWriter(args.experiment_name)

    train(args.prepared_train_labels, args.train_images_folder, args.num_refinement_stages, args.base_lr, args.batch_size,
          args.batches_per_iter, args.num_workers, args.checkpoint_path, args.weights_only, args.from_mobilenet,
          checkpoints_folder, args.log_after, args.val_labels, args.val_images_folder, args.val_output_name,
          args.checkpoint_after, args.val_after, writer, args.num_teachers)
