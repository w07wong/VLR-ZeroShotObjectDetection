import argparse
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import Dataset
from feature_net import FeatureNet
from bb_net import BoundingBoxNet
from constants import TrainingConstants, DeviceConstants
from math import ceil
from torch.utils.data import SubsetRandomSampler
import cv2

def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU value
    """
    # print(box1, box2)
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    xmin1 *= 640
    xmin2 *= 640

    xmax1 *= 640
    xmax2 *= 640

    ymin1 *= 480
    ymin2 *= 480

    ymax1 *= 480
    ymax2 *= 480

    box1_area = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    box2_area = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)

    intersection = max(0, min(xmax1, xmax2) - max(xmin1, xmin2) + 1) * max(0, min(ymax1, ymax2) - max(ymin1, ymin2) + 1)

    iou = intersection / (box1_area - intersection + box2_area)
    return iou

def eval(feature_net, dataset, data_loader, device):
    feature_criterion = nn.L1Loss(reduction="none")
    bb_criterion = nn.L1Loss(reduction="mean")

    feature_eval_losses = []
    bb_eval_losses = []
    ious = []
    with torch.no_grad():
        i = 0
        for batch_idx, (scene_img, target_img, bb, bb_original) in enumerate(data_loader):
        # for i in range(len(dataset)):
            '''
            scene_img, target_img, bb, bb_original = dataset[i]
            scene_img = torch.tensor(scene_img).to(device)
            target_img = torch.tensor(target_img).to(device)
            bb = torch.tensor(bb).to(device)
            '''
            scene_img = scene_img.to(device)
            target_img = target_img.to(device)
            bb = bb.to(device)
            

            bb_idx, scene_rpn_boxes, scene_rpn_losses = feature_net((scene_img, target_img, bb_original))
            bb_output = torch.index_select(scene_rpn_boxes, 0, (torch.clamp(bb_idx, min=0, max=1) * (len(scene_rpn_boxes) - 1)).squeeze().long())
                        
            # print('gt', bb[0])
            # print('pred', bb_output[0])
            bb_loss = bb_criterion(bb_output, bb)

            bb_eval_losses.append(bb_loss.item())

            pred_bb = bb_output[0]
            '''
            x_min = min(1, max(0, pred_bb[0].item()))
            x_max = max(0, min(1.0, pred_bb[0].item()+pred_bb[2].item()))
            y_min = min(1, max(0, pred_bb[1].item()))
            y_max = max(0, min(1.0, pred_bb[1].item()+pred_bb[3].item()))
            '''
            x_min = max(0.0, pred_bb[0].item())
            y_min = max(0.0, pred_bb[1].item())
            x_max = min(1.0, pred_bb[2].item())
            y_max = min(1.0, pred_bb[3].item())
            # print(bb, pred_bb, x_min, x_max, y_min, y_max)

            # Calculate iou
            ious.append(iou(bb[0], pred_bb).cpu().detach().numpy())

            # Plot image to wandb
            scene_img = dataset[i][0].transpose(1, 2, 0)
            # scene_img = cv2.cvtColor(scene_img, cv2.COLOR_RGB2BGR)
            '''
            x_min /= scene_img.shape[1]
            x_max /= scene_img.shape[1]
            y_min /= scene_img.shape[0]
            y_max /= scene_img.shape[0]
            '''
            if batch_idx % 100 == 0:
                '''
                wandb.log({"image_proposals": wandb.Image(scene_img, boxes={
                    "proposals": {
                        "box_data": [{"position": {"minX": x_min, "minY": y_min, "maxX": x_max, "maxY": y_max}, "class_id": 0}],
                        "class_labels": {0: "prediction"}
                    }
                })})
                '''
                gt_x_min, gt_y_min, gt_x_max, gt_y_max = bb[0][0], bb[0][1], bb[0][2], bb[0][3]
                # gt_x_max += gt_x_min
                # gt_y_max += gt_y_min 

                '''
                gt_x_min /= scene_img.shape[1]
                gt_x_max /= scene_img.shape[1]
                gt_y_max /= scene_img.shape[0]
                gt_y_min /= scene_img.shape[0]
                '''
                # print(gt_x_min, gt_y_min, gt_x_max, gt_y_max)
                # print(x_min, y_min, x_max, y_max)

                wandb.log({"target": wandb.Image(target_img)})

                wandb.log({"image_gt": wandb.Image(scene_img, boxes={
                    "bounding_boxes": {
                        "box_data": [{"position": {"minX": x_min, "minY": y_min, "maxX": x_max, "maxY": y_max}, "class_id": 0}, {"position": {"minX": gt_x_min, "minY": gt_y_min, "maxX": gt_x_max, "maxY": gt_y_max}, "class_id": 1}],
                    "class_labels": {0: "prediction", 1: "ground_truth"}
                    }
                })})

            i += 1

            if i % 100 == 0:
                print(i, np.mean(ious))

    print(feature_eval_losses)
    print(bb_eval_losses)
    print('mean iou', np.mean(ious))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a network.')
    parser.add_argument(
        'data_dir',
        type=str,
        help='Path to the training data.'
    )
    parser.add_argument(
        '--feature_net',
        type=str,
        help='Path to feature extraction network'
    )
    parser.add_argument(
        '--bb_net',
        type=str,
        help='Path to bouding box network'
    )
    parser.add_argument(
        '--cuda',
        action='store_true',
        help='Enable CUDA support and utilize GPU devices.'
    )
    args = parser.parse_args()


    dataset = Dataset(args.data_dir)

    '''
    total_size=1.0
    val_size=0.2
    ind = np.arange(len(dataset))
    # np.random.shuffle(ind)
    ind = ind[:ceil(total_size*len(ind))]
    train_ind = ind[:ceil((1-val_size)*len(ind))]
    val_ind = ind[ceil((1-val_size)*len(ind)):]
    train_sampler = SubsetRandomSampler(train_ind)
    val_sampler = SubsetRandomSampler(val_ind)
    data_loader = torch.utils.data.DataLoader(
                                dataset,
                                batch_size=1,
                                num_workers=1,
                                pin_memory=True,
                                sampler=train_sampler)
    '''
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1)

    if args.cuda:
        device = DeviceConstants.CUDA
    else:
        device = DeviceConstants.CPU

    feature_net = FeatureNet(dataset.scene_mean, dataset.scene_std, dataset.target_mean, dataset.target_std, name=TrainingConstants.FEATURE_NET_NAME)
    feature_net.cuda()
    feature_net.load_state_dict(torch.load(args.feature_net, map_location=torch.device(device)))
    feature_net.eval()

    # Use wandb to visualize bounding boxes
    wandb.init(project="vlr-project", reinit=True)

    eval(feature_net, dataset, data_loader, device)
