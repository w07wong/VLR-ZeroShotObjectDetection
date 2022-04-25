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

def eval(feature_net, dataset, data_loader, device):
    feature_criterion = nn.L1Loss(reduction="none")
    bb_criterion = nn.L1Loss(reduction="mean")

    feature_eval_losses = []
    bb_eval_losses = []
    with torch.no_grad():
        i = 0
        for batch_idx, (scene_img, target_img, bb, bb_original) in enumerate(data_loader):
            print(bb_original)
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
            
            bb_output = feature_net((scene_img, target_img))
            # print(bb_output)
            bb_loss = bb_criterion(bb_output, bb)

            bb_eval_losses.append(bb_loss.item())

            pred_bb = bb_output[0]
            x_min = int(np.floor(max(0, pred_bb[0].item())))
            x_max = int(np.ceil(min(1.0, pred_bb[0].item()+pred_bb[2].item())))
            y_min = int(np.floor(max(0, pred_bb[1].item())))
            y_max = int(np.ceil(min(1.0, pred_bb[1].item()+pred_bb[3].item())))

            # Plot image to wandb
            scene_img = dataset[i][0].transpose(1, 2, 0)
            '''
            x_min /= scene_img.shape[1]
            x_max /= scene_img.shape[1]
            y_min /= scene_img.shape[0]
            y_max /= scene_img.shape[0]
            '''
            # print(x_min, x_max, y_min, y_max)
            wandb.log({"image_proposals": wandb.Image(scene_img, boxes={
                "proposals": {
                    "box_data": [{"position": {"minX": x_min, "minY": y_min, "maxX": x_max, "maxY": y_max}, "class_id": 0}],
                    "class_labels": {0: "prediction"}
                }
            })})
            gt_x_min, gt_y_min, gt_x_max, gt_y_max = dataset[i][2]
            gt_x_max += gt_x_min
            gt_y_max += gt_y_min 

            '''
            gt_x_min /= scene_img.shape[1]
            gt_x_max /= scene_img.shape[1]
            gt_y_max /= scene_img.shape[0]
            gt_y_min /= scene_img.shape[0]
            '''
            # print(gt_x_min, gt_y_min, gt_x_max, gt_y_max)

            wandb.log({"image_gt": wandb.Image(scene_img, boxes={
                "ground_truth": {
                    "box_data": [{"position": {"minX": gt_x_min, "minY": gt_y_min, "maxX": gt_x_max, "maxY": gt_y_max}, "class_id": 0}],
                    "class_labels": {0: "ground_truth"}
                }
            })})

            i += 1

    print(feature_eval_losses)
    print(bb_eval_losses)

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
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1)

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
