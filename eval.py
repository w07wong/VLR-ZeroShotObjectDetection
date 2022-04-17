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

def eval(feature_net, bb_net, dataset, data_loader, device):
    feature_criterion = nn.L1Loss(reduction="none")
    bb_criterion = nn.L1Loss(reduction="none")

    feature_eval_losses = []
    bb_eval_losses = []
    with torch.no_grad():
        i = 0
        for batch_idx, (scene_img, target_img, bb) in enumerate(data_loader):
            i += 1
            scene_img = scene_img.to(device)
            target_img = target_img.to(device)
            bb = bb.to(device)
            
            feature_output = feature_net((scene_img, target_img))
            bb_output = bb_net(feature_output)

            bb_loss = bb_criterion(bb_output, bb).sum(1)

            target_feature_map = feature_net.forward_target(target_img)
            for i in range(len(bb_output)):
                pred_bb = bb_output[i]
                x_min = int(np.floor(max(0, pred_bb[0].item())))
                x_max = int(np.ceil(min(scene_img.shape[2], pred_bb[1].item() + 1)))
                y_min = int(np.floor(max(0, pred_bb[2].item())))
                y_max = int(np.ceil(min(scene_img.shape[3], pred_bb[3].item() + 1)))
                scene_img[i] = F.interpolate(scene_img[i, :, x_min:x_max, y_min:y_max].unsqueeze(0), size=(scene_img.shape[2], scene_img.shape[3]), mode='bilinear')
            bb_feature_map = feature_net.forward_scene(scene_img)
            feature_loss = feature_criterion(target_feature_map, bb_feature_map).sum(1)

            feature_eval_losses.append(feature_loss.item())
            bb_eval_losses.append(bb_loss.item())

            # Plot image to wandb
            scene_img = dataset[i][0].transpose(1, 2, 0)
            x_min /= scene_img.shape[1]
            x_max /= scene_img.shape[1]
            y_min /= scene_img.shape[0]
            y_max /= scene_img.shape[0]
            wandb.log({"image_proposals": wandb.Image(scene_img, boxes={
                "proposals": {
                    "box_data": [{"position": {"minX": x_min, "minY": y_min, "maxX": x_max, "maxY": y_max}, "class_id": 0}],
                    "class_labels": {0: "prediction"}
                }
            })})
            gt_x_min, gt_y_min, gt_x_max, gt_y_max = dataset[i][2]
            gt_x_min /= scene_img.shape[1]
            gt_x_max /= scene_img.shape[1]
            gt_y_min /= scene_img.shape[0]
            gt_y_max /= scene_img.shape[0]
            wandb.log({"image_gt": wandb.Image(scene_img, boxes={
                "ground_truth": {
                    "box_data": [{"position": {"minX": gt_x_min, "minY": gt_y_min, "maxX": gt_x_max, "maxY": gt_y_max}, "class_id": 0}],
                    "class_labels": {0: "ground_truth"}
                }
            })})

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
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1)

    if args.cuda:
        device = DeviceConstants.CUDA
    else:
        device = DeviceConstants.CPU

    feature_net = FeatureNet(dataset.scene_mean, dataset.scene_std, dataset.target_mean, dataset.target_std, name=TrainingConstants.FEATURE_NET_NAME)
    feature_net.load_state_dict(torch.load(args.feature_net, map_location=torch.device(device)))
    feature_net.eval()

    bb_net = BoundingBoxNet(name=TrainingConstants.BOUNDING_BOX_NET_NAME)
    bb_net.load_state_dict(torch.load(args.bb_net, map_location=torch.device(device)))
    bb_net.eval()

    # Use wandb to visualize bounding boxes
    wandb.init(project="vlr-project", reinit=True)

    eval(feature_net, bb_net, dataset, data_loader, device)