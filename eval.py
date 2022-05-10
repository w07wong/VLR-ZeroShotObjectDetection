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
from torch.utils.data import SubsetRandomSampler
from math import ceil
from torchvision import transforms

np.random.seed(1337)
def eval(feature_net, bb_net, dataset, data_loader, device):
    feature_criterion = nn.L1Loss(reduction="none")
    bb_criterion = nn.SmoothL1Loss()

    feature_eval_losses = []
    bb_eval_losses = []
    with torch.no_grad():
        i = 0
        for batch_idx, (scene_img, target_img, bb, scene_img2) in enumerate(data_loader):
            scene_img = scene_img.to(device)

            #scene_img2 = scene_img.clone()
            target_img = target_img.to(device)
            bb = bb.to(device)
            
            feature_output = feature_net((scene_img, target_img))
            bb_output = bb_net(feature_output)
            #bb_output[:,2] += bb_output[:,0]
            #bb_output[:,3] += bb_output[:,1]

            bb_loss = bb_criterion(bb_output, bb)

            #target_feature_map = feature_net.forward_target(target_img)
            for j in range(len(bb_output)):
                pred_bb = bb_output[j]
                x_min = max(0, pred_bb[0].item())
                x_max = min(scene_img.shape[2], pred_bb[2].item())
                y_min = max(0, pred_bb[1].item())
                y_max = min(scene_img.shape[3], pred_bb[3].item())
                #print(x_min, x_max, y_min, y_max, pred_bb, bb)
                #scene_img[j] = F.interpolate(scene_img[j, :, y_min:y_max, x_min:x_max].unsqueeze(0), size=(scene_img.shape[2], scene_img.shape[3]), mode='bilinear')
           # bb_feature_map = feature_net.forward_scene(scene_img)
           # feature_loss = feature_criterion(target_feature_map, bb_feature_map).sum(1).mean()

            #feature_eval_losses.append(feature_loss.item())
            #bb_eval_losses.append(bb_loss.item())

            # Plot image to wandb
            scene_img = scene_img2[0].cpu().detach().permute(1,2,0).numpy()
            scene_img = scene_img.astype(np.float32)
         
            print(scene_img.max(), scene_img.min())
            # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            #scene_img = transforms.Normalize([0,0,0],[1/0.229, 1/0.224, 1/0.225])(scene_img)
            #scene_img = transforms.Normalize([-0.485, -0.456, -0.406],[1, 1, 1])(scene_img)
            #scene_img = scene_img#.permute(1,2,0).cpu().detach().numpy().astype(np.float64)

            #x_min /= scene_img.shape[1]
            #x_max /= scene_img.shape[1]
            #y_min /= scene_img.shape[0]
            #y_max /= scene_img.shape[0]
            gt_x_min, gt_y_min, gt_x_max, gt_y_max = bb[0].cpu().detach().numpy()
            gt_x_max += gt_x_min
            gt_y_max += gt_y_min 
           
            x_min, y_min, x_max, y_max = bb_output[0].cpu().detach().numpy().astype(np.float64)
            x_max += x_min
            y_max += y_min 
            print(x_min, x_max, y_min, y_max)
            #print(type(gt_x_min), type(gt_x_max), gt_y_min, gt_y_max)

            wandb.log({"image": wandb.Image(scene_img, boxes={
                "proposals": {
                    "box_data": [{"position": {"minX": x_min, "minY": y_min, "maxX": x_max, "maxY": y_max}, "class_id": 0},
                                 {"position": {"minX": gt_x_min, "minY": gt_y_min, "maxX": gt_x_max, "maxY": gt_y_max}, "class_id": 1}],
                    "class_labels": {0: "prediction", 1: "ground_truth"}
                }
            })})


            #gt_x_min /= scene_img.shape[1]
            #gt_x_max /= scene_img.shape[1]
            #gt_y_max /= scene_img.shape[0]
            #gt_y_min /= scene_img.shape[0]

            #wandb.log({"image_gt": wandb.Image(scene_img, boxes={
            #    "ground_truth": {
            #        "box_data": [{"position": {"minX": gt_x_min, "minY": gt_y_min, "maxX": gt_x_max, "maxY": gt_y_max}, "class_id": 0}],
            #        "class_labels": {0: "ground_truth"}
            #    }
            #})})
            i += 1


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

    is_train = False
    dataset = Dataset(args.data_dir, is_train=is_train)

    total_size=0.1
    val_size=TrainingConstants.VAL_SIZE


    ind = np.arange(len(dataset))
    np.random.shuffle(ind)
    ind = ind[:ceil(total_size*len(ind))]
    train_ind = ind[:ceil((1-val_size)*len(ind))]
    train_sampler = SubsetRandomSampler(train_ind)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, pin_memory=True,sampler=train_sampler, shuffle=False)

    #data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

    if args.cuda:
        device = DeviceConstants.CUDA
    else:
        device = DeviceConstants.CPU

    feature_net = FeatureNet(name=TrainingConstants.FEATURE_NET_NAME)
    feature_net.cuda()
    feature_net.load_state_dict(torch.load(args.feature_net, map_location=torch.device(device)))
    feature_net.eval()

    bb_net = BoundingBoxNet(name=TrainingConstants.BOUNDING_BOX_NET_NAME)
    bb_net.cuda()
    bb_net.load_state_dict(torch.load(args.bb_net, map_location=torch.device(device)))
    bb_net.eval()

    # Use wandb to visualize bounding boxes
    wandb.init(project="vlrzero")#, reinit=True)

    eval(feature_net, bb_net, dataset, data_loader, device)
