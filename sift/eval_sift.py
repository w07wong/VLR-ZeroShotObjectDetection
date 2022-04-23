import argparse
import wandb
import numpy as np
import cv2
import os

from iou import iou
from sift import get_sift_bb

def eval(data_dir, save_viz):
    ious = []
     # Assume files are named '000_scene.png or 000_target.png'
    scene_fnames = ([data_dir + '/' + fname for fname in os.listdir(data_dir) if '_scene.png' in fname]) 

    for i in range(len(scene_fnames)):
        # print(i)
        scene_fname = scene_fnames[i]
        tag = scene_fname[:scene_fname.rfind('_')] # this should extract only the hash from the filename (i.e. 000 from 000_scene.png)

        # Load the scene image that corresponds with the hash
        scene_fname = os.readlink(scene_fname)
        scene_img = cv2.imread(scene_fname)
        
        # Load the target image that corresponds with the scene by hash
        target_fname = '{}_target.png'.format(tag)
        target_fname = os.readlink(target_fname)
        target_img = cv2.imread(target_fname)

        # Load the bounding box that corresponds with the scene by hash
        bb_fname = '{}.npy'.format(tag)
        gt_bb = np.load(bb_fname)

        # Get sift bounding box
        path = 'sift/images/'
        os.makedirs('sift/images/', exist_ok=True)
        viz_fname = path + str(i) + '.png'
        pred_bb = get_sift_bb(scene_img, target_img, gt_bb, save_viz, viz_fname)
        x_min, y_min, x_max, y_max = pred_bb

        # Plot image to wandb
        x_min /= scene_img.shape[1]
        x_max /= scene_img.shape[1]
        y_min /= scene_img.shape[0]
        y_max /= scene_img.shape[0]
        # wandb.log({"sift-image_proposals": wandb.Image(scene_img, boxes={
        #     "proposals": {
        #         "box_data": [{"position": {"minX": x_min, "minY": y_min, "maxX": x_max, "maxY": y_max}, "class_id": 0}],
        #         "class_labels": {0: "sift_prediction"}
        #     }
        # })})
        gt_x_min, gt_y_min, gt_x_max, gt_y_max = gt_bb
        gt_x_max += gt_x_min
        gt_y_max += gt_y_min 

        gt_x_min /= scene_img.shape[1]
        gt_x_max /= scene_img.shape[1]
        gt_y_max /= scene_img.shape[0]
        gt_y_min /= scene_img.shape[0]

        # wandb.log({"image_gt": wandb.Image(scene_img, boxes={
        #     "ground_truth": {
        #         "box_data": [{"position": {"minX": gt_x_min, "minY": gt_y_min, "maxX": gt_x_max, "maxY": gt_y_max}, "class_id": 0}],
        #         "class_labels": {0: "ground_truth"}
        #     }
        # })})
        ious.append(iou(pred_bb, gt_bb))

    # print(ious)
    print(np.mean(ious))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a network.')
    parser.add_argument(
        'data_dir',
        type=str,
        help='Path to the evaluation data.'
    )
    parser.add_argument(
        '--save_viz',
        action='store_true',
        help='Save visualizations.'
    )
    args = parser.parse_args()

    # Use wandb to visualize bounding boxes
    # wandb.init(project="vlr-project", reinit=True)

    eval(args.data_dir, args.save_viz)
