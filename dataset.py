import os
import numpy as np
import cv2
from torch.utils.data.dataset import Dataset as TorchDataset

class Dataset(TorchDataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir # Path to data directory

        # Assume files are named '000_scene.png or 000_target.png'
        self.scene_fnames = ([data_dir + '/' + fname for fname in os.listdir(self.data_dir) if '_scene.png' in fname]) 
        
        moments = np.load(data_dir + '/moments.npz')
        self.scene_mean, self.scene_std = moments['scene_mean'].astype(np.float32), moments['scene_std'].astype(np.float32)
        self.target_mean, self.target_std = moments['target_mean'].astype(np.float32), moments['target_std'].astype(np.float32)

        self.img_width = 640
        self.img_height = 480

    def __len__(self):
        return len(self.scene_fnames)

    def _resize_img(self, img):
        # Resize scene img to 640x480 if needed
        if img.shape[1] != 640 and img.shape[0] != 480:
            img = cv2.resize(img, (self.img_width, self.img_height))
        return img

    def __getitem__(self, idx):
        scene_fname = self.scene_fnames[idx] # this is how we index the dataset
        tag = scene_fname[:scene_fname.rfind('_')] # this should extract only the hash from the filename (i.e. 000 from 000_scene.png)

        scene_fname = os.readlink(scene_fname)
        scene_img = cv2.imread(scene_fname)
        scene_img = self._resize_img(scene_img) # Can remove later
        scene_img = np.transpose(scene_img, (2, 0, 1)).astype(np.float32) # transposing to (Channel, Width, Height)
        
        target_fname = '{}_target.png'.format(tag) # find the target image that corresponds with the scene by hash
        target_fname = os.readlink(target_fname)
        target_img = cv2.imread(target_fname)
        target_img = self._resize_img(target_img) # Can remove later
        target_img = np.transpose(target_img, (2, 0, 1)).astype(np.float32) # transposing to (Channel, Width, Height)
        
        bb_fname = '{}.npy'.format(tag) # find the bounding box that corresponds with the scene by hash
        bb = np.load(bb_fname)

        # bb - [x_min, y_min, x_max, y_max]
        bb = np.array([bb[0], bb[1], bb[2], bb[3]])

        bb[2] = (bb[2] - bb[0]) / scene_img.shape[1]
        bb[3] = (bb[3] - bb[1]) / scene_img.shape[2]

        bb[0] /= scene_img.shape[1]
        bb[1] /= scene_img.shape[2]

        return (scene_img, target_img, bb)
