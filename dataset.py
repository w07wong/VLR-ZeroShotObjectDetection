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

    def _resize_img(self, img, width, height):
        # if img.shape[1] != self.img_width and img.shape[0] != self.img_height:
        #     img = cv2.resize(img, (self.img_width, self.img_height))
        img = cv2.resize(img, (width, height))
        return img

    def crop(self, img):
        y_nonzero, x_nonzero, _ = np.nonzero(img)
        return img[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

    def __getitem__(self, idx):
        scene_fname = self.scene_fnames[idx]
        tag = scene_fname[:scene_fname.rfind('_')] # this should extract only the hash from the filename (i.e. 000 from 000_scene.png)

        scene_fname = os.readlink(scene_fname)
        scene_img = cv2.imread(scene_fname)
        scene_img = cv2.cvtColor(scene_img, cv2.COLOR_RGB2BGR)
        scene_img = self._resize_img(scene_img, 640, 480) # Can remove later
        scene_img = np.transpose(scene_img, (2, 0, 1)).astype(np.float32) # transposing to (Channel, Width, Height)
        
        target_fname = '{}_target.png'.format(tag) # find the target image that corresponds with the scene by hash
        target_fname = os.readlink(target_fname)
        target_img = cv2.imread(target_fname)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)
        # Crop the target image
        target_img = self.crop(target_img)
        target_img = self._resize_img(target_img, 224, 224) # Can remove later
        target_img = np.transpose(target_img, (2, 0, 1)).astype(np.float32) # transposing to (Channel, Width, Height)
        
        bb_fname = '{}.npy'.format(tag) # find the bounding box that corresponds with the scene by hash
        bb = np.load(bb_fname)

        # bb - [x_min, y_min, x_max, y_max]
        bb[2] = bb[2] / self.img_width
        bb[3] = bb[3] / self.img_height

        bb[0] /= self.img_width
        bb[1] /= self.img_height
        
        return (scene_img, target_img, bb, np.load(bb_fname))
