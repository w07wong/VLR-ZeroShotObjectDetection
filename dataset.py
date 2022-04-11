import os
import numpy as np
from cv2 import cv2
from torch.utils.data.dataset import Dataset as TorchDataset

class Dataset(TorchDataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir # Path to data directory

        # Assume files are named '000_scene.png or 000_target.png'
        self.scene_fnames = ([data_dir + '/' + fname for fname in os.listdir(self.data_dir) if '_scene.png' in fname]) 
        
        moments = np.load(data_dir + '/moments.npz')
        self.scene_mean, self.scene_std = moments['scene_mean'].astype(np.float32), moments['scene_std'].astype(np.float32)
        self.target_mean, self.target_std = moments['target_mean'].astype(np.float32), moments['target_std'].astype(np.float32)

    def __len__(self):
        return len(self.scene_fnames)

    def __getitem__(self, idx):
        scene_fname = self.scene_fnames[idx] # this is how we index the dataset
        scene_img = np.transpose(cv2.imread(scene_fname), (2, 0, 1)).astype(np.float32) # transposing to (Channel, Width, Height)
        tag = scene_fname[:scene_fname.rfind('_')] # this should extract only the hash from the filename (i.e. 000 from 000_scene.png)
        
        target_fname = '{}_target.png'.format(tag) # find the target image that corresponds with the scene by hash
        target_img = np.transpose(cv2.imread(target_fname), (2, 0, 1)).astype(np.float32) # transposing to (Channel, Width, Height)
        
        bb_fname = '{}.npy'.format(tag) # find the bounding box that corresponds with the scene by hash
        bb = np.load(bb_fname)

        return (scene_img, target_img, bb)