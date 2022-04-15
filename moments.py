import argparse
import os
import numpy as np
from cv2 import cv2
from constants import TrainingConstants

class Moments:
    def __init__(self, data_dir):
        self.data_dir = data_dir # Path to data directory

        # 000_scene.png, 000_target.png
        self.scene_fnames = ([data_dir + '/' + fname for fname in os.listdir(self.data_dir) if '_scene.png' in fname])
        self.scene_fnames.sort()
        self.target_fnames = ([data_dir + '/' + fname for fname in os.listdir(self.data_dir) if '_target.png' in fname])
        self.target_fnames.sort()
        
        self.scene_mean, self.scene_std = self._get_moments(self.scene_fnames)
        self.target_mean, self.target_std = self._get_moments(self.target_fnames)
        
        moments_path = os.path.join(data_dir, 'moments')
        np.savez(moments_path, scene_mean=self.scene_mean, scene_std=self.scene_std, \
            target_mean=self.target_mean, target_std=self.target_std)

    def _get_moments(self, input_fnames):
        count = float(len(input_fnames))

        # Keep running average
        i = 0
        input_mean = []
        input_sq_mean = []

        for input_fname in input_fnames:
            i += 1
           
            image = np.array(np.transpose(cv2.imread(input_fname), (2, 0, 1)), dtype=np.float32)
            image_mean = np.divide(image, count)
            image_squared_mean = np.divide(image**2, count)

            if i == 1:
                input_mean.append(image_mean)
                input_sq_mean.append(image_squared_mean)
            else:
                input_mean = np.stack((input_mean, image_mean), axis=0)
                input_sq_mean = np.stack((input_sq_mean, image_squared_mean), axis=0)
            input_mean = np.sum(input_mean, axis=0)
            input_sq_mean = np.sum(input_sq_mean, axis=0)
        
        input_std = (input_sq_mean - (input_mean**2)).clip(min=0)**0.5
        return input_mean, input_std

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Data directory.'
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    Moments(data_dir)