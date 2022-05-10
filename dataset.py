import os
import numpy as np
import cv2
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision import transforms
from PIL import Image
import torch
import albumentations as A
from collections import namedtuple
import csv


class Dataset(TorchDataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir # Path to data directory

        # Assume files are named '000_scene.png or 000_target.png'
        self.scene_fnames = ([data_dir + '/' + fname for fname in os.listdir(self.data_dir) if '_scene.png' in fname]) 
        
        #moments = np.load(data_dir + '/moments.npz')
        #self.scene_mean, self.scene_std = moments['scene_mean'].astype(np.float32), moments['scene_std'].astype(np.float32)
        #self.target_mean, self.target_std = moments['target_mean'].astype(np.float32), moments['target_std'].astype(np.float32)

        self.img_width = 640
        self.img_height = 480
        
        self.resize_img_width = 640
        self.resize_img_height = 480


        if not is_train:
            f = open('/home/rohanc/vlr/project/dataset/YCB/output/train/train.csv')
        else:
            f = open('/home/rohanc/vlr/project/dataset/YCB/output/test/test.csv')


        reader = csv.reader(f)
        self.rows = []
        for row in reader:
            self.rows.append(row)


        self.scene_transform = transforms.Compose([ #transforms.Resize((self.resize_img_height, self.resize_img_width)),
                                             transforms.ToTensor(),
                                             #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                             #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                           ])

        self.target_transform = transforms.Compose([#transforms.Resize((self.resize_img_height, self.resize_img_width)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                           ])

        #aug_policy = [[policies.POLICY_TUPLE('Translate_X', 0.5, np.random.randint(0, 10)), 
        #               policies.POLICY_TUPLE('Translate_Y', 0.5, np.random.randint(0, 10))] for _ in range(10)]

        #self.policy_container = policies.PolicyContainer(aug_policy)

        # select a random policy from the policy set
        #self.random_policy = self.policy_container.select_random_policy() 

        self.bbox_transform = A.Compose([#A.RandomScale(scale_limit=0.2, p=0.4),
                                         A.RandomBrightnessContrast(p=0.3),
                                         #A.CLAHE(p=0.3),
                                         A.Affine(translate_percent={'x':0.3, 'y':0.2}, p=0.9),
                                         #A.RandomSizedBBoxSafeCrop(width=320, height=240, erosion_rate=0.2, p=0.3),
                                         A.HorizontalFlip(p=0.5),
                                         #A.RandomRotate90(p=0.5),
                                         A.Resize(width=640, height=480),
                                        ], bbox_params=A.BboxParams(format='pascal_voc',  label_fields=['class_labels']))

        self.is_train = is_train

    def __len__(self):
        #return len(self.scene_fnames)
        return len(self.rows)

    def _resize_img(self, img):
        # Resize scene img to 640x480 if needed
        if img.shape[1] != self.resize_img_width and img.shape[0] != self.resize_img_height:
            img = cv2.resize(img, (self.img_width, self.img_height))
        return img

    def _crop_coords(self, img):
        y_nonzero, x_nonzero, _ = np.nonzero(img)
        return [np.min(y_nonzero), np.max(y_nonzero), np.min(x_nonzero), np.max(x_nonzero)]

    def __getitem__(self, idx):
        scene_fname = self.rows[idx][0] # this is how we index the dataset
        #tag = scene_fname[:scene_fname.rfind('_')] # this should extract only the hash from the filename (i.e. 000 from 000_scene.png)

        #scene_fname = os.readlink(scene_fname)
        scene_img = Image.open(scene_fname)
        scene_img = (transforms.ToTensor()(scene_img) * 255).permute(1,2,0).numpy()

        #scene_img = self._resize_img(scene_img) # Can remove later
        #scene_img = np.transpose(scene_img, (2, 0, 1)).astype(np.float32) # transposing to (Channel, Height, Width)

        #target_fname = '{}_target.png'.format(tag) # find the target image that corresponds with the scene by hash
        #target_fname = os.readlink(target_fname)

        targets = []
        for target_fname in self.rows[idx][5:]:

            target_img = Image.open(target_fname)
            targets.append(self.target_transform(target_img))

        targets = torch.stack(targets)
        #target_img = (transforms.ToTensor()(target_img)*255).permute(1,2,0)
        #target_img = self._resize_img(target_img) # Can remove later
        #target_img = np.transpose(target_img, (2, 0, 1)).astype(np.float32) # transposing to (Channel, Height, Width)
        
        #bb_fname = '{}.npy'.format(tag) # find the bounding box that corresponds with the scene by hash
        bb = self.rows[idx][1:5]
        #bb = np.load(bb_fname)

        # bb - [x_min, y_min, x_max, y_max]
        bb = [[float(b) for b in bb]]

        transformed = self.bbox_transform(image=scene_img, bboxes=bb, class_labels=[0])
        scene_aug = transformed['image']


        if self.is_train and len(transformed['bboxes']) > 0:
             scene_img = torch.from_numpy(scene_aug).float().permute(2,0,1) / 255.0
             #bb = np.array(bbs_aug[0])[1:].astype(np.float32)
             bb = np.array(transformed['bboxes'][0])
        else:
             scene_img = torch.from_numpy(scene_img).float().permute(2, 0, 1) / 255.0   
             bb = np.array(bb[0])

        scene_img_ = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(scene_img)
        bb[2] = (bb[2]-bb[0]) / self.img_width
        bb[3] = (bb[3]-bb[1]) / self.img_height
        bb[0] /= self.img_width
        bb[1] /= self.img_height
        bb = torch.from_numpy(bb)


        return (scene_img_, targets, bb, scene_img)
