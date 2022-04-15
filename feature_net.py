import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
from constants import TrainingConstants
import numpy as np

torch.set_default_dtype(torch.float32)

class FeatureNet(nn.Module):
    def __init__(self, scene_mean, scene_std, target_mean, target_std, name='feature_net'):
        super(FeatureNet, self).__init__()
        self.name = name
        self._device = TrainingConstants.DEVICE

        # TODO: may need to downsample input images.

        self.scene_mean = scene_mean
        self.scene_std = scene_std
        self.target_mean = target_mean
        self.target_std = target_std

        # TODO: FPN?
        resnet18_scene = torchvision.models.resnet18(pretrained=True)
        self.scene_features = nn.Sequential(
            resnet18_scene.conv1,
            resnet18_scene.bn1,
            resnet18_scene.relu,
            resnet18_scene.maxpool,
            resnet18_scene.layer1,
            resnet18_scene.layer2,
            resnet18_scene.avgpool
        )

        resnet18_target = torchvision.models.resnet18(pretrained=True)
        self.target_features = nn.Sequential(
            resnet18_target.conv1,
            resnet18_target.bn1,
            resnet18_target.relu,
            resnet18_target.maxpool,
            resnet18_target.layer1,
            resnet18_target.layer2,
            resnet18_target.avgpool
        )

    def forward_scene(self, scene):
        scene_normalized = (scene - torch.tensor(self.scene_mean, dtype=torch.float32, device=self._device)) / torch.tensor(self.scene_std + 1e-10, dtype=torch.float32, device=self._device)
        return self.scene_features(scene_normalized)

    def forward_target(self, target):
        target_normalized = (target - torch.tensor(self.target_mean, dtype=torch.float32, device=self._device)) / torch.tensor(self.target_std + 1e-10, dtype=torch.float32, device=self._device)
        return self.target_features(target_normalized)

    def forward(self, x):
        scene_img = x[0]
        target_img = x[1]

        scene = self.forward_scene(scene_img)
        target = self.forward_target(target_img)
        
        return torch.mul(scene, target)

    def save(self, dir_path, net_fname, net_label):
        net_path = os.path.join(dir_path, net_label + net_fname)
        moments_path = os.path.join(dir_path, 'moments')
        torch.save(self.state_dict(), net_path)
        if net_label == 'final':
            np.savez(moments_path, scene_mean=self.scene_mean, scene_std=self.scene_std, \
                target_mean=self.target_mean, target_std=self.target_std)

    def load(self, dir_path, net_fname):
        net_path = os.path.join(dir_path, net_fname)
        moments_path = os.path.join(dir_path, 'moments')
        state_dict = torch.load(net_path)
        self.load_state_dict(state_dict)
        moments = np.load(moments_path)
        self.scene_mean = moments['scene_mean']
        self.scene_std = moments['scene_std']
        self.target_mean = moments['target_mean']
        self.target_std = moments['target_std']