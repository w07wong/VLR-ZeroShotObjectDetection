import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
from constants import TrainingConstants
import cv2
import numpy as np

torch.set_default_dtype(torch.float32)

class ImageList:
    def __init__(self, tensors, image_sizes):
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)

class FeatureNet(nn.Module):
    def __init__(self, scene_mean, scene_std, target_mean, target_std, name='feature_net'):
        super(FeatureNet, self).__init__()
        self.name = name
        self._device = TrainingConstants.DEVICE

        ''' Resize mean and std to correct sizes '''
        self.scene_downsample_rate = 640 / 640
        scene_mean = np.transpose(scene_mean, (1, 2, 0))
        h, w, c = scene_mean.shape
        img = cv2.resize(scene_mean, (int(w * self.scene_downsample_rate), int(h * self.scene_downsample_rate)))
        scene_mean = np.transpose(img, (2, 0, 1))

        scene_std = np.transpose(scene_std, (1, 2, 0))
        h, w, c = scene_std.shape
        img = cv2.resize(scene_std, (int(w * self.scene_downsample_rate), int(h * self.scene_downsample_rate)))
        scene_std = np.transpose(img, (2, 0, 1))

        target_mean = np.transpose(target_mean, (1, 2, 0))
        h, w, c = target_mean.shape
        img = cv2.resize(target_mean, (int(w * 224 / 640), int(h * 224 / 480)))
        target_mean = np.transpose(img, (2, 0, 1))

        target_std = np.transpose(target_std, (1, 2, 0))
        h, w, c = target_std.shape
        img = cv2.resize(target_std, (int(w * 224 / 640), int(h * 224 / 480)))
        target_std = np.transpose(img, (2, 0, 1))

        ''' Convert mean and std deviations to tensors '''
        self.scene_mean = torch.tensor(scene_mean, dtype=torch.float32, device=self._device)
        self.scene_std = torch.tensor(scene_std + 1e-10, dtype=torch.float32, device=self._device)
        self.target_mean = torch.tensor(target_mean, dtype=torch.float32, device=self._device)
        self.target_std = torch.tensor(target_std + 1e-10, dtype=torch.float32, device=self._device)

        ''' Define scene feature extractor '''
        faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        #for param in faster_rcnn.parameters():
        #    param.requires_grad = False
        self.faster_rcnn_backbone = faster_rcnn.backbone
        self.faster_rcnn_rpn = faster_rcnn.rpn
        self.faster_rcnn_roi_pool = faster_rcnn.roi_heads.box_roi_pool

        ''' Define target feature extractor '''
        resnet50_target = torchvision.models.resnet50(pretrained=True)
        self.target_features = nn.Sequential(
            resnet50_target.conv1,
            resnet50_target.bn1,
            resnet50_target.relu,
            resnet50_target.maxpool,
            resnet50_target.layer1,
            resnet50_target.layer2,
            resnet50_target.layer3,
            resnet50_target.layer4,
            resnet50_target.avgpool
        )

        self.fc1 = nn.Linear(2048, 256)
        self.bb1 = nn.Linear(10000, 4096)
        self.bb2 = nn.Linear(4096, 4096)
        self.bb3 = nn.Linear(4096, 2048)
        self.bb4 = nn.Linear(2048, 4)

    def forward_scene(self, scene, bb):
        scene_normalized = (scene - self.scene_mean) / self.scene_std

        image_shapes = [(img.shape[1], img.shape[2]) for img in scene_normalized]
        scene_features = self.faster_rcnn_backbone(scene_normalized)

        if bb is not None:
            bb = [{'boxes': torch.unsqueeze(box, dim=0)} for box in bb]
        scene_rpn_boxes, scene_rpn_losses = self.faster_rcnn_rpn(ImageList(scene_normalized, image_shapes), scene_features, bb)

        roi_pool = self.faster_rcnn_roi_pool(scene_features, scene_rpn_boxes, image_shapes)
        return roi_pool, scene_rpn_boxes

    def forward_target(self, target):
        target_normalized = (target - self.target_mean) / self.target_std
        return self.target_features(target_normalized)

    def forward(self, x):
        scene_img = x[0].to(self._device)
        target_img = x[1].to(self._device)
        if len(x) > 2:
            bb = x[2].to(self._device)
        else:
            bb = None

        scene, scene_rpn_boxes = self.forward_scene(scene_img, bb)
        scene = F.avg_pool2d(scene, scene.shape[2])
        scene = scene.view(scene_img.shape[0], -1, scene.shape[1], scene.shape[2], scene.shape[3]).squeeze()
        # if not self.training:
        #     scene = scene.unsqueeze(0)
        scene = F.normalize(scene, p=2.0, dim=2)

        target = self.forward_target(target_img).squeeze()
        target = self.fc1(target)
        # if not self.training:
        #     target = target.unsqueeze(0)
        target = F.normalize(target, p=2.0, dim=1)

        output = torch.einsum('bcd,bd->bc', scene, target)
        output = F.softmax(output, dim=1).unsqueeze(2)
        scene_rpn_boxes = torch.stack(scene_rpn_boxes)
        scene_rpn_boxes[:, 0] /= 640
        scene_rpn_boxes[:, 2] /= 640
        scene_rpn_boxes[:, 1] /= 480
        scene_rpn_boxes[:, 3] /= 480
        output = torch.cat((output, scene_rpn_boxes), dim=2)
        if output.shape[1] < 2000:
            output = torch.cat((output, torch.full((output.shape[0], 2000 - output.shape[1], output.shape[2]), 0).to(self._device)), dim=1)

        x = output.view(output.shape[0], -1)
        x = self.bb1(x)
        x = F.relu(x)
        x = self.bb2(x)
        x = F.relu(x)
        x = self.bb3(x)
        x = F.relu(x)
        x = self.bb4(x)
        # return torch.clamp(x, min=0, max=1)
        return x


        return output

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
