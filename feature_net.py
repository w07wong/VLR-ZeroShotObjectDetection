import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
from constants import TrainingConstants
import numpy as np


class FeatureNet(nn.Module):
    def __init__(self, name='feature_net'):
        super(FeatureNet, self).__init__()
        self.name = name
        self._device = TrainingConstants.DEVICE

        # TODO: may need to downsample input images.

        #self.scene_mean = scene_mean
        #self.scene_std = scene_std
        #self.target_mean = target_mean
        #self.target_std = target_std

        # TODO: FPN?
        #resnet18_scene = torchvision.models.resnet18(pretrained=False)
        resnet_fpn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        resnet_fpn.train()
        self.scene_fpn = nn.Sequential(*(list(resnet_fpn.children())[1:-2]))

        #self.scene_features = nn.Sequential(
        #    resnet18_scene.conv1,
        #    resnet18_scene.bn1,
        #    resnet18_scene.relu,
        #    resnet18_scene.maxpool,
        #    resnet18_scene.layer1,
        #    resnet18_scene.layer2,
        #    resnet18_scene.layer3,
        #    nn.ConvTranspose2d(256, 128, 2, stride=2),
        #    nn.ReLU(inplace=True),
        #    nn.ConvTranspose2d(128, 64, 2, stride=2),
        #    nn.ReLU(inplace=True),
        #    nn.ConvTranspose2d(64, 64, 2, stride=2),
        #    nn.ReLU(inplace=True),
        #    nn.ConvTranspose2d(64, 32, 2, stride=2)
            #resnet18_scene.avgpool
        #)

        #resnet18_target = torchvision.models.resnet18(pretrained=False)

        vgg_target = torchvision.models.vgg16(pretrained=False)
        vgg_target.train()
        self.target_features = nn.Sequential(*(list(vgg_target.children())[:-2]))
        #self.target_features = nn.Sequential(*[
        #    resnet18_target.conv1,
        #    resnet18_target.bn1,
        #    resnet18_target.relu,
        #    resnet18_target.maxpool,
        #    resnet18_target.layer1,
        #    resnet18_target.layer2,
        #    resnet18_target.layer3,
            #nn.ConvTranspose2d(256, 128, 2, stride=2),
            #nn.ReLU(inplace=True),
            #nn.ConvTranspose2d(128, 64, 2, stride=2),
            #nn.ReLU(inplace=True),
            #nn.ConvTranspose2d(64, 32, 2, stride=2),
            #nn.ReLU(inplace=True),
            #nn.ConvTranspose2d(32, 3, 2, stride=2)
        #])

        #self.linear = nn.Sequential(*[nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Linear(64, 8)])
        self.conv1d = nn.Conv2d(512, 256, 3)
        self.avg_pool = nn.AvgPool2d(3)
        
        self.upsample = torch.nn.Upsample((120, 160), mode='bilinear')

    def forward_scene(self, scene):

        x = self.scene_fpn(scene)
        return x

    def forward_target(self, target):
        x = self.target_features(target)

        x = self.conv1d(x)
        x = self.avg_pool(x)
        #x = x.view(x.shape[0], -1)
        #x = self.linear(x)

        return x #.view(x.shape[0], -1, 1, 1)

    def forward(self, x):
        scene_img = x[0]
        target_img = x[1]

        scene_feat = self.forward_scene(scene_img)

        B, N, C, H, W = target_img.shape
        target_feat = self.forward_target(target_img.reshape(B*N, C, H, W))

        _, C, H, W = target_feat.shape
        target_feat = target_feat.reshape(B, N, C, H, W)

        attention = []
        for b in range(B):
        
            atten_img = []
            for k in scene_feat.keys():

                if k == 'pool':continue
                #print(scene_feat[k][b].shape, target_feat[b].shape)
                atten = F.conv2d(scene_feat[k][b], target_feat[b])
                #print("scene", scene_feat[k][b].max(), scene_feat[k][b].min())
                #print()
                #print("target", target_feat[b].max(), target_feat[b].min())
                atten = self.upsample(atten.unsqueeze(0)).squeeze(0)
                atten_img.append(atten)

            atten_img = torch.stack(atten_img)
            attention.append(atten_img)

        attention = torch.stack(attention)

        B, S, C, H, W = attention.shape # Batch x Scales x C x H x W

        attention = attention.reshape(B, S*C, H, W)
        #attention = torch.cat((scene_img, attention), axis=1)

        return attention#, scene_feat, target_feat

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
