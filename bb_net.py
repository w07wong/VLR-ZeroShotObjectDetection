import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 

import os
from constants import TrainingConstants
import numpy as np

torch.set_default_dtype(torch.float32)

class BoundingBoxNet(nn.Module):
    def __init__(self, name='bb_net'):
        super(BoundingBoxNet, self).__init__()
        self.name = name
        self._device = TrainingConstants.DEVICE

        # TODO: idk the input size right now so putting 1000.

        self.conv = torchvision.models.resnet18(pretrained=False)

        self.bb1 = nn.Linear(128, 128)
        self.bb2 = nn.Linear(128, 64)
        self.bb3 = nn.Linear(64, 4)

    def forward(self, x):
        print(x.shape)

        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.bb1(x)
        x = F.relu(x)
        x = self.bb2(x)
        x = F.relu(x)
        return self.bb3(x)

    def save(self, dir_path, net_fname, net_label):
        net_path = os.path.join(dir_path, net_label + net_fname)
        torch.save(self.state_dict(), net_path)

    def load(self, dir_path, net_fname):
        net_path = os.path.join(dir_path, net_fname)
        state_dict = torch.load(net_path)
        self.load_state_dict(state_dict)
