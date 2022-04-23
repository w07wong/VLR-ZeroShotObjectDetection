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

        self.encoder = torchvision.models.resnet18(pretrained=True)

        # Modify input channel dimension of encoder
        with torch.no_grad():
            w = self.encoder.conv1.weight
            w = torch.cat([w, torch.full((64, 13, 7, 7), 0.01)], dim=1)
            self.encoder.conv1.weight = nn.Parameter(w)

        self.bb1 = nn.Linear(1000, 1000)
        self.bb2 = nn.Linear(1000, 512)
        self.bb3 = nn.Linear(512, 256)
        self.bb4 = nn.Linear(256, 128)
        self.bb5 = nn.Linear(128, 64)
        self.bb6 = nn.Linear(64, 32)
        self.bb7 = nn.Linear(32, 16)
        self.bb8 = nn.Linear(16, 4)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):

        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.bb1(x)
        x = F.relu(x)
        x = self.bb2(x)
        x = F.relu(x)
        x = self.bb3(x)
        x = F.relu(x)
        x = self.bb4(x)
        x = F.relu(x)
        x = self.bb5(x)
        x = F.relu(x)
        x = self.bb6(x)
        x = F.relu(x)
        x = self.bb7(x)
        x = F.relu(x)
        return torch.sigmoid(self.bb8(x))

    def save(self, dir_path, net_fname, net_label):
        net_path = os.path.join(dir_path, net_label + net_fname)
        torch.save(self.state_dict(), net_path)

    def load(self, dir_path, net_fname):
        net_path = os.path.join(dir_path, net_fname)
        state_dict = torch.load(net_path)
        self.load_state_dict(state_dict)
