import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class FCNet(nn.Module):
    def __init__(self, num_classes=4):
        super(FCNet, self).__init__()
        self.num_classes = num_classes
        self.linear1 = nn.Linear(8, 32)
        self.linear2 = nn.Linear(32, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear4 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
        self.name = 'FC'

    def forward(self, x):
        x1 = x[..., 0].long()
        x2 = x[..., 1:]
        x1 = torch.nn.functional.one_hot(x1, self.num_classes).float()
        x = torch.cat((x1, x2), -1)

        y = self.linear1(x)
        y = self.relu(y)

        y = self.linear2(y)
        y = self.relu(y)

        y = self.linear3(y)
        y = self.dropout(y)
        y = self.relu(y)

        y = self.linear4(y)

        y = torch.sum(y, dim=-2)
        y = torch.sigmoid(y)

        return y