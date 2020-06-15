import numpy as np
import torch
from torch import nn


class YOLOLite(nn.Module):
    def __init__(self):
        super(YOLOLite, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5)
        )
        self.c4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.c5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(128, 128, 3, 1, 1),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5)
        )
        self.c6 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.c7 = nn.Sequential(
            nn.Conv2d(256, 125, 1, 1, 0),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(125 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 7 * 7 * 12),
            nn.Sigmoid()
        )

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        c6 = self.c6(c5)
        c7 = self.c7(c6).contiguous().view(-1, 6125)
        f1 = self.fc1(c7)
        f2 = self.fc2(f1).contiguous().view(-1, 7, 7, 12)
        return f2

    
        

