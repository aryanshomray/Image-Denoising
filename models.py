import numpy as np
from PIL import Image
import torch
from torch.nn import Conv2d, ConvTranspose2d
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_ssim
import os
import wandb


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = Conv2d(3, 8, (3, 3), 1, 1)
        self.layer2 = Conv2d(8, 16, (3, 3), 1, 1)
        self.layer3 = Conv2d(16, 32, (3, 3), 1, 1)
        self.layer4 = Conv2d(32, 64, (3, 3), 1, 1)
        self.layer5 = Conv2d(64, 32, (3, 3), 1, 1)
        self.layer6 = Conv2d(32, 16, (3, 3), 1, 1)
        self.layer7 = Conv2d(16, 8, (3, 3), 1, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        x = F.relu(x)
        x = self.layer5(x)
        x = F.relu(x)
        x = self.layer6(x)
        x = F.relu(x)
        x = self.layer7(x)
        x = F.relu(x)
        return x


class unet(torch.nn.Module):

    def __init__(self):
        super(unet, self).__init__()
        self.layer1 = Conv2d(8, 32, (3, 3), 1, 1)
        self.layer2 = Conv2d(32, 32, (3, 3), 1, 1)
        self.layer3 = Conv2d(32, 32, (3, 3), 1, 1)
        self.layer4 = Conv2d(32, 64, (3, 3), 2, 1)
        self.layer5 = Conv2d(64, 64, (3, 3), 1, 1)
        self.layer6 = Conv2d(64, 64, (3, 3), 1, 1)
        self.layer7 = Conv2d(64, 128, (3, 3), 2, 1)
        self.layer8 = Conv2d(128, 128, (3, 3), 1, 1)
        self.layer9 = Conv2d(128, 128, (3, 3), 1, 1)
        self.layer10 = ConvTranspose2d(128, 64, (3, 3), 2, 1, output_padding=1)
        self.layer11 = Conv2d(64, 64, (3, 3), 1, 1)
        self.layer12 = Conv2d(64, 64, (3, 3), 1, 1)
        self.layer13 = ConvTranspose2d(64, 32, (3, 3), 2, 1, output_padding=1)
        self.layer14 = Conv2d(32, 32, (3, 3), 1, 1)
        self.layer15 = Conv2d(32, 32, (3, 3), 1, 1)
        self.layer16 = Conv2d(32, 16, (3, 3), 1, 1)
        self.layer17 = Conv2d(16, 3, (3, 3), 1, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x3 = self.layer3(x)
        x3 = F.relu(x3)
        x = self.layer4(x3)
        x = F.relu(x)
        x = self.layer5(x)
        x = F.relu(x)
        x6 = self.layer6(x)
        x6 = F.relu(x6)
        x = self.layer7(x6)
        x = F.relu(x)
        x = self.layer8(x)
        x = F.relu(x)
        x = self.layer9(x)
        x = F.relu(x)
        x10 = self.layer10(x)
        x10 = F.relu(x10)
        x = torch.add(x6, x10)
        x = self.layer11(x)
        x = F.relu(x)
        x = self.layer12(x)
        x = F.relu(x)
        x13 = self.layer13(x)
        x13 = F.relu(x13)
        x = torch.add(x3, x13)
        x = F.relu(x)
        x = self.layer14(x)
        x = F.relu(x)
        x = self.layer15(x)
        x = F.relu(x)
        x = self.layer16(x)
        x = F.relu(x)
        x = self.layer17(x)
        x = F.relu(x)
        return x
