import numpy as np
from PIL import Image
import torch
from torch.nn import Conv2d, ConvTranspose2d
import torch.nn.functional as F
from torch.optim import Adam
import pandas as pd
import pytorch_ssim
import os
import random
from torch.utils.data import DataLoader
from time import time

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def Print(txt):
    print(str(txt))


import torch.distributed as dist



print(torch.cuda.device_count())
Print(device)

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


class main_model(torch.nn.Module):

    def __init__(self):
        super(main_model, self).__init__()
        self.model = Model().to(device)
        self.unet = unet().to(device)
        self.model.train()
        self.unet.train()

    def forward(self, x):
        x = self.model(x)
        x = self.unet(x)
        return x


class Dataset(torch.utils.data.Dataset):

    def __init__(self, type='train', data='Sony', bit=8):
        super(Dataset, self).__init__()
        self.type = type
        self.csv = pd.read_csv('{}_{}_list.txt'.format(
            data, type), delimiter=' ', header=None)
        if data == 'Sony':
            self.image_size = (2848, 4256, 3)
        else:
            self.image_size = (4032, 6032, 3)
        self.bit = bit

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, idx):
        train = np.memmap(
            self.csv.iat[idx, 0], dtype=np.uint16, shape=self.image_size)/(2**self.bit)
        test = np.memmap(
            self.csv.iat[idx, 1], dtype=np.uint16, shape=self.image_size)/(2**self.bit)
        return {'train': train.astype('float32'), 'test': test.astype('float32')}


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        x = random.randint(0, sample['train'].shape[2] - self.output_size[1])
        y = random.randint(0, sample['train'].shape[1] - self.output_size[0])
        train = sample['train']
        test = sample['test']
        train = train[:, y:y+self.output_size[0], x:x+self.output_size[1], :]
        test = test[:, y:y+self.output_size[0], x:x+self.output_size[1], :]
        return {'train': train, 'test': test}


class ToTensor(object):
    def __call__(self, sample):
        train, test = sample['train'], sample['test']
        train = train.permute((0, 3, 1, 2))
        test = test.permute((0, 3, 1, 2))
        return {'train': (train).to(device),
                'test':  (test).to(device)}


class Loss(torch.nn.Module):

    def __init__(self):
        super(Loss, self).__init__()
        self.loss = torch.nn.MSELoss()

    def __call__(self, y_true, y_pred):
        return self.loss(y_true, y_pred)


class Training:

    def __init__(self):
        self.Epochs = 1000
        self.lr = 1e-4
        self.batch_size = 16
        self.Network = main_model().to(device)
        self.Network=torch.nn.DataParallel(self.Network, device_ids=[0, 1])
        self.loss = Loss().to(device)
        self.sony_dataset_train = DataLoader(
            Dataset('train', 'Sony', 16), batch_size=self.batch_size)
        self.fuji_dataset_train = DataLoader(
            Dataset('train', 'Fuji', 16), batch_size=self.batch_size)
        self.crop = RandomCrop((512, 512))
        self.opt = Adam(self.Network.parameters(), self.lr)
        self.loading()
    def train(self):
        for epoch in range(self.Epochs):
            Print('Epoch:{}/{}:'.format(epoch, self.Epochs))
            for idx, data in enumerate((self.sony_dataset_train)):
                a=time()
                print(data['train'].size())
                print(data['test'].size())
                data = self.crop(data)
                data = ToTensor()(data)
                out = self.Network(data['train'])
                losses = self.loss(out, data['test'])
                Print('Loss:{}'.format(losses.item()))
                self.opt.zero_grad()
                losses.backward()
                self.opt.step()
                del data
                print(time()-a)
            for idx, data in enumerate((self.fuji_dataset_train)):
                a=time()
                print(data['train'].size())
                data = self.crop(data)
                data = ToTensor()(data)
                out = self.Network(data['train'])
                losses = self.loss(out, data['test'])
                Print('Loss:{}'.format(losses.item()))
                self.opt.zero_grad()
                losses.backward()
                self.opt.step()
                del data
                print(time()-a)
            torch.save(self.Network.state_dict(),'model{}.pt'.format(epoch))
    def saving(self):
        torch.save(self.Network.state_dict(), 'model.pt')

    def loading(self):
        self.Network.load_state_dict(torch.load('model1.pt'))
        self.Network.train()
        print('Loaded!!')

a = Training()
a.train()
a.loading()
a.saving()
