import os
import numpy as np
import torch
from torch import nn
from torchvision.models import vgg16
from torch.utils.data import Dataset, DataLoader
from YOLOLite import YOLOLite
from YOLOLoss import yoloLoss
from LoadData import LoadData, getLabel


def get_vgg16():
    model = vgg16(pretrained=True)
    features = list(model.features)[:31]
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    return nn.Sequential(*features)


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.vgg16 = get_vgg16()
        self.classifier = torch.nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096,  7 * 7 * 12),
            nn.Sigmoid())


    def forward(self, x):
        vgg = self.vgg16(x).contiguous().view(-1, 512 * 7 * 7)
        out = self.classifier(vgg).contiguous().view(-1, 7, 7, 12)
        return out


class VGG16Trainer(nn.Module):
    def __init__(self):
        super(VGG16Trainer, self).__init__()
        self.net = VGG16()
        self.loss = yoloLoss()
        self.optimizer = self.get_optimizer()

    def forward(self, x, y):
        pred = self.net(x)
        loss = self.loss(pred, y)
        return loss

    def get_optimizer(self):
        lr = 3e-4
        params = []
        for key, value in dict(self.named_parameters()).items():
            if key.startswith('features'):
                params += [{'params': [value], 'lr': lr * 1}]
            else:
                params += [{'params': [value], 'lr': lr}]
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)

    def train_step(self, x, y):
        self.optimizer.zero_grad()
        loss = self.forward(x, y)
        loss.backward()
        self.optimizer.step()
        return loss

    def save(self, path=None):
        torch.save(self.state_dict(), path)


def train():
    batch_size = 10
    epochs = 50
    bestloss = 1e10
    learning_rate = 5e-4
    Trainer = VGG16Trainer().cuda()

    path = './train'
    trainLabel = getLabel(path)
    traindata = LoadData(path, Label=trainLabel)
    dataloader = DataLoader(traindata, batch_size, shuffle=True)
    valLabel = getLabel('./val')
    valdata = LoadData('./val', Label=valLabel)
    valdataloader = DataLoader(valdata, batch_size, shuffle=True)
    count = 0
    for epoch in range(epochs):

        if count == 5:
            learning_rate *= 0.5
            for param_group in Trainer.optimizer.param_groups:
                param_group['lr'] = learning_rate

        if count == 10:
            break


        Trainer.train()
        totalloss = 0
        for i_batch, batch_data in enumerate(dataloader):
            image = batch_data['image']
            label = batch_data['label'].cuda()
            image = image.cuda().float() / 255.
            loss = Trainer.train_step(image, label)
            totalloss += loss
        print('train loss:')
        print(totalloss / len(dataloader))

        Trainer.eval()
        valloss = 0
        with torch.no_grad():
            for i_batch, batch_data in enumerate(valdataloader):
                image = batch_data['image']
                label = batch_data['label'].cuda()
                image = image.cuda().float() / 255.
                valloss += Trainer.forward(image, label)
        print('val loss:')
        valloss_a = valloss / len(valdataloader)
        print(valloss_a)
        if valloss_a < bestloss:
            bestloss = valloss_a
            print('saved')
            Trainer.save('VGG.pkl')
            count = 0
        else: count += 1


if __name__ == "__main__":
    train()