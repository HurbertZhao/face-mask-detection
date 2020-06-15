import os
import xml.sax
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from YOLOLite import YOLOLite
from YOLOLoss import yoloLoss
from LoadDataNew import LoadData, getLabel
from tensorboardX import SummaryWriter



class YOLOLiteTrainer(nn.Module):
    def __init__(self):
        super(YOLOLiteTrainer, self).__init__()
        self.net = YOLOLite()
        self.loss = yoloLoss()
        self.optimizer = self.get_optimizer()


    def forward(self, x, y):
        pred = self.net(x)
        loss = self.loss(pred, y)
        return loss

    def get_optimizer(self):
        lr = 1e-3
        params = []    
        for key, value in dict(self.named_parameters()).items():
            if key.startswith('features'):
                params += [{'params': [value], 'lr': lr * 1}]
            else:
                params += [{'params': [value], 'lr': lr}]
        return torch.optim.SGD(params,lr = lr, momentum=0.9, weight_decay=5e-4)
    
    def train_step(self, x, y):
        self.optimizer.zero_grad()
        loss = self.forward(x, y)
        loss.backward()
        self.optimizer.step()
        return loss

    def save(self,name):
        torch.save(self.state_dict(), name)

def train():
    #param
    batch_size = 10
    epochs = 50
    learning_rate = 5e-4
    Trainer = YOLOLiteTrainer().cuda()
    path = './train'

    writer = SummaryWriter('runs/test')
    trainLabel = getLabel(path)
    traindata = LoadData(path,Label=trainLabel)
    dataloader = DataLoader(traindata, batch_size, shuffle=True)
    valLabel = getLabel('./val')
    valdata = LoadData('./val', Label=valLabel)
    valdataloader = DataLoader(valdata, batch_size, shuffle=True)

    bestloss = 1e5
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
        print('epoch:')
        print(epoch)
        for i_batch, batch_data in enumerate(dataloader):
            image = batch_data['image']
            label = batch_data['label'].cuda()
            image = image.cuda().float() / 255.
            loss = Trainer.train_step(image, label)
            totalloss += loss

        print('train loss:')
        print(totalloss / len(dataloader))
        writer.add_scalar('trainloss', totalloss / len(dataloader), global_step=epoch)
        Trainer.eval()
        valloss = 0
        with torch.no_grad():
            for i_batch, batch_data in enumerate(valdataloader):
                image = batch_data['image']
                label = batch_data['label'].cuda()
                image = image.cuda().float() / 255.
                valloss += Trainer.forward(image,label)
        print('val loss:')
        valloss_a = valloss / len(valdataloader)
        writer.add_scalar('validloss', valloss_a, global_step=epoch)
        print(valloss_a)
        if valloss_a < bestloss:
            bestloss = valloss_a
            print('saved')
            Trainer.save('params_test.pkl')
            count = 0
        else: count += 1




if __name__ == "__main__":
    train()


