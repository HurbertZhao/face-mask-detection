import numpy as np
import torch
import os
from torch import nn
from predict import decoder
from VGGYOLO import VGG16Trainer
from LoadDataNew import LoadData, getLabel
from torch.utils.data import Dataset, DataLoader
import datetime

path = './test-image'

Label = getLabel(path)
data = LoadData(path, Label=Label)
dataloader = DataLoader(data, batch_size=1, shuffle=False)
Names = list(Label.keys())

model = VGG16Trainer()
model.load_state_dict(torch.load('VGG_best.pkl'))

model.eval()
totaltime = 0.
d_path = './yolodetection/'
g_path = './yolotruth/'

isExists = os.path.exists(d_path)
if isExists==False:
    os.makedirs(d_path)

isExists = os.path.exists(g_path)
if isExists==False:
    os.makedirs(g_path)

for i_batch, batch_data in enumerate(dataloader):

    label = batch_data['label']
    image = batch_data['image'][0]
    name = Names[i_batch]
    starttime = datetime.datetime.now()
    img = image.float() / 255.
    img = img.view(1,3,224,224)
    label = label.reshape(1, 7, 7, 12)
    pred = model.net(img)
    boxes, cls_indexs, probspred = decoder(pred)
    endtime = datetime.datetime.now()
    totaltime += (endtime - starttime).seconds + (endtime - starttime).microseconds*1e-6


    boxes_n = np.array(boxes)
    cls_indexs_n = np.array(cls_indexs)
    prob_n = np.array(probspred)
    f = open(d_path + name + '.txt', 'w')
    for i in range(boxes_n.shape[0]):
        string_pred =  str(cls_indexs_n[i]) + ' '
        string_pred = string_pred + str(prob_n[i]) + ' '
        for j in range(4):
            string_pred = string_pred + str(boxes_n[i, j]) + ' '
        string_pred = string_pred + "\n"
        f.write(string_pred)
    f.close()

    boxes1, cls1, _ = decoder(label)
    boxes1_n = np.array(boxes1)
    cls1_n = np.array(cls1)
    f = open(g_path + name + '.txt', 'w')
    for i in range(boxes1_n.shape[0]):
        string = str(cls1_n[i]) + ' '
        for j in range(4):
             string = string + str(boxes1_n[i, j]) + ' '
        string = string + "\n"
        f.write(string)
    f.close()

print('average time:')
print(totaltime/len(dataloader))





