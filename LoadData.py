import os
import xml.sax
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

class LabelReader(xml.sax.ContentHandler):
    def __init__(self, LableOutput,root):
        self.CurrentData = ""
        self.name = []
        self.xmin = []
        self.ymin = []
        self.xmax = []
        self.ymax = []
        self.w = 0
        self.h = 0
        self.filename = ""
        self.LableOutput = LableOutput
        self.root = root

    def clear(self):
        self.CurrentData = ""
        self.name = []
        self.xmin = []
        self.ymin = []
        self.xmax = []
        self.ymax = []
        self.w = 0
        self.h = 0
        self.filename = ""

    def startElement(self, tag, attributes):
        self.CurrentData = tag

    def endElement(self, tag):    
        self.CurrentData = ""

    def characters(self, content):
        if self.CurrentData == "name":
            self.name.append(content != "face")
        elif self.CurrentData == "xmin":
            self.xmin.append(float(content))
        elif self.CurrentData == "xmax":
            self.xmax.append(float(content))
        elif self.CurrentData == "ymin":
            self.ymin.append(float(content))
        elif self.CurrentData == "ymax":
            self.ymax.append(float(content))
        elif self.CurrentData == "filename":
            self.filename = content[:-4]
        elif self.CurrentData == "width":
            self.w = float(content)
        elif self.CurrentData == "height":
            self.h = float(content)

    def endDocument(self):
        if self.w == 0 or self.h == 0:
            path = self.root + '/' + self.filename + ".jpg"
            img = cv2.imread(path)
            self.h = img.shape[0]
            self.w = img.shape[1]
        coordinate = anchor2loc(w=self.w, h=self.h, xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax)
        target = encoder(coordinate, torch.tensor(self.name, dtype=bool))
        self.LableOutput[self.filename] = target
        self.clear()

def anchor2loc(w, h, xmin, xmax, ymin, ymax):
    '''
    loc: Length * [x, y, h, w]
    原始数据中x为横坐标，y为纵坐标
    导出数据中x为横坐标，y为纵坐标
    归一化
    '''
    loc = torch.zeros((len(xmin), 4))
    scale_w = 1 / w
    scale_h = 1 / h
    for i in range(len(xmin)):
        loc[i][0] = 0.5 * (xmax[i] + xmin[i]) * scale_w
        loc[i][1] = 0.5 * (ymax[i] + ymin[i]) * scale_h
        loc[i][2] = (xmax[i] - xmin[i]) * scale_w
        loc[i][3] = (ymax[i] - ymin[i]) * scale_h

    return loc


def getLabel(path="./train"):
    Label = dict()
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    D = LabelReader(Label,path)
    parser.setContentHandler(D)
    TrainFileList = os.listdir(path)
    length = len(TrainFileList)
    for i in range(length):
        if TrainFileList[i][-4:] == ".xml":
            parser.parse(path +'/' + TrainFileList[i])
    return Label


class LoadData(Dataset):

    def __init__(self, root_dir="./train", transform=None, Label=None):
        self.root_dir = root_dir
        self.transform = transform
        allfiles = os.listdir(self.root_dir)
        self.images = [i for i in allfiles if i[-4:] == ".jpg"]
        self.Label = Label
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        path = self.root_dir + "/" + self.images[index]
        name = self.images[index][:-4]
        img = cv2.imread(path)
        img = img[:,:,:3]
        img = torch.from_numpy(cv2.resize(img, (224, 224))).permute(2, 0, 1)
        img = (img * 255).byte()
        label = self.Label[name]
        sample = {'image':img, 'label':label}
        
        if self.transform:
            sample = self.transform(sample)
        return sample

def encoder(boxes,labels):
    '''
    boxes (tensor) [[x1,y1,w,h],[]]
    labels (tensor) [...]
    return 7x7x30
    '''
    grid_num = 7
    target = torch.zeros((grid_num,grid_num,12), dtype=torch.float32)
    cell_size = 1./grid_num
    wh = boxes[:,2:]
    cxcy = boxes[:,:2]
    for i in range(cxcy.size()[0]):
        cxcy_sample = cxcy[i]
        ij = (cxcy_sample/cell_size).ceil()-1 #
        target[int(ij[1]),int(ij[0]),4] = 1
        target[int(ij[1]),int(ij[0]),9] = 1
        target[int(ij[1]),int(ij[0]),int(labels[i])+10] = 1
        xy = ij * cell_size
        delta_xy = (cxcy_sample - xy) / cell_size
        target[int(ij[1]),int(ij[0]),2:4] = wh[i]
        target[int(ij[1]),int(ij[0]),:2] = delta_xy
        target[int(ij[1]),int(ij[0]),7:9] = wh[i]
        target[int(ij[1]),int(ij[0]),5:7] = delta_xy
    return target



