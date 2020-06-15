# README
## 文件作用
getDataSet.py文件随机抽取40%的验证集中的图片作为测试集（实验时使用，测试要求只测试test-image中的图片）.<br>
LoadData.py文件用于提取图片和标签,在训练VGG16网络时使用.<br>
LoadDataNew.py文件用于提取图片和标签，在训练YOLO-LITE网络时使用，与LoadData不同的时，LoadDataNew在提取图片时会将所有图片存在内存中，以便加速训练，这只是本人在处理这种数据集不大的情况下方便实验提速用的一种策略,显然当训练集很大时这种方法不奏效，但是这种方法在训练VGG时会出现存储空间不足的情况，因此训练两个网络用了不同的方法提取图片。<br>
YOLOLite.py文件是YOLO-LITE网络结构。<br>
YOLOLoss.py文件是损失函数。<br>
VGGYOLO.py是VGG16网络以及训练的文件。<br>
train.py是YOLO-LITE网络训练的文件。<br>
testTrainedYOLO.py是对训练好的YOLO-LITE模型测试的文件，该文件生成表示预测结果和原便签的txt文件，分别存储在yolodetection和yolotruth文件夹中，以便计算mAP和画图。<br>
testBestYOLO.py是对最好的YOLO-LITE模型测试的文件，该文件生成表示预测结果和原便签的txt文件，分别存储在yolodetection和yolotruth文件夹中，以便计算mAP和画图。<br>
testTrainedVGG.py是对训练好的VGG16模型测试的文件，该文件生成表示预测结果和原便签的txt文件，分别存储在yolodetection和yolotruth文件夹中，以便计算mAP和画图。<br>
testBestVGG.py是对最好参数的VGG16模型测试的文件，该文件生成表示预测结果和原便签的txt文件，分别存储在yolodetection和yolotruth文件夹中，以便计算mAP和画图。<br>
pascalvoc.py文件是计算mAP并画每类的PR曲线的。<br>
Plot.py文件在图片中画出原始便签框和预测的框。<br>
report.pdf是报告文件


## 软件环境和版本
本次实验我使用操作系统为Windows10，语言为Python3.8，所需的库有：<br>
numpy 1.18.0 
```shell
pip install numpy
```
torch 1.4.0
```shell
pip install Pytorch
```
tensorflow 2.2.0 
```shell
pip install tensorflow
```
tensorboarX 2.0
```shell
pip install tensorboarX
```
opencv-python 4.2.0.34
```shell
pip install opencv-python
```
## 下载数据及整理数据
数据集下载链接：https://cloud.tsinghua.edu.cn/d/af356cf803894d65b447/?p=%2FAIZOO&mode=list<br>
下载之后请解压到项目codes目录下,使得codes目录下含有train文件夹和val文件夹<br>
得到测试集（这一步的目的只是为了和自己实验的时候步骤保持一致，没有更改训练集中的数据）：<br>
```shell
python getDataSet.py
```
## 训练与测试模型
训练YOLO-LITE：<br>
```shell
python train.py
```
测试上一步训练得到的YOLO-LITE模型：<br>
```shell
python testTrainedYOLO.py
python pascalvoc.py
python pascalvoc.py -t 0.7
python pascalvoc.py -t 0.9
```
如果画图可以使用：<br>
```shell
Plot.py
```
测试最优参数YOLO-LITE模型：<br>
由于文件过大，请先下载参数，将params_best.pkl文件放置codes目录下进行测试
https://cloud.tsinghua.edu.cn/f/db83809187064f05ae38/?dl=1
```shell
python testBestYOLO.py
python pascalvoc.py
python pascalvoc.py -t 0.7
python pascalvoc.py -t 0.9
```
如果画图可以使用：<br>
```shell
Plot.py
```
训练VGG16模型：<br>
```shell
python VGGYOLO.py
```
测试上一步训练得到的VGG16模型：<br>
```shell
python testTrainedVGG.py
python pascalvoc.py
python pascalvoc.py -t 0.7
python pascalvoc.py -t 0.9
```
如果画图可以使用：<br>
```shell
Plot.py
```
测试最优参数VGG16模型：<br>
由于文件过大，请先下载参数，将VGG_best.pkl文件放置codes目录下进行测试
https://cloud.tsinghua.edu.cn/f/b9390c2a676f4961a4fe/?dl=1
```shell
python testBestVGG.py
python pascalvoc.py
python pascalvoc.py -t 0.7
python pascalvoc.py -t 0.9
```
如果画图可以使用：<br>
```shell
Plot.py
```

如果测试中遇到任何问题可以与我联系：电话、微信：13020025667
