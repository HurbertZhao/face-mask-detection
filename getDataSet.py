import os
import random
import shutil

fileDir = './val/'
tarDir = './test/'
rate = 0.4
isExists = os.path.exists(tarDir)
if isExists==False:
    os.makedirs(tarDir)
pathDir = os.listdir(fileDir)  #scan
images = [i for i in pathDir if i[-4:] == ".jpg"]

filenumber = len(images)
picknumber = int(filenumber * rate)
print('total {} pictures'.format(filenumber))
print('moved {} pictures to {}'.format(picknumber, tarDir))

images = random.sample(images, picknumber)
for image in images:
    name = image[:-4]
    shutil.move(fileDir + name + '.jpg', tarDir + name + '.jpg')
    shutil.move(fileDir + name + '.xml', tarDir + name + '.xml')
    print(name)
print('succeed moved {} pictures from {} to {}'.format(picknumber, fileDir, tarDir))
