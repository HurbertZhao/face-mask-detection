import cv2
import os


path = './test-image'
path1 = './yolodetection/'
path2 = './yolotruth/'

for root,dirs,files in os.walk(path):
    for i in range(len(files)):
        if(files[i][-3:] == 'jpg'):
            file_path = root + '/' + files[i]
            img = cv2.imread(file_path)
            cv2.imshow('img', img)

            h,w = img.shape[0],img.shape[1]
            with open(path1 + files[i][:-4] + ".txt", "r") as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    str_list = line.split(" ")
                    xmin = int(float(str_list[2]) * w)
                    ymin = int(float(str_list[3]) * h)
                    xmax = int(float(str_list[4]) * w)
                    ymax = int(float(str_list[5]) * h)
                    cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0))
                    cv2.putText(img, str_list[0], (xmin, ymin), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)


            with open(path2 + files[i][:-4] + ".txt", "r") as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    str_list = line.split(" ")
                    xmin = int(float(str_list[1]) * w)
                    ymin = int(float(str_list[2]) * h)
                    xmax = int(float(str_list[3]) * w)
                    ymax = int(float(str_list[4]) * h)
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0))

            cv2.imshow('img',img)
            cv2.waitKey()
            cv2.destroyAllWindows()



def plot_img(img,labels,preds):
    img = img.reshape(224,224,3)
    # for label in labels:
    #     cv2.rectangle(img, (10,10), (20,20), (0, 255, 0))
        # cv2.rectangle(img,(int(label[0]),int(label[1])),(int(label[2]),int(label[3])),(0,255,0))
    # for pred in preds:
    #     cv2.rectangle(img, (int(pred[0]),int(pred[1])),(int(pred[2]),int(pred[3])),(255,0,0))
    cv2.imshow('img',img)