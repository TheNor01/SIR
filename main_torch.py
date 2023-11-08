import numpy as np 
import pandas as pd 

import os

import torch
from glob import glob
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transform
from torch.utils.data import DataLoader

from bin.loaders import ImagesDataset
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from bin.utils import prepare_tensor_dataset,preprocess
from labels import id2label
import calendar

from bin.models.UNET import UNet
from bin.models.DEEPLAB import DeepLab50
from bin.metrics_eval import train_classifier,test_classifier


IMAGE_SIZE = [128, 128]
IMAGE_SHAPE = IMAGE_SIZE + [3,]



#to do:

"""

1) normalization
2) adam

3) standard deeplab + ppsnet

"""

if __name__ == '__main__':

    choosedModel = 0
    #https://www.kaggle.com/code/sudhupandey/cityscape-segmentation-unet-pytorch#kln-108


    train_path = glob("dataset/cityscapes_data/cityscapes_data/train_small/*")
    valid_path = glob("dataset/cityscapes_data/cityscapes_data/val_small/*")

    fig,ax = plt.subplots(5,2,figsize=(10,30))
    for i in range(1):
        img = plt.imread(train_path[i])
        #ax[i][0].imshow(img[:,:256])
        #ax[i][1].imshow(img[:,256:])

        #plt.show()

    mytransformsImage = transform.Compose(
        [
            transform.ToTensor(),
            #transform.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transform.RandomHorizontalFlip(p=0.9)
        ]
    )

    mytransformsLabel = transform.Compose(
        [
            transform.ToTensor(),
        ]
    )


    traindata = ImagesDataset(train_path, mytransformsImage, mytransformsLabel)
    # val dataset
    valdata = ImagesDataset(valid_path, mytransformsImage, mytransformsLabel)

    sample = traindata[0]
    #print(sample)


    batch_size = 4
    train_loader = DataLoader(traindata,batch_size)
    vaild_loader = DataLoader(valdata,1)

    X_train, Y_train, X_valid, Y_valid = prepare_tensor_dataset("dataset/cityscapes_data/cityscapes_data/train_small", "dataset/cityscapes_data/cityscapes_data/val_small",id2label)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_valid = np.array(X_valid)
    Y_valid = np.array(Y_valid)


    model = None
    modelString = ""
    if(choosedModel==0):
        model = UNet(3).float().to("cpu")
        modelString= "unet"
    elif(choosedModel==1):
        model = DeepLab50(len(id2label)) #to fix
        modelString= "deeplab"

    print("MODEL HAS BEEN CREATED...\n")

    EPOCHS = 40
    BATCH_SIZE = 16
    LR = 0.1
    #https://www.kaggle.com/code/sudhupandey/cityscape-segmentation-unet-pytorch

    current_GMT = time.gmtime()
    ts = calendar.timegm(current_GMT)
    print("Current timestamp:", ts)

    trained = train_classifier(model, train_loader,vaild_loader, exp_name=str(ts)+"_color", epochs = EPOCHS,lr=LR,momentum=0.5)

    exit()

    torch.save(model.state_dict(), "./checkpoint_model/"+modelString+".pth")

    accuracy_score, iou_score = test_classifier(trained,train_loader) #first on train

    print(accuracy_score,iou_score)

    accuracy_score, iou_score = test_classifier(trained,vaild_loader)

    print(accuracy_score,iou_score)