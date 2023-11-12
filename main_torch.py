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
import torchvision
from bin.loaders import ImagesDataset
import time
from torch.utils.tensorboard import SummaryWriter
from config import labels
import calendar
import cv2
from bin.models.UNET import UNet
from bin.models.DEEPLAB import DeepLab50
from bin.models.RESNET import R2U_Net
#import segmentation_models_pytorch as smp
from bin.metrics_eval import train_classifier,test_classifier

from bin.utils import draw_segmentation_map,decode_segmap

#to do:

"""

1) normalization
2) adam

3) standard deeplab + ppsnet

"""

if __name__ == '__main__':

    choosedModel = 2
    customSize = 128 #for unet at least 128
    labels.init()

    
    #https://stackoverflow.com/questions/56650201/how-to-convert-35-classes-of-cityscapes-dataset-to-19-classes/64242989#64242989

    dictIDxColors = labels.id2labelValid

    ignore_index = 250
    #dictIDxColors[ignore_index] = (0,0,0)
    validLabels = list(dictIDxColors.keys())
    class_map = dict(zip(validLabels, range(len(validLabels))))

    #print(dictIDxColors)
    #print(class_map)
    
    voidIdLabels = (labels.voidLabels)
    
    dictNameXLabels = labels.namelabelValid
    voidIdLabels.add(-1)
    label_colous = dict(zip(range(len(validLabels)), dictIDxColors.values()))
    print(label_colous)
    #we have to define a undefined label? as void?


    print(class_map)
    #CREATE IMAGES DATASET

    path_data = './dataset/'

    train_data = ImagesDataset(
        path_data, 
        'train',
        validLabels,
        voidIdLabels,
        ignore_index,
        class_map,
        customSize
    )

    img = train_data[5]
    #print(img['image'],img['label'].shape)

    #fig,ax=plt.subplots(ncols=2,nrows=1,figsize=(16,8))

    imgTrain = img['image']
    maskTrain = img['label']

    print(imgTrain.shape)
    print(maskTrain.shape)

    #ax[0].imshow(imgTrain.permute(1, 2, 0))
    #ax[1].imshow(maskTrain)
    #plt.show()
    #plt.clf()

    #print(train_data[0]['image'].shape)
    #print(train_data[0]['label'].shape)


    val_data = ImagesDataset(
        path_data, 
        'val',
        validLabels,
        voidIdLabels,
        ignore_index,
        class_map,
        customSize
    )
    


    print(val_data[0]['image'].shape)
    print(val_data[0]['label'].shape)



    EPOCHS = 2
    BATCH_SIZE = 15
    LR = 0.001
    WORKERS = 4

    train_loader = DataLoader(
        train_data,
        batch_size = BATCH_SIZE,
        #shuffle=True,
        num_workers = WORKERS,
    )

    val_loader = DataLoader(
        val_data,
        batch_size = BATCH_SIZE,
        num_workers = WORKERS,
    )


    #============================

    class_values= list(labels.fullLabelColor.keys())

    RGB_values = list(labels.fullLabelColor.values())



    model = None
    modelString = ""
    if(choosedModel==0):
        #model = UNet(3,len(validLabels)).float().to("cpu")
        #modelString= "unet"
        model = R2U_Net(img_ch=3,output_ch=len(class_values)).to("cpu")
        #modelString= "resnet"
        modelString= "resnet_online"
    elif(choosedModel==1):
        model = DeepLab50(len(validLabels)) #to fix
        modelString= "deeplab"
    elif(choosedModel==2):
        model = UNet(3,classes=len(class_values)).to("cpu")
        modelString= "unet"

    print(modelString+"-- MODEL HAS BEEN CREATED...\n")

    """
    train_epochs = 8
    n_classes = 19
    batch_size = 1
    num_workers = 1
    learning_rate = 1e-6
    """

    #https://www.kaggle.com/code/sudhupandey/cityscape-segmentation-unet-pytorch

    current_GMT = time.gmtime()
    ts = calendar.timegm(current_GMT)
    print("Current timestamp:", ts)

    #use ignore index
    doTrain = 1
    doVal = 0
    trained=None
    if(doTrain):
        trained = train_classifier(model, train_loader,val_loader, exp_name=str(ts)+"_color", epochs = EPOCHS,lr=LR,momentum=0.5)
        torch.save(model.state_dict(), "./checkpoint_model/"+modelString+".pth")
    else:
        model.load_state_dict(torch.load("./checkpoint_model/"+modelString+".pth"))
        trained = model
    print("TRAINING COMPLETED")

    if(doVal):
        accuracy_score, iou_score = test_classifier(trained,train_loader,len(validLabels),label_colous) #first on train
        print("TRAINING")
        print(accuracy_score,iou_score)
        accuracy_score, iou_score = test_classifier(trained,val_loader)

        print("VALIDATION")
        print(accuracy_score,iou_score)




    #Single inference
    dummy = (train_data[0]['image']).to("cpu")
    dummyLabel = (train_data[0]['label']).to("cpu")

    gt = dummyLabel.data.cpu().numpy()

    plt.imshow(dummy.permute(1, 2, 0)  )
    plt.show()
    plt.clf()

    print(dummyLabel.shape)
    print(dummyLabel)
    print(gt.shape)

    model.eval()

    with torch.no_grad():
        val_pred = model(dummy.unsqueeze(0)).to("cpu")

        print(dummy.shape,dummyLabel.shape,val_pred.shape)

        mask = draw_segmentation_map(val_pred,RGB_values)


        #mask2 = decode_segmap(gt[0],len(validLabels),label_colous)
        #plt.imshow(mask2)
        #plt.show()

        cv2.imshow('Segmented image', mask)
        cv2.waitKey(0)

        

        #backtorgb = cv2.cvtColor(dummyLabel.numpy(),cv2.COLOR_GRAY2RGB)

        cv2.imwrite("./my_out.png",mask)
        cv2.imwrite("./my_mask.png",dummyLabel.numpy())