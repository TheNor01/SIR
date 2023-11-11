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
from config import labels
import calendar

from bin.models.UNET import UNet
from bin.models.DEEPLAB import DeepLab50
from bin.models.RESNET import R2U_Net
#import segmentation_models_pytorch as smp
from bin.metrics_eval import train_classifier,test_classifier

from bin.utils import decode_segmap

#to do:

"""

1) normalization
2) adam

3) standard deeplab + ppsnet

"""

if __name__ == '__main__':

    choosedModel = 0
    customSize = 64 #for unet at least 128
    labels.init()
    #https://www.kaggle.com/code/sudhupandey/cityscape-segmentation-unet-pytorch#kln-108


    train_path = glob("dataset/cityscapes_data/cityscapes_data/train_small/*")
    valid_path = glob("dataset/cityscapes_data/cityscapes_data/val_small/*")

    #======================

    
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

    #we have to define a undefined label? as void?

    """
    print("RGB colors: "+ str(len(dictIDxColors)))
    print(validLabels)
    print(dictIDxColors)
    print(dictNameXLabels)
    print("void colors: "+ str(len(voidIdLabels)))
    print(voidIdLabels)
    """
    
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
    #ax[0].imshow(img['image'].permute(1, 2, 0))
    #ax[1].imshow(img['label'].permute(1, 2, 0))
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
    

    img = val_data[5]
    #print(img['image'],img['label'].shape)

    fig,ax=plt.subplots(ncols=2,nrows=1,figsize=(16,8))
    ax[0].imshow(img['image'].permute(1, 2, 0))
    #ax[1].imshow(img['label'].permute(1, 2, 0))
    ax[1].imshow(img['label'].permute(1, 2, 0))

    plt.show()
    plt.clf()
    
    print(torch.unique(img['label']))
    print(len(torch.unique(img['label'])))


    """
    def encode_segmap(mask):
    #remove unwanted classes and recitify the labels of wanted classes
        for _voidc in voidIdLabels:
            mask[mask == _voidc] = ignore_index
        for _validc in validLabels:
            mask[mask == _validc] = class_map[_validc]
        return mask

    res=encode_segmap(img['label'].clone())
    print(res.shape)
    print(torch.unique(res))
    print(len(torch.unique(res)))

    fig,ax=plt.subplots(ncols=2,figsize=(12,10))  
    ax[0].imshow(res,cmap='gray')
    ax[1].imshow(res1)

    res1=decode_segmap(res.clone(),len(validLabels),label_colous)
    """

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


    model = None
    modelString = ""
    if(choosedModel==0):
        #model = UNet(3,len(validLabels)).float().to("cpu")
        #modelString= "unet"
        model = R2U_Net(img_ch=3,output_ch=len(validLabels)).to("cpu")
        modelString= "resnet"
    elif(choosedModel==1):
        model = DeepLab50(len(validLabels)) #to fix
        modelString= "deeplab"

    print("MODEL HAS BEEN CREATED...\n")

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
    doTrain = 0
    trained=None
    if(doTrain):
        trained = train_classifier(model, train_loader,val_loader, exp_name=str(ts)+"_color", epochs = EPOCHS,lr=LR,momentum=0.5)
        torch.save(model.state_dict(), "./checkpoint_model/"+modelString+".pth")
    else:
        model.load_state_dict(torch.load("./checkpoint_model/"+modelString+".pth"))
        trained = model
    print("TRAINING COMPLETED")

    #accuracy_score, iou_score = test_classifier(trained,train_loader) #first on train

    #Single inference
    model.eval()
    
    dummy = (train_data[0]['image']).to("cpu")
    dummyLabel = (train_data[0]['label']).to("cpu")


    plt.imshow(dummy.permute(1, 2, 0)  )
    plt.show()
    plt.clf()

    val_pred = model(dummy.unsqueeze(0))

    print(dummy.shape,dummyLabel.shape,val_pred.shape)




    prediction = val_pred.data.max(1)[1].cpu().numpy()
    ground_truth = dummyLabel.data.cpu().numpy()


    plt.imshow(ground_truth[:,:,0], cmap='gray')
    plt.show()
    plt.clf()


    print(label_colous)

    """
    with torch.no_grad():

        # Model Prediction
        decoded_pred = decode_segmap(prediction[0],len(validLabels),label_colous)
        plt.imshow(decoded_pred)
        plt.show()
        plt.clf()
        
        # Ground Truth
        decode_gt = decode_segmap(ground_truth,len(validLabels),label_colous)
        plt.imshow(decode_gt)
        plt.show()
    """



    #print(accuracy_score,iou_score)
    
    exit()
    accuracy_score, iou_score = test_classifier(trained,vaild_loader)

    print(accuracy_score,iou_score)