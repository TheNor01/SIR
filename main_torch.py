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


    for i in range(1):
        img = plt.imread(train_path[i])
        #ax[i][0].imshow(img[:,:256])
        #ax[i][1].imshow(img[:,256:])

        image,label = img[:,:int(img.shape[1]/2)],img[:,int(img.shape[1]/2):]

        transformTest = transform.Compose([
            transform.ToPILImage(),
            transform.Resize((512,640)),
            transform.ToTensor()
        ])

        im = transformTest(image)
        lb = transformTest(label)
        print(im)

        print(type(label))
        print(label)

        #plt.show()




    #======================

    transformPreprocess = transform.Compose([
            transform.ToPILImage(),
            transform.Resize((customSize,customSize)),
            transform.ToTensor()
        ])


    colors = labels.id2labelValid
    voidLabels = labels.voidLabels
    nameLabels = labels.namelabelValid
    validLabels = list(colors.keys())
    ignore_index = 250


    #we have to define a undefined label? as void?


    print("RGB colors: "+ str(len(colors)))
    print(validLabels)
    print(nameLabels)
    print("void colors: "+ str(len(voidLabels)))
    print(voidLabels)

    rgbColors = len(colors)
    print(rgbColors)



    traindataset = ImagesDataset(train_path, validLabels,voidLabels,ignore_index,rgbColors,transformPreprocess)
    print(traindataset[0]['image'].shape)
    print(traindataset[0]['label'].shape)

    print(traindataset[0]['label'])


    
    # val dataset
    valdataset = ImagesDataset(valid_path, validLabels,voidLabels,ignore_index,rgbColors,transformPreprocess)

    #Normalization step ====

    #Nan value...
    """
    Per poter allenare un classificatore su questi dati, abbiamo bisogno di normalizzarli in modo che essi abbiano media nulla e deviazione standard unitaria. Calcoliamo media e varianza dei
    pixel contenuti in tutte le immagini del training set:

    m = np.zeros(3)
    print(m)
    for sample in traindataset:
        print(type(sample))
        m+=sample['image'].sum(1).sum(1).numpy() 

    m=m/(len(traindataset)*customSize*customSize)
    
    s = np.zeros(3)
    for sample in traindataset:
        s+=((sample['image']-torch.Tensor(m).view(3,1,1))**2).sum(1).sum(1).numpy()
    
    s=np.sqrt(s/(len(traindataset)*customSize*customSize))


    print("Medie",m)

    print(type(m))
    print("Dev.Std.",s)

    """
    
    #============================


    sample = traindataset[0]
    #print(sample)

    """
    normalizedTransform  = transform.Compose([
            transform.ToPILImage(),
            transform.Resize((customSize,customSize)),
            transform.ToTensor(),
            #transform.Normalize(m,s),
        ])
    
    testTransform  = transform.Compose([
            transform.ToPILImage(),
            transform.Resize((customSize,customSize)),
            transform.ToTensor(),
        ])
    
    """

    #normalizedTraindataset = ImagesDataset(train_path, normalizedTransform)

    #testDataset = ImagesDataset(valid_path, testTransform)

    """
    print(normalizedTraindataset[0]['image'].shape)
    print(normalizedTraindataset[0]['label'].shape)

    print(testDataset[0]['image'].shape)
    print(testDataset[0]['label'].shape)
    """



    



    model = None
    modelString = ""
    if(choosedModel==0):
        #model = UNet(3,rgbColors).float().to("cpu")
        model = R2U_Net().to("cpu")
        modelString= "unet"
    elif(choosedModel==1):
        model = DeepLab50(len(labels.id2label)) #to fix
        #model = smp.DeepLabV3(classes=numClasses)
        modelString= "deeplab"

    print("MODEL HAS BEEN CREATED...\n")

    EPOCHS = 4
    BATCH_SIZE = 15
    LR = 0.01

    """
    learning_rate = 1e-6
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

    train_loader = DataLoader(traindataset,BATCH_SIZE,num_workers=4)
    vaild_loader = DataLoader(valdataset,num_workers=4)


    doTrain = 0

    trained=None
    if(doTrain):
        trained = train_classifier(model, train_loader,vaild_loader, exp_name=str(ts)+"_color", epochs = EPOCHS,lr=LR,momentum=0.5)
        torch.save(model.state_dict(), "./checkpoint_model/"+modelString+".pth")
    else:
        model.load_state_dict(torch.load("./checkpoint_model/"+modelString+".pth"))
        trained = model

    print("TRAINING COMPLETED")

    #accuracy_score, iou_score = test_classifier(trained,train_loader) #first on train

    #print(accuracy_score,iou_score)

    accuracy_score, iou_score = test_classifier(trained,vaild_loader)

    print(accuracy_score,iou_score)