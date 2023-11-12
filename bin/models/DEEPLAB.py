from torch import nn
from torchvision.models.segmentation import deeplabv3_resnet50

def GetDeeplabModel(num_classes):
    model =  deeplabv3_resnet50(weights='DEFAULT',classes=num_classes)

    for param in model.parameters():
            param.requires_grad = False
    
    model.classifier[4] = nn.Conv2d(256, num_classes, 1) # decoder to -->adapt last layer to our classes
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1) # decoder to -->adapt last layer to our classes

    return model