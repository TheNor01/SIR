from torch import nn
import segmentation_models_pytorch as smp


class DeepLab50(nn.Module):
    def __init__(self,numClasses):
        self.model =  smp.DeepLabV3(classes=numClasses)