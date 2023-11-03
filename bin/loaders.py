
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

class ImagesDataset(Dataset):
    
    def __init__(self, images_path ,transform_img=None ,transform_label=None):
        
        self.images_path = images_path
        self.transform_img = transform_img
        self.transform_label = transform_label

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        
        img = plt.imread(self.images_path[idx])
        image,label = img[:,:int(img.shape[1]/2)],img[:,int(img.shape[1]/2):]
    
        if self.transform_img:
            image = self.transform_img(image)
            
        if self.transform_label:
            label = self.transform_label(label)
            
        return image, label


