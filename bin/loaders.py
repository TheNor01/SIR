
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

class ImagesDataset(Dataset):
    
    def __init__(self, images_path,transform=None):
    
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        
        img = plt.imread(self.images_path[idx])
        image,label = img[:,:int(img.shape[1]/2)],img[:,int(img.shape[1]/2):]

        #im = Image.fromarray(image.astype('uint8'), 'RGB')
        #lb = Image.fromarray(label.astype('uint8'), 'RGB')

        #test = Image.open(self.images_path[idx])
        
        imageTransformed = self.transform(image)
        labelTransformed = self.transform(label)
        #labelTransformed = self.transform(label)
            
            
        return {'image' : imageTransformed, 'label':labelTransformed}


