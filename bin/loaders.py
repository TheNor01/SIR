
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt


class ImagesDataset(Dataset):
    
    def __init__(self, images_path,valid_classes,voidLabels,ignore_index,rgbColors,transform=None):
    
        self.images_path = images_path
        self.transform = transform
        self.valid_classes = valid_classes
        self.voidLabels = voidLabels
        self.ignore_index = ignore_index
        self.classMap = dict(zip(valid_classes, range((rgbColors))))

    def __len__(self):
        return len(self.images_path)
    

    # there are different class 0...33
    # we are converting that info to 0....18; and 250 for void classes
    # final mask has values 0...18 and 250
    def encode_segmap(self, mask):
        # Put all void classes to ignore_index
        for _voidc in self.voidLabels:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.classMap[_validc]
        return mask

    def __getitem__(self, idx):
        
        img = plt.imread(self.images_path[idx])
        image,label = img[:,:int(img.shape[1]/2)],img[:,int(img.shape[1]/2):]

        #im = Image.fromarray(image.astype('uint8'), 'RGB')
        #lb = Image.fromarray(label.astype('uint8'), 'RGB')

        #test = Image.open(self.images_path[idx])
        

        encodedLabel = self.encode_segmap(np.array(label, dtype=np.uint8))

        #non va bene, deve essere una maskera encodata
        
        imageTransformed = self.transform(image)
        labelTransformed = self.transform(encodedLabel)
            
            
        return {'image' : imageTransformed, 'label':labelTransformed}


