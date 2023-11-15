
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import scipy.misc as m
import torch
from PIL import Image
import os

import imageio
import cv2

from config import labels

labels.init()

def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]



class ImagesDataset(Dataset):
    
    #def __init__(self,path,split,valid_classes,voidLabels,ignore_index,classMap,size):
    def __init__(self,path,split,size,fullLabelColor):

        #self.valid_classes = valid_classes
        #self.voidLabels = voidLabels
        #self.ignore_index = ignore_index
        self.size = size
        self.split = split
        self.fullLabelColor = fullLabelColor
        #self.classMap = classMap


        self.files = {}


        self.images_base = os.path.join(path, "leftImg8bit", split ) #my X
        self.annotations_base = os.path.join(path, "gtFine", split) #my Y


        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")



    def __len__(self):
        return len(self.files[self.split])
    
     # there are different class 0...33
    # we are converting that info to 0....18; and 250 for void classes
    # final mask has values 0...18 and 250
    
    """
    def encode_segmap(self, mask):
        # Put all void classes to ignore_index
        for _voidc in self.voidLabels:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.classMap[_validc]
        return mask
    """
    
    def get_label_mask(self,mask, class_values, label_colors_list):
            label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8) #3d zero

            for value in class_values:
                for i, label in enumerate(label_colors_list):
                    if value == label_colors_list.index(label):
                        label = np.array(label)
                        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = value
            label_mask = label_mask.astype(int)
            return label_mask
    
    def decode_segmap(self,temp,n_classes,dictLabelRange):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, n_classes):
            r[temp == l] = dictLabelRange[l][0]
            g[temp == l] = dictLabelRange[l][1]
            b[temp == l] = dictLabelRange[l][2]


        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb
        


    def __getitem__(self, idx):
        
        img_path = self.files[self.split][idx].rstrip()

        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_color.png", #we link image with gtFine labels id
        )


        """
        # read image
        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(lbl_path)
        encodedLabel = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        #img.show()

        transformEdit=transforms.Compose(
            [
            transforms.ToPILImage(),
            #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.ToTensor(),
            transforms.Resize((self.size,self.size),antialias=True),
            ]
        )

        #img = np.array(img, dtype=np.uint8)
        img = np.array(img)
        lbl = np.array(lbl)

        transfImg = transformEdit(img)
        #transfLabel = transformEdit(lbl)

        #img = np.array(img, dtype=np.uint8)
        #TRANSFORM
        #img = np.array(Image.fromarray(img,'RGB').resize([self.size,self.size])) # uint8 with RGB mode
        
        lbl = encodedLabel.astype(float)
        lbl = np.array(Image.fromarray(lbl,mode='F').resize([self.size,self.size]))
        lbl = lbl.astype(int)

        #img = img[:, :, ::-1]  # RGB -> BGR for cv2
        # change data type to float64
        #img = img.astype(np.float64)

        #print(img)

        #NCHW
        #img = img.transpose(2, 0, 1)
        # subtract mean

        #img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        """


        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self.size, self.size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
        image = image / 255.0

        #img = image[...,::-1] #plt use RGB
        #plt.imshow(img)
        #plt.show()

        mask = cv2.imread(lbl_path, cv2.IMREAD_COLOR)


        mask = cv2.resize(mask, (self.size, self.size))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB).astype('float32')

        
        # Get colored label mask.
        class_values= list(self.fullLabelColor.keys())
        label_colors_list= list(self.fullLabelColor.values())
        
        idXRgb =  dict(zip(range(len(label_colors_list)), label_colors_list))
        #print(idXRgb)
        #remove parameters --> to self

        mask = self.get_label_mask(mask, class_values, label_colors_list)

        #test
        maskDecoded = self.decode_segmap(mask, len(class_values), idXRgb)

        #print(mask)
        #print(np.unique(mask))

        #print(maskDecoded)
        #print(np.unique(maskDecoded))

        image = np.transpose(image, (2, 0, 1))
        
        image = torch.tensor(image, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.long) 

        """
        print(mask/255)
        plt.imshow(mask/255)
        plt.show()
        plt.clf()

        plt.imshow(maskDecoded)
        plt.show()
        plt.clf()
        """

        return {'image' : image, 'label':mask}
    
