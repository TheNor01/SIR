
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
    
    def __init__(self,path,split,valid_classes,voidLabels,ignore_index,classMap,size):


    
        self.valid_classes = valid_classes
        self.voidLabels = voidLabels
        self.ignore_index = ignore_index
        self.size = size
        self.split = split
        self.classMap = classMap


        self.files = {}


        self.images_base = os.path.join(path, "leftImg8bit", split ) #my X
        self.annotations_base = os.path.join(path, "gtFine", split) #my Y


        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")


        """
        transformPreprocessTrain = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size,size)),
            transforms.ToTensor()
        ])

        self.transform = transformPreprocessTrain
        """

    def __len__(self):
        return len(self.files[self.split])
    
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
        image = cv2.resize(image, (128, 128))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
        image = image / 255.0

        img = image[...,::-1] #plt use RGB
        #plt.imshow(img)
        #plt.show()

        mask = cv2.imread(lbl_path, cv2.IMREAD_COLOR)

        img = mask[...,::-1] #plt use RGB
        #plt.imshow(img)
        #plt.show()


        mask = cv2.resize(mask, (128, 128))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB).astype('float32')


        img = mask[...,::-1] #plt use RGB
        #plt.imshow(img)
        #plt.show()

        #transformed = self.tfms(image=image, mask=mask)
        #image = transformed['image']
        #mask = transformed['mask']

        def get_label_mask(mask, class_values, label_colors_list):


            #print(class_values)
            #print(len(class_values))

            #print(label_colors_list)
            #print(len(label_colors_list))
            """
            This function encodes the pixels belonging to the same class
            in the image into the same label

            :param mask: NumPy array, segmentation mask.
            :param class_values: List containing class values, e.g car=0, bus=1.
            :param label_colors_list: List containing RGB color value for each class.
            """
            label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
            for value in class_values:
                for ii, label in enumerate(label_colors_list):
                    if value == label_colors_list.index(label):
                        label = np.array(label)
                        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = value
            label_mask = label_mask.astype(int)
            return label_mask
        
        label_colors_list= list(labels.fullLabelColor.values())
        class_values= list(labels.fullLabelColor.keys())

        # Get colored label mask.


        mask = get_label_mask(mask, class_values, label_colors_list)

        #plt.imshow(mask, interpolation='nearest')
        #plt.show()
       
        image = np.transpose(image, (2, 0, 1))
        
        image = torch.tensor(image, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.long) 

    
        #return image, mask



        return {'image' : image, 'label':mask}
    
   

class CityscapeDataset(Dataset):
    
    def __init__(self, image_dir, label_model):
        self.image_dir = image_dir
        self.image_fns = os.listdir(image_dir)
        self.label_model = label_model
        
    def __len__(self):
        return len(self.image_fns)
    
    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp).convert('RGB')
        image = np.array(image)
        cityscape, label = self.split_image(image)
        label_class = self.label_model.predict(label.reshape(-1, 3)).reshape(256, 256)
        cityscape = self.transform(cityscape)
        label_class = torch.Tensor(label_class).long()
        return cityscape, label_class
    
    def split_image(self, image):
        image = np.array(image)
        cityscape, label = image[:, :256, :], image[:, 256:, :]
        return cityscape, label
    
    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.Resize((256,256),antialias=True),
        ])
        return transform_ops(image)