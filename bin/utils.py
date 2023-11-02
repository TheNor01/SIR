
import numpy as np
from PIL import Image
from tqdm import tqdm 
import os
from matplotlib import pyplot as plt

def preprocess(path,maps):
    img = Image.open(path)
    img1 = img.crop((0, 0, 256, 256)).resize((128, 128))
    img2 = img.crop((256, 0, 512, 256)).resize((128, 128))


    img1 = np.array(img1) / 255.
    img2 = np.array(img2)


    mask = np.zeros(shape=(img2.shape[0], img2.shape[1]), dtype = np.uint32) #mask


    for row in range(img2.shape[0]):
        for col in range(img2.shape[1]):
            a = img2[row, col, :]
            final_key = None
            final_d = None
            for key, value in maps.items():         #According to id
                d = np.sum(np.sqrt(pow(a - value, 2)))
                if final_key == None:
                    final_d = d
                    final_key = key
                elif d < final_d:
                    final_d = d
                    final_key = key
            mask[row, col] = final_key
    mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
    del img2
    return img1, mask


def prepare_tensor_dataset(train_path, val_path,maps):

    X_train = []
    Y_train = []
    X_val = []
    Y_val = []

    for file in tqdm(os.listdir(train_path)):
        img, mask = preprocess(f"{train_path}/{file}",maps)
        X_train.append(img)
        Y_train.append(mask)
    
    for file in tqdm(os.listdir(val_path)):
        img, mask = preprocess(f"{val_path}/{file}",maps)
        X_val.append(img)
        Y_val.append(mask)

    return X_train, Y_train, X_val, Y_val

def show(img,output,label,denorm = False):
    img,output,label = img.cpu(),output.cpu(),label.cpu()
    fig,ax = plt.subplots(len(output),3,figsize=(10,10))

    ax = ax.flatten() #necessary

    #permute torch method in order to display it
    
    for i in range(len(output)):
        if(len(output) == 3):
            Img,Lab,act = img[i],output[i],label[i]
            Img,Lab,act = Img,Lab.detach().permute(1,2,0).numpy(),act
            ax[i][0].imshow(Img.permute(1,2,0))
            ax[i][1].imshow(Lab)
            ax[i][2].imshow(act.permute(1,2,0))
        else:
            Img,Lab,act = img[i],output[i],label[i]
            Img,Lab,act = Img,Lab.detach().permute(1,2,0).numpy(),act

            ax[0].imshow(Img.permute(1,2,0))
            ax[1].imshow(Lab)
            ax[2].imshow(act.permute(1,2,0))
    plt.show()