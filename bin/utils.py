
import numpy as np
from PIL import Image
from tqdm import tqdm 
import os
from matplotlib import pyplot as plt



def decode_segmap(temp,n_classes,dictLabelRange):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, n_classes):
            r[temp == l] = dictLabelRange[l][0]
            g[temp == l] = dictLabelRange[l][1]
            b[temp == l] = dictLabelRange[l][2]

            #print(r[temp == l])
            #print(g[temp == l])
            #print(b[temp == l])

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))

        #print(rgb.shape)
        #print(r.shape)

        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb


