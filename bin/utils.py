
import numpy as np
from PIL import Image
from tqdm import tqdm 
import os
from matplotlib import pyplot as plt
import torch

def draw_segmentation_map(outputs,RGB_values):
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()

    # create Numpy arrays containing zeros
    # later to be used to fill them with respective red, green, and blue pixels
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)
    
    for label_num in range(0, len(RGB_values)):
        index = labels == label_num
        red_map[index] = np.array(RGB_values)[label_num, 0]
        green_map[index] = np.array(RGB_values)[label_num, 1]
        blue_map[index] = np.array(RGB_values)[label_num, 2]
        
    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map

## alternative --??

def decode_segmap(temp,n_classes,dictLabelRange):
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


