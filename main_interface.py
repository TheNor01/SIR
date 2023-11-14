import csv
from os import path
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
import pandas as pd
import numpy as np
from torch import nn
import pickle
#Preprocessing
import os
import librosa
import librosa.display
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import ImageTk, Image
from config import labels
from bin.models.UNET import UNet
from bin.models.RESNET import R2U_Net

import cv2

from bin.utils import draw_segmentation_map,decode_segmap
from bin.models.DEEPLAB import GetDeeplabModel


screen = tk.Tk()
globalPath = ""

listbox = tk.Listbox(screen, selectmode=tk.SINGLE)
classifierNames = ['unet', 'deeplabv3','ppsnet','resnet']

for classifier in classifierNames:
    listbox.insert(0, classifier)

labels.init()
fullLabelColor = labels.fullLabelColor
validLabels = list(fullLabelColor.keys())

RGB_values = list(fullLabelColor.values())
idXRgb =  dict(zip(range(len(RGB_values)), RGB_values))
print(idXRgb)

size=256

if __name__ == '__main__':

    #load model
    modelUnet = UNet(3,classes=len(validLabels)).to("cpu")
    modelUnet.load_state_dict(torch.load("./checkpoint_model/unet.pth"))
    
    modelRes = R2U_Net(img_ch=3,output_ch=len(validLabels)).to("cpu")
    modelRes.load_state_dict(torch.load("./checkpoint_model/resnet_online.pth"))


    modelDeep = GetDeeplabModel(len(validLabels))
    modelDeep.load_state_dict(torch.load("./checkpoint_model/deeplab.pth"))

    model = None

    def select_file():

        file_path = filedialog.askopenfilename(defaultextension=".png",initialdir="./dataset/leftImg8bit")

        if(not file_path.endswith(".png")):
            tk.messagebox.showerror(title="ERROR", message= "Not a valid file")
            return

        print("Immagine selezionata:", file_path)
        
        if file_path:

            globalPath = file_path

            print("Gp:"+globalPath)

            path_entry.insert(0,globalPath)


            imageToLoad = Image.open(globalPath)
            imageToLoad = imageToLoad.resize([size,size])

            imageToLoad.save("./resources/interface/"+"image_to_analyze.png")

            img = ImageTk.PhotoImage(imageToLoad)
            imagebox.config(image=img)
            imagebox.image = img
            #globalImage = im_trimmed
            
    def classify():

        choosenModel=None

        print("model load")
        valueModel = [listbox.get(idx) for idx in listbox.curselection()]
        if(len(valueModel)==0):
             model = None
        else:
            choosenModel = valueModel[0]


        if(choosenModel=="unet"):
            model = modelUnet
        elif(choosenModel=="resnet"):
            model = modelRes
        elif(choosenModel=="deeplabv3"):
            model = modelDeep
        else:
            model = None

        if(model is None):
            tk.messagebox.showerror(title="ERROR", message="CHOOSE ONE MODEL")
            return

        if not path_entry.get():
            tk.messagebox.showerror(title="ERROR", message="PATH EMPTY")
            return

        print("classify: "+ path_entry.get())
        imageToLoad = Image.open("./resources/interface/"+"image_to_analyze.png")
        new_image = imageToLoad.resize((64, 64))

        tf=transforms.Compose([
                transforms.ToTensor(),
            ])
        
        imageTransformed = tf(new_image)
        print(imageTransformed.shape)
        model.eval()
        with torch.no_grad():
            output = model(imageTransformed.unsqueeze(0))
            
            if(choosenModel=="deeplabv3"):
                output = output['out']
            
            output= output.to("cpu")
            print("predicted OUTPUT")
            print(output.shape)

            mask = draw_segmentation_map(output,RGB_values)

            cv2.imshow('Segmented image', mask)
            cv2.waitKey(0)
            
            out = output.data.max(1)[1].cpu().numpy()
 
            print(np.unique(out[0]))
            print(out.shape)
            mask2 = decode_segmap(out[0],len(idXRgb),idXRgb)

            plt.imshow(mask2)
            plt.show()
            plt.clf()


            cv2.imwrite("./resources/interface/my_out.png",mask)


    def clear():
        #true_label.delete(0,'end')
        path_entry.delete(0,'end')
        #predicted_label.delete(0,'end')
        imagebox.config(image='')

    screen.minsize(400,400)
    screen.title("Classify genre")

    select_button = tk.Button(screen, text="Load your .png img", command=select_file)
    select_button.grid(row=1)

    path_entry = tk.Entry(screen,textvariable = globalPath, width = "100")
    path_entry.grid(column=0,row=0)


    predicted_entry = tk.Label(screen, text = "label classified", width = "40")
    predicted_entry.grid(column = 0, row = 10)

    predicted_label = tk.Entry(screen,textvariable = "", width = "40",justify='center')
    predicted_label.grid(column=0,row=20)

    filename_audio= tk.StringVar()

    # label to show the image
    imagebox = tk.Label(screen)
    imagebox.grid(column=0,row=350)
    

    classify_button = tk.Button(screen, text="Classify", command=classify)
    classify_button.grid(column=0,row=400)

    clear_button = tk.Button(screen, text="Clear", command=clear)
    clear_button.grid(column=5)

    listbox.grid(row=400,column=5)



    screen.mainloop()



    