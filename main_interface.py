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
from PIL import ImageChops,Image
from config import labels
from bin.models.UNET import UNet
from bin.models.RESNET import R2U_Net


screen = tk.Tk()

globalPath = ""

listbox = tk.Listbox(screen, selectmode=tk.SINGLE)
classifierNames = ['unet', 'deeplabv3','ppsnet','resnet']

for classifier in classifierNames:
    listbox.insert(0, classifier)

#alert if audio is not a .wav
labels.init()
dictIDxColors = labels.id2labelValid

ignore_index = 250

#dictIDxColors[ignore_index] = (0,0,0)
validLabels = list(dictIDxColors.keys())

size=64

if __name__ == '__main__':


    #load model
    #modelUnet = UNet(3,classes=len(validLabels)).to("cpu")
    #modelUnet.load_state_dict(torch.load("./checkpoint_model/unet.pth"))
    #modelUnet.eval()
    
    modelRes = R2U_Net(img_ch=3,output_ch=len(validLabels)).to("cpu")
    modelRes.load_state_dict(torch.load("./checkpoint_model/resnet.pth"))
    modelRes.eval()

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
            #mageToLoad = np.array(imageToLoad, dtype=np.uint8)
            imageToLoad = imageToLoad.resize([64,64])

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
        else:
            model = None

        if(model is None):
            tk.messagebox.showerror(title="ERROR", message="CHOOSE ONE MODEL")
            return

        predicted_label.delete(0,'end')
        if not path_entry.get():
            tk.messagebox.showerror(title="ERROR", message="PATH EMPTY")
            return

        print("classify: "+ path_entry.get())
        imageToLoad = Image.open("./resources/interface/"+"image_to_analyze.png")
        new_image = imageToLoad.resize((64, 64))

        tf=transforms.Compose([
                transforms.ToTensor(),
            ])
        
        imageTransformed = tf(new_image).unsqueeze(0)

        print(imageTransformed.shape)
        #model.eval()
        with torch.no_grad():
            output = model(imageTransformed.to("cpu"))
            print("predicted OUTPUT")
            print(output.shape)
            plt.imshow(output.permute(1, 2, 0))



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

    #true_entry = tk.Label(screen, text = "true entry", width = "20")
    #true_entry.grid(column = 0, row = 5)

    #true_label = tk.Entry(screen,textvariable = "", width = "20",justify='center')
    #true_label.grid(column=0,row=7)


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



    