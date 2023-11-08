from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from os.path import join
from torch import nn
import torch
import numpy as np
from bin.utils import show

import sklearn.metrics as skm

class AverageValueMeter():
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.sum = 0
        self.num = 0
 
    def add(self, value, num):
        self.sum += value*num
        self.num += num
 
    def value(self):
        try:
            return self.sum/self.num
        except:
            return None


def train_classifier(model, train_loader, test_loader, exp_name='experiment',lr=0.001, epochs=100, momentum=0.9):
    criterion = nn.CrossEntropyLoss() 
    optimizer = SGD(model.parameters(), lr, momentum=momentum) 
    #meters
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()
    #writer
    writer = SummaryWriter("./logs/metrics/"+str(exp_name)+"_lr."+str(lr)+"_e."+str(epochs))
    #device
    device = "cpu"
    model.to(device)
    #definiamo un dizionario contenente i loader di training e test
    loader = {
    'train' : train_loader,
    'test' : test_loader
    }
    #inizializziamo il global step
    global_step = 0
    for e in range(epochs):
    #iteriamo tra due modalità: train e test
        for mode in ['train','test']:
            loss_meter.reset(); acc_meter.reset()
            model.train() if mode == 'train' else model.eval()
            with torch.set_grad_enabled(mode=='train'): #abilitiamo i gradienti solo in training
                for img,label  in (loader[mode]):

                    x=img #"portiamoli sul device corretto"
                    y=label


                    print(type(x))
                    print(type(y))



                    output = model(x)

                    print(type(output))
                    print(output)
                    
                    #if(e%5==0):
                    # show(img,output,label)

                    #aggiorniamo il global_step
                    #conterrà il numero di campioni visti durante il training
                    n = x.shape[0] #numero di elementi nel batch
                    global_step += n

                    l = criterion(output,y)
                    if mode=='train':
                        l.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    acc = accuracy_score(y.data.to('cpu'),output.data.to('cpu').max(1)[1])
                    loss_meter.add(l.item(),n)
                    acc_meter.add(acc,n)
                    #loggiamo i risultati iterazione per iterazione solo durante il training

                    if mode=='train':
                        writer.add_scalar('loss/train', loss_meter.value(), global_step=global_step)
                        writer.add_scalar('accuracy/train', acc_meter.value(), global_step=global_step)



            #una volta finita l'epoca (sia nel caso di training che test, loggiamo le stime finali)
            writer.add_scalar('loss/' + mode, loss_meter.value(), global_step=global_step)
            writer.add_scalar('accuracy/' + mode, acc_meter.value(), global_step=global_step)
        

        #conserviamo i pesi del modello alla fine di un ciclo di training e test
        #torch.save(model.state_dict(),'./resources/archive/stored/models/'+str(exp_name)+"_"+str(e)+".pth")
    return model




def test_classifier(model, loader):

    acc_sh = []
    js_sh = []


    device = "cpu"
    model.to(device)

    model.eval()

    predictions, labels = [], []
    for img,label in loader:


        x = img.to(device)
        y = label.to(device)


        output = model(x)

        print(x.shape)
        print(y.shape)


        pred = output.data.max(1)[1].cpu().numpy()
        gt = y.data.cpu().numpy()

        sh_metrics = metrics(gt.flatten(), pred.flatten())

        acc_sh.append(sh_metrics[0])
        js_sh.append(sh_metrics[1])


        predictions.extend(list(pred))
        labels.extend(list(gt))

    acc_s = sum(acc_sh)/len(acc_sh)
    js_s = sum(js_sh)/len(js_sh)



    return acc_s, js_s


def metrics(true_label,predicted_label):
    #Accuracy Score
    acc = skm.accuracy_score(true_label, predicted_label, normalize=True)
    
    #Jaccard Score/IoU
    js = skm.jaccard_score(true_label, predicted_label, average='micro')
    
    result_gm_sh = [acc, js]
    return(result_gm_sh)