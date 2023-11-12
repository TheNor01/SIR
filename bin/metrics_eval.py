from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from os.path import join
from torch import nn
import torch
import numpy as np

from torchmetrics.classification import MulticlassJaccardIndex
from sklearn.metrics import jaccard_score as jsc
from sklearn.metrics import accuracy_score as acs
import sklearn.metrics as skm
import torch.nn.functional as F

from matplotlib import pyplot as plt

from bin.utils import decode_segmap

from tqdm import tqdm

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


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss


def train_classifier(model, modelString,train_loader, test_loader, exp_name='experiment',lr=0.001, epochs=100, momentum=0.9):
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

        nsamples = 100
        for mode in ['train','test']:
            loss_meter.reset(); acc_meter.reset()
            model.train() if mode == 'train' else model.eval()
            running_loss = 0.0
            with torch.set_grad_enabled(mode=='train'): #abilitiamo i gradienti solo in training
                for i,batch in tqdm(enumerate(loader[mode],0)):

                    x=batch['image'].to(device)
                    y=batch['label'].to(device)

                    if i > nsamples: #to discard
                        break
                    

                    #pred = batch['image'].data.max(1)[1].cpu().numpy()
                    #gt = batch['label'].data.cpu().numpy()

                    output = model(x)

                    if(modelString=="deeplab"):
                        output = output['out']


                    #aggiorniamo il global_step
                    #conterrà il numero di campioni visti durante il training
                    n = x.shape[0] #numero di elementi nel batch
                    global_step += n

                    print(x.shape)
                    #print(y.shape)
                    print(output.shape)

                    #mask = torch.argmax(y, dim=1)

                    l = criterion(output,y)
                    #l = cross_entropy2d(output,y)
                    if mode=='train':
                        l.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    #acc = accuracy_score(y.data.to('cpu'),output.data.to('cpu').max(1)[1])
                    loss_meter.add(l.item(),n)

                    running_loss += l.item()
                    if i % 100 == 0:
                            print('[%d, %5d] loss: %.3f' %
                            (e + 1, i + 1, running_loss / 2000))
                            running_loss = 0.0

                    #acc_meter.add(acc,n)
                    #loggiamo i risultati iterazione per iterazione solo durante il training

                    if mode=='train':
                        writer.add_scalar('loss/train', loss_meter.value(), global_step=global_step)
                        #writer.add_scalar('accuracy/train', acc_meter.value(), global_step=global_step)



            #una volta finita l'epoca (sia nel caso di training che test, loggiamo le stime finali)
            writer.add_scalar('loss/' + mode, loss_meter.value(), global_step=global_step)
            #writer.add_scalar('accuracy/' + mode, acc_meter.value(), global_step=global_step)
        
    return model


#https://github.com/sacmehta/ESPNet/blob/master/train/IOUEval.py to add

def test_classifier(model,modelString, loader,validLabels,label_colous):

    acc_sh = []
    js_sh = []

    device = "cpu"
    model.to(device)
    #eval mode?
    model.eval()
    nsamples = 100

    print("EVALUATING...")
    for i,batch in tqdm(enumerate(loader)):

        if i > nsamples:
            break
        #[batch_size, channels, height, width].
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        #print(y) #dovrebbe essere int --> da fare in transformation
        #resolved

        output = model(x)
        if(modelString=="deeplab"):
            output = output['out']

        #pred = output.data.cpu()
        #gt = y.data.cpu()
        

        #https://stackoverflow.com/questions/54083220/why-does-this-semantic-segmentation-network-have-no-softmax-classification-layer
        #no softmax?

        pred = output.data.max(1)[1].cpu().numpy()
        gt = y.data.cpu().numpy()

        if i % 100 == 3:
            
            # Model Prediction
            decoded_pred = decode_segmap(pred[0],validLabels,label_colous) #doesn't work as expected here.
            plt.imshow(decoded_pred)
            plt.show()
            plt.clf()

            
            # Ground Truth
            decode_gt = decode_segmap(gt[0],validLabels,label_colous)
            plt.imshow(decode_gt)
            plt.show()

            exit()


    
        sh_metrics = metrics(gt.flatten(), pred.flatten())

        #add more metrics -- ex miou, pixel..

        acc_sh.append(sh_metrics[0])
        js_sh.append(sh_metrics[1])


    #acc_s = sum(acc_sh)/len(acc_sh)
    acc_s = sum(acc_sh)/(nsamples)
    #js_s = sum(js_sh)/len(js_sh)
    js_s = sum(js_sh)/(nsamples)

    print("Different Metrics were: ", acc_s) 
    print("Different Metrics were: ", js_s) 

    return acc_s, js_s


def metrics(true_label,predicted_label):
    #Accuracy Score
    acc = acs(true_label,predicted_label,normalize=True)

    
    #Jaccard Score/IoU https://scikit-learn.org/stable/modules/model_evaluation.html#jaccard-similarity-score
    js = skm.jaccard_score(true_label, predicted_label, average='micro')
    
    result_gm_sh = [acc, js]
    return(result_gm_sh)