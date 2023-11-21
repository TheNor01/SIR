from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from os.path import join
from torch import nn
import torch
import numpy as np
from statistics import mean
from sklearn.metrics import jaccard_score as jsc
from sklearn.metrics import accuracy_score as acs
import sklearn.metrics as skm
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR
from bin.utils import decode_segmap
import time
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
        


class RunningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_metrics(self):
        # confusion matrix
        hist = self.confusion_matrix
       
        TP = np.diag(hist)
        TN = hist.sum() - hist.sum(axis = 1) - hist.sum(axis = 0) + np.diag(hist)
        FP = hist.sum(axis = 1) - TP
        FN = hist.sum(axis = 0) - TP
        
        # 1e-6, aggiunto per evitare diviso 0--> non 1, altrimenti 
        
        
        # Senstivity/Recall: TP / TP + FN
        sensti_cls = (TP) / (TP + FN + 1e-6)
        sensti = np.nanmean(sensti_cls)
        
        # Precision: TP / (TP + FP)
        prec_cls = (TP) / (TP + FP + 1e-6)
        prec = np.nanmean(prec_cls)
        
        # F1 = 2 * Precision * Recall / Precision + Recall
        f1 = (2 * prec * sensti) / (prec + sensti + 1e-6)

        #Dice formula 
        dsc_cls = (2* TP) / ( (2* TP) + FP + FN + 1e-6)
        dsc = np.nanmean(dsc_cls)
        
        return [f1,dsc]

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))



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
    
    start = time.time()
    
    
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


    #DECAY SCHEDULER
    scheduler_lr = StepLR(optimizer, step_size=5, gamma=0.1)
    for e in range(epochs):
        print(f"Epoch {e + 1}\n-------------------------------")
        #iteriamo tra due modalità: train e test

        #nsamples = 100
        for mode in ['train','test']:
            print(mode)
            loss_meter.reset(); acc_meter.reset()
            model.train() if mode == 'train' else model.eval()
            running_loss = 0.0
            with torch.set_grad_enabled(mode=='train'): #abilitiamo i gradienti solo in training
                for i,batch in tqdm(enumerate(loader[mode],0)):

                    x=batch['image'].to(device)
                    y=batch['label'].to(device)
                    output = model(x)

                    if(modelString=="deeplab"):
                        output = output['out']
                    #aggiorniamo il global_step
                    #conterrà il numero di campioni visti durante il training
                    n = x.shape[0] #numero di elementi nel batch
                    global_step += n
                    l = criterion(output,y)
                    #l = cross_entropy2d(output,y)
                    if mode=='train':
                        l.backward()
                    optimizer.step()
                    #scheduler_lr.step()
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

                scheduler_lr.step()

            #una volta finita l'epoca (sia nel caso di training che test, loggiamo le stime finali)
            print(loss_meter.value())
            writer.add_scalar('loss/' + mode, loss_meter.value(), global_step=global_step)
            #writer.add_scalar('accuracy/' + mode, acc_meter.value(), global_step=global_step)
        
    end = time.time()
    print("ELAPSED TIME: s:"+str(end - start))
    return model


#https://github.com/sacmehta/ESPNet/blob/master/train/IOUEval.py to add alternative to jacc?

#Dice Coefficient https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou

def test_classifier(model,modelString, loader,validLabels,label_colous,epochs):

    acc_sh = []
    js_sh = []
    f1_score = []
    dsc_score = []

    device = "cpu"
    model.to(device)
    #eval mode?
    model.eval()

    f1_score_s = 0
    dsc_score_s = 0
    js_s = 0
    acc_s = 0

    running_metrics_val = RunningScore((validLabels))

    print("EVALUATING...")
    with torch.no_grad():
        for e in range(epochs):
            print(f"Epoch {e + 1}\n-------------------------------")
            local_acc_sh = []
            local_js_sh = []
            local_f1_score = []
            local_dsc_score = []
            for i,batch in tqdm(enumerate(loader)):

                print("BATCH # "+str(i))
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

                """
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
                """
                running_metrics_val.update(gt, pred) #update histogram in order to computer F1 score
                sh_metrics = metrics(gt.flatten(), pred.flatten())

                #add more metrics -- ex miou, pixel..

                local_acc_sh.append(sh_metrics[0])
                local_js_sh.append(sh_metrics[1])
                cm_metrics = running_metrics_val.get_metrics()
                local_f1_score.append(cm_metrics[0])
                local_dsc_score.append(cm_metrics[1])
                
                print(sh_metrics[0])
        
            f1_score_s = sum(local_f1_score)/len(local_f1_score)
            dsc_score_s = sum(local_dsc_score)/len(local_dsc_score)
            js_s = sum(local_js_sh)/len(local_js_sh)
            acc_s = sum(local_acc_sh)/len(local_acc_sh)

            running_metrics_val.reset() #RESET CM

            acc_sh.append(acc_s)
            js_sh.append(js_s)
            f1_score.append(f1_score_s)
            dsc_score.append(dsc_score_s)

            #end EPOCH


    print("AVERAGE METRICS")
    print("ACC was: ", round(mean(acc_sh), 2)) 
    print("JS: was", round(mean(js_sh), 2))
    print("F1: was", round(mean(f1_score), 2)) 
    print("DICE: was", round(mean(dsc_score), 2))

    return acc_sh, js_sh, f1_score,dsc_score


def metrics(true_label,predicted_label):
    #Accuracy Score
    acc = acs(true_label,predicted_label,normalize=True)

    
    #Jaccard Score/IoU https://scikit-learn.org/stable/modules/model_evaluation.html#jaccard-similarity-score
    js = skm.jaccard_score(true_label, predicted_label, average='micro')
    
    result_gm_sh = [acc, js]
    return(result_gm_sh)