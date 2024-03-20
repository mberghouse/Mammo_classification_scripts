from __future__ import print_function, division
import numpy as np
import pandas as pd
import os


import torch
from skimage import io, transform
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torchvision import datasets
from torchvision.transforms import ToTensor
import cv2
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.optim import Adam, SGD
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torchvision import models
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryROC, BinaryAUROC
import PIL
from PIL import Image

# visualisation
import seaborn as sns

# helpers
from tqdm import tqdm
import time
import copy
import gc
from enum import Enum

import warnings
warnings.filterwarnings('ignore')


#torchvision.disable_beta_transforms_warning()
#import torchvision.transforms.v2 as transforms

#plt.ion()   # interactive mode

##########################################
##### HERE ARE THE AUGMENTATIONS!!! ######
##########################################
# affine_transfomer = T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))

augmentator = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=(.6,1.2)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

small_aug = transforms.Compose([
    # input for augmentator is always PIL image
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


class CBISDataset(Dataset):
    """CBIS-DDSM dataset."""

    def __init__(self, labels, filenames, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = labels
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fname = self.filenames[idx]
        image = cv2.imread(fname)
        label = self.labels[idx]    
        if self.transform:
            image = self.transform(image)

        return image, label
		

def find_optim_thres(fpr, tpr, thresholds):
    optim_thres = thresholds[0]
    inx = 0
    min_dist = 1.0
    for i in range(len(fpr)):
        dist = np.linalg.norm(np.array([0.0, 1.0]) - np.array([fpr[i], tpr[i]]))
        if dist < min_dist:
            min_dist = dist
            optim_thres = thresholds[i]
            inx = i
            
    return optim_thres, inx


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    metricf1 = BinaryF1Score()
    precision = BinaryPrecision()
    recall = BinaryRecall()
    accuracy = BinaryAccuracy()
    roc = BinaryROC()
    auc = BinaryAUROC()
    best_model_wts = model.state_dict()
    stop_count = 0
    best_f1 = -1.0
    train_metrics = {'loss' : [], 'acc' : [], 'f1': [], 'precision': [], 'recall': [], 'auc': []}
    val_metrics = {'loss' : [], 'acc' : [], 'f1': [], 'precision': [], 'recall': [], 'auc': []}
    # inital threshold for first epoch, it will change afterwards
    threshold = 0.5
    sched_steps=[]
    print('Starting training...')
    print('-' * 20)
    for epoch in range(num_epochs):
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # empty 'all' tensors for saving
            # for calculating aoc at the end of epoch, and for calculating new threshold
            all_outputs = torch.Tensor([])
            all_labels = torch.Tensor([])
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            n_samples = 0
            n_correct = 0
            running_f1 = 0.0
            # Iterate over data.
            print(f'{phase} for epoch {epoch + 1}')
            for inputs, labels in tqdm(dataloaders[phase]):
                labels = torch.unsqueeze(torch.tensor(labels), 1).to(dtype=torch.float)               
                inputs = inputs.float()
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = (outputs > threshold).double()
                    
                    # concatenating all outputs and labels for calculation aoc and new threshold
                    all_outputs = torch.cat((all_outputs, outputs.to('cpu')))
                    all_labels = torch.cat((all_labels, labels.to('cpu')))                  
                   
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                running_loss += loss.item()
                # collect any unused memmory
                gc.collect()
                torch.cuda.empty_cache()           
            # statistics
            epoch_loss = running_loss / len(dataloaders[phase])            
            # find true positive and false positive rates for ROC curve
            all_labels=all_labels.to(dtype=torch.long)
            fpr, tpr, thresholds = roc(all_outputs, all_labels)
            epoch_auc = auc(all_outputs, all_labels)
            # find new threshold
            threshold, _ = find_optim_thres(fpr, tpr, thresholds)
            print(f'New threshold is {threshold}')
            # calculate metrics using new optimized threshold
            epoch_f1 = metricf1(all_outputs > threshold, all_labels)
            epoch_acc = accuracy(all_outputs > threshold, all_labels)
            epoch_precision = precision(all_outputs > threshold, all_labels)
            epoch_recall = recall(all_outputs > threshold, all_labels)
            print(f'{phase} F1 is {epoch_f1}')            
            # save all of the statistics for latter analysis
            if phase == 'train':
                wandb.log({"train_acc": epoch_acc, "train_loss": epoch_loss, "train_f1": epoch_f1, "train_precision": epoch_precision
                         , "train_recall": epoch_recall, "train_auc": epoch_auc})
                train_metrics['loss'].append(epoch_loss)
                train_metrics['acc'].append(epoch_acc)
                train_metrics['f1'].append(epoch_f1)
                train_metrics['precision'].append(epoch_precision)
                train_metrics['recall'].append(epoch_recall)
                train_metrics['auc'].append(epoch_auc)
            else:
                wandb.log({"val_acc": epoch_acc, "val_loss": epoch_loss, "val_f1": epoch_f1, "val_precision": epoch_precision
                         , "val_recall": epoch_recall, "val_auc": epoch_auc})
                val_metrics['loss'].append(epoch_loss)
                val_metrics['acc'].append(epoch_acc)
                val_metrics['f1'].append(epoch_f1)
                val_metrics['precision'].append(epoch_precision)
                val_metrics['recall'].append(epoch_recall)
                val_metrics['auc'].append(epoch_auc)

            # deep copy the model
                if val_metrics['f1'][-1] > best_f1:
                    best_f1 = val_metrics['f1'][-1]
                    best_model_wts = model.state_dict()
                    checkpoint['threshold'] = threshold
                    torch.save(checkpoint, 'checkpoint.pth')
               
        # cant be formated in strin g
        tr_loss, tr_acc, tr_f1, tr_prec, tr_rec, tr_auc = train_metrics['loss'][-1], train_metrics['acc'][-1],  train_metrics['f1'][-1], train_metrics['precision'][-1], train_metrics['recall'][-1], train_metrics['auc'][-1]
        val_loss, val_acc, val_f1, val_prec, val_rec, val_auc = val_metrics['loss'][-1], val_metrics['acc'][-1], val_metrics['f1'][-1], val_metrics['precision'][-1], val_metrics['recall'][-1], val_metrics['auc'][-1]
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{num_epochs}, learning rate: {lr}')
        print(f'Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}, Train f1: {tr_f1:.4f}, Train Precision: {tr_prec:.4f}, Train Recall: {tr_rec:.4f}, Train AUC: {tr_auc:.4f}')
        print(f'Valitadion Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}, Vall f1: {val_f1:.4f}, Val Precision: {val_prec:.4f}, Val Recall: {val_rec:.4f}, Val AUC: {val_auc:.4f}')       
        if (tr_loss<.5)&(val_loss < .48):
            break
        elif tr_loss<.38:
            break
        # if epoch == 0:
            # val_f1_best=val_f1
        # else:
            # if val_f1 > val_f1_best:
                # val_f1_best=val_f1
                # stop_count = 0
            # else:
                # stop_count = stop_count + 1
        
        # if stop_count == 30:
            # break
                
#         if earlystoper.early_stop(val_f1):
#             break       
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val auc: {best_f1:4f}')
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
    
    
def test_model(model,criterion,dataloader, threshold=.5):
    device='cuda'
    since = time.time()
    metricf1 = BinaryF1Score()
    precision = BinaryPrecision()
    recall = BinaryRecall()
    accuracy = BinaryAccuracy()
    roc = BinaryROC()
    auc = BinaryAUROC()
    stop_count = 0
    best_f1 = -1.0
    test_metrics = {'loss' : [], 'acc' : [], 'f1': [], 'precision': [], 'recall': [], 'auc': []}
    # inital threshold for first epoch, it will change afterwards
#     threshold = 0.5
    print('Starting testing...')
    # empty 'all' tensors for saving
    all_outputs = torch.Tensor([])
    all_labels = torch.Tensor([])
    model.eval()   # Set model to evaluate mode
    running_loss = 0.0
    n_samples = 0
    n_correct = 0
    running_f1 = 0.0
    # Iterate over data.
    for inputs, labels in tqdm(dataloader):
        labels = torch.unsqueeze(torch.tensor(labels), 1).to(dtype=torch.float)               
        #labels=torch.tensor(labels)
        inputs = inputs.float()
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
        # zero the parameter gradients
            outputs = model(inputs)
            #preds = (outputs > threshold).double()
            # concatenating all outputs and labels for calculation aoc and new threshold
            all_outputs = torch.cat((all_outputs, outputs.to('cpu')))
            all_labels = torch.cat((all_labels, labels.to('cpu')))                  
            #print(labels)
            # _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
        running_loss += loss.item()
        #n_correct += (preds == labels).sum().item()
        # collect any unused memmory
        gc.collect()
        torch.cuda.empty_cache()  
        
    # statistics
    epoch_loss = running_loss / len(dataloader)            
    # find true positive and false positive rates for ROC curve
    #print ('outputs: ', all_outputs, 'labels', all_labels)
    all_labels=all_labels.to(dtype=torch.long)
    fpr, tpr, thresholds = roc(all_outputs, all_labels)
    epoch_auc = auc(all_outputs, all_labels)
    # find new threshold
    threshold, _ = find_optim_thres(fpr, tpr, thresholds)
    print(f'New threshold is {threshold}')
    # calculate metrics using new optimized threshold
    epoch_f1 = metricf1(all_outputs > threshold, all_labels)
    epoch_acc = accuracy(all_outputs > threshold, all_labels)
    epoch_precision = precision(all_outputs > threshold, all_labels)
    epoch_recall = recall(all_outputs > threshold, all_labels)
    # save all of the statistics for latter analysis
    test_metrics['loss'].append(epoch_loss)
    test_metrics['acc'].append(epoch_acc)
    test_metrics['f1'].append(epoch_f1)
    test_metrics['precision'].append(epoch_precision)
    test_metrics['recall'].append(epoch_recall)
    test_metrics['auc'].append(epoch_auc)               
    time_elapsed = time.time() - since
    wandb.log({"test_acc": epoch_acc, "test_loss": epoch_loss, "test_f1": epoch_f1, "test_precision": epoch_precision
                         , "test_recall": epoch_recall, "test_auc": epoch_auc})
    print(f'Inference complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'F1 Score = : {epoch_f1:4f}')
    print(f'AUC Score = : {epoch_auc:4f}')
    print(f'Acc Score = : {epoch_acc:4f}')
    return test_metrics
    
    
import wandb
import timm
import numpy as np
model_scores=[]

for i in range(383):
    if i<96:
        dataset = "300x500-v6-mass"
        metadata_list=['mass_case_description_test_set','mass_case_description_train_set']
    elif i<192:
        dataset = "300x500-v6-calc"
        metadata_list=['calc_case_description_test_set','calc_case_description_train_set']
    elif i<288:
        dataset = "600x1000-v6-mass"
        metadata_list=['mass_case_description_test_set','mass_case_description_train_set']
    else:
        dataset = "600x1000-v6-calc"
        metadata_list=['calc_case_description_test_set','calc_case_description_train_set']


    df_list=[]
    for j in range(len(metadata_list)):
        print (metadata_list[j])
        df=pd.read_csv('/home/mbadhan/Desktop/mberghouse/'+metadata_list[j]+'.csv')
        print (len(df))
        
        fname=[]

        df=df.rename(columns={"file path": "filename","pathology":"class", 'image file path':'filename','cropped image file path':'patch_filename'})
    ###Remove multiple-counted images for whole image
        for f in range(len(df)):
            fname.append(df['filename'].loc[f])
            if f>0:
                if fname[f] == fname[f-1]:
                    df.drop(f, inplace=True)
        print ('df length after removal of repeats: ', len(df))
        
        
        nan_count=0

        for k in range(len(df)):
    ### For whole images
            df.filename.iloc[k]='/home/mbadhan/Desktop/mberghouse/CBIS-DDSM-preprocessed/'+dataset[:-8]+'_v6/'+df.filename.iloc[k].rsplit('/',3)[0]+'/1-1.png'
            if 'MALIGNANT' in df['class'].iloc[k]:
                df['class'].iloc[k]=1
            else:
                df['class'].iloc[k]=0
        df_list.append(df)
        

    df_test=df_list[0]
    df_train=df_list[1]

    filenames_train=[]
    labels_train=[]
    for m in range(len(df_train)):
        filenames_train.append(df_train.filename.iloc[m])
        labels_train.append(df_train['class'].iloc[m])

    filenames_test=[]
    labels_test=[]
    for m in range(len(df_test)):
        filenames_test.append(df_test.filename.iloc[m])
        labels_test.append(df_test['class'].iloc[m])
        
    import torchvision
    import numpy.matlib as np_mlb
            
    dataset = CBISDataset(labels_train,filenames_train,transform=augmentator)

    val_pct = 0.05
    val_size = int(val_pct * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    test_dataset = CBISDataset(labels_test,filenames_test, transform=small_aug)
    val_size =  len(val_dataset)
    train_size = len(train_dataset)
    if i<192:
    	batch_size = 16
    else:
    	batch_size = 8
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, pin_memory = True,num_workers=16 )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle = True, pin_memory = True,num_workers=16 )
    


        
    epochs = 8#np.random.randint(18,36)
    #epochs=20
    hidden_size = 64
    dropout = np.random.uniform(0.01,0.4)   
    
      #################   
     ### Regnetx ###
     #################  
 ###########################              
 ######### Mass ############
 ###########################  
 
    if i<8:
        model_name = 'SEregnetx_064'
        #baseline_name
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,6e-5)
        dataset = "300x500-v6-mass"       
              
    elif i<16:
        model_name = 'CBAMregnetx_064'
        #epochs = np.random.randint(8,20)
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,6e-5)
        dataset = "300x500-v6-mass"

    elif i<24:
        model_name = 'ECAregnetx_064'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,6e-5)
        dataset = "300x500-v6-mass"
        

        
    elif i<32:
        model_name = 'regnetx_064'
        epochs = np.random.randint(8,20)
        #baseline_name
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,6e-5)
        dataset = "300x500-v6-mass"
       
        
     #################   
     ### Densenets ###
     #################  
 ###########################              
 ######### Mass ############
 ###########################       
        
      
       
    elif i<40:
        model_name = 'SEdensenet169'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,7e-5)
        dataset = "300x500-v6-mass"
        
        
    elif i<48:
        model_name = 'CBAMdensenet169'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,7e-5)
        dataset = "300x500-v6-mass"

        
    elif i<56:
        model_name = 'ECAdensenet169'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,7e-5)
        dataset = "300x500-v6-mass"  
        
        
    elif i<64:
        model_name = 'densenet169'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,7e-5)
        dataset = "300x500-v6-mass"  

     #################   
     ### Resnets ###
     #################  
 ###########################              
 ######### Mass ############
 ###########################       
              
    elif i<72:
        model_name = 'SEresnet50'
        #baseline_name
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,6e-5)
        dataset = "300x500-v6-mass"
       
        
    elif i<80:
        model_name = 'CBAMresnet50'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,6e-5)
        dataset = "300x500-v6-mass"

        
    elif i<88:
        model_name = 'ECAresnet50'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,6e-5)
        dataset = "300x500-v6-mass"
    
    elif i<96:
        model_name = 'tv_resnet50'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,6e-5)
        dataset = "300x500-v6-mass"
       
           


     #################   
     ### Regnetx ###
     #################  
 ###########################              
 ######### Calc ############
 ###########################   
       
    elif i<104:
        model_name = 'SEregnetx_064'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "300x500-v6-calc"
       
    elif i<112:
        #epochs = np.random.randint(8,20)
        model_name = 'CBAMregnetx_064'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "300x500-v6-calc"
         
        
    elif i<120:
        model_name = 'ECAregnetx_064'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "300x500-v6-calc"
           
    
    elif i<128:
        model_name = 'regnetx_064'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "300x500-v6-calc"
       
#################   
     ### Densenets ###
     #################  
 ###########################              
 ######### Calc ############
 ###########################             
        
    elif i<136:
        model_name = 'SEdensenet169'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "300x500-v6-calc"
        
        
    elif i<144:
        model_name = 'CBAMdensenet169'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "300x500-v6-calc"
        
    elif i<152:
        model_name = 'ECAdensenet169'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "300x500-v6-calc"
    
       
    elif i<160:
        model_name = 'densenet169'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "300x500-v6-calc"
        

 #################   
     ### Resnets ###
     #################         
 ###########################              
 ######### Calc ############
 ###########################       
                
    elif i<168:
        model_name = 'SEresnet50'
        #baseline_name
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "300x500-v6-calc"
       
        
    elif i<176:
        model_name = 'CBAMresnet50'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "300x500-v6-calc"
        
        
    elif i<184:
        model_name = 'ECAresnet50'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "300x500-v6-calc"
       

        
    elif i<192:
        model_name = 'tv_resnet50'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "300x500-v6-calc"
       

        

 ###########################              
 ######### 600x1000 ############
 ###########################       
        
    elif i<200:
        model_name = 'SEregnetx_064'
        #baseline_name
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,6e-5)
        dataset = "600x1000-v6-mass"       
    
    elif i<208:
        model_name = 'CBAMregnetx_064'
        #epochs = np.random.randint(8,20)
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,6e-5)
        dataset = "600x1000-v6-mass"


        
    elif i<216:
        model_name = 'ECAregnetx_064'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,6e-5)
        dataset = "600x1000-v6-mass"
       

        
    elif i<224:
        model_name = 'regnetx_064'
        #epochs = np.random.randint(8,20)
        #baseline_name
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,6e-5)
        dataset = "600x1000-v6-mass"
       

        
     #################   
     ### Densenets ###
     #################  
 ###########################              
 ######### Mass ############
 ###########################       
        

       
    elif i<232:
        model_name = 'SEdensenet169'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,7e-5)
        dataset = "600x1000-v6-mass"
        

        
    elif i<240:
        model_name = 'CBAMdensenet169'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,7e-5)
        dataset = "600x1000-v6-mass"

        
    elif i<248:
        model_name = 'ECAdensenet169'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,7e-5)
        dataset = "600x1000-v6-mass"  

        
    elif i<256:
        model_name = 'SEdensenet169'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,7e-5)
        dataset = "600x1000-v6-mass"  
     #################   
     ### Resnets ###
     #################  
 ###########################              
 ######### Mass ############
 ###########################       
              
    elif i<264:
        model_name = 'SEresnet50'
        #baseline_name
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,6e-5)
        dataset = "600x1000-v6-mass"
       

        
    elif i<272:
        model_name = 'CBAMresnet50'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,6e-5)
        dataset = "600x1000-v6-mass"
        

        
    elif i<280:
        model_name = 'ECAresnet50'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,6e-5)
        dataset = "600x1000-v6-mass"
       
             
    elif i<288:
        model_name = 'tv_resnet50'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(3e-5,6e-5)
        dataset = "600x1000-v6-mass"
        



     #################   
     ### Regnetx ###
     #################  
 ###########################              
 ######### Calc ############
 ###########################   
       
    elif i<296:
        model_name = 'SEregnetx_064'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "600x1000-v6-calc"
       
 
        
    elif i<304:
        #epochs = np.random.randint(8,20)
        model_name = 'CBAMregnetx_064'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "600x1000-v6-calc"
       
 
        
    elif i<312:
        model_name = 'ECAregnetx_064'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "600x1000-v6-calc"
       
   
    
    elif i<320:
        model_name = 'regnetx_064'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "600x1000-v6-calc"
       

#################   
     ### Densenets ###
     #################  
 ###########################              
 ######### Calc ############
 ###########################             
        

       
    elif i<328:
        model_name = 'SEdensenet169'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "600x1000-v6-calc"
        
  
    elif i<336:
        model_name = 'CBAMdensenet169'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "600x1000-v6-calc"
        

       
    elif i<344:
        model_name = 'ECAdensenet169'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "600x1000-v6-calc"
        
    elif i<352:
        model_name = 'densenet169'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "600x1000-v6-calc"
        
        

 #################   
     ### Resnets ###
     #################         
 ###########################              
 ######### Calc ############
 ###########################       
                
    elif i<360:
        model_name = 'SEresnet50'
        #baseline_name
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "600x1000-v6-calc"
       

    elif i<368:
        model_name = 'CBAMresnet50'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "600x1000-v6-calc"
        

        
    elif i<376:
        model_name = 'ECAresnet50'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "600x1000-v6-calc"
        
    elif i<384:
        model_name = 'resnet50'
        model = timm.create_model(model_name, pretrained=True)
        lr = np.random.uniform(4e-5,8e-5)
        dataset = "600x1000-v6-calc"
       

    

     
    print (model_name)
        
    
    #Initiate Weights and Biases Tracking
    wandb.init(
        # set the wandb project where this run will be logged
        project="Model Testing 5_3",
        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "architecture": model_name,
        "dataset": dataset,
        "epochs": epochs,
        "batch size": batch_size,
        "hidden size": hidden_size, 
        "optimizer_name": 'AdamW',
        "dropout_rate": dropout,
        "weight decay": 1e-4,       
        }
    )
        
        
#     for i in range(3):
#         model.layer1[i].att2.channel.act = nn.LeakyReLU(.1)
#         model.layer2[i].att2.channel.act = nn.LeakyReLU(.1)
#         model.layer3[i].att2.channel.act = nn.LeakyReLU(.1)
#         model.layer4[i].att2.channel.act = nn.LeakyReLU(.1)
    dropout_rate = dropout
    num_in_features = model.get_classifier().in_features
    for name, param in model.named_parameters():
        ijk=0
        
    # Replace the existing classifier. It's named: classifier
    if "head.fc" in name:
        model.head.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features=num_in_features, out_features=hidden_size, bias=False),
        nn.LeakyReLU(.1,inplace=True),
        #nn.Dropout(dropout_rate),
        nn.Linear(in_features=hidden_size, out_features=1, bias=False),
        nn.Sigmoid())
    elif "fc" in name:
        model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features=num_in_features, out_features=hidden_size, bias=False),
        nn.LeakyReLU(.1,inplace=True),
        #nn.Dropout(dropout_rate),
        nn.Linear(in_features=hidden_size, out_features=1, bias=False),
        nn.Sigmoid())
    elif "classifier" in name:
        model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features=num_in_features, out_features=hidden_size, bias=False),
        nn.LeakyReLU(.1,inplace=True),
        #nn.Dropout(dropout_rate),
        nn.Linear(in_features=hidden_size, out_features=1, bias=False),
        nn.Sigmoid())
    elif "head" in name:
        model.head = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features=num_in_features, out_features=hidden_size, bias=False),
        nn.LeakyReLU(.1,inplace=True),
        #nn.Dropout(dropout_rate),
        nn.Linear(in_features=hidden_size, out_features=1, bias=False),
        nn.Sigmoid())
    print (model)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.0001)
    if i < 192:
        scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 600, T_mult=2, eta_min=1e-12, last_epoch=- 1, verbose=False)
    else:
        scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 300, T_mult=2, eta_min=1e-12, last_epoch=- 1, verbose=False)


    criterion = nn.BCELoss()
    test_size=len(test_dataset)
    dataloaders = {'train' : train_dataloader, 'val' : val_dataloader}
    dataset_sizes = {'train': train_size, 'val' : val_size}
    checkpoint = {'model': 'abc',
              'state_dict': model.state_dict(),
              'optimizer' : optimizer.state_dict(),
                 'threshold' : 0.5}
    
    
    
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=epochs)

    test_metrics = test_model(model, criterion,test_dataloader, threshold = .4)
    wandb.finish() 
    #model_scores.append(test_metrics.auc)
    
    # if i == 0:
        # best_model = model
        # best_model_score = model_scores[i]
    # else:
        # if model_scores[i] > best_model_score:
            # best_model score = model_scores[i] 

    gc.collect()
    torch.cuda.empty_cache()  
    
    
 
    if i%8 == 0:
        PATH='/home/mbadhan/Desktop/mberghouse/pytorch_models/weights/5_02_long/'+model_name+'_weights'
        torch.save(model.state_dict(), PATH)
        PATH='/home/mbadhan/Desktop/mberghouse/pytorch_models/models/5_02_long/'+model_name+'_model'
        torch.save(model, PATH)
    
    

