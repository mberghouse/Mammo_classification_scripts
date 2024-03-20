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
    transforms.ColorJitter(brightness=(.7,1.2)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    #transforms.Resize((1024,1024))
])

small_aug = transforms.Compose([
    # input for augmentator is always PIL image
    transforms.ToPILImage(),
    
    transforms.ToTensor(),
    #transforms.Resize((1024,1024))
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


def train_model(model, model_name,criterion, optimizer, scheduler, dataloaders, fold, num_epochs=25):
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
            # if epoch == 18:
                # for name, param in model.named_parameters():
                    # if ('att' in name)|('head' in name):  # Change 'att' to match the name of your attention submodule
                        # param.requires_grad = False
                # for name, param in model.named_parameters():
                    # print(f'Parameter name: {name}, Trainable: {param.requires_grad}')
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
                    outputs = outputs.to(device)
                    #labels = labels.to(device)
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
            epoch_f1 = metricf1(all_outputs>threshold, all_labels)
            epoch_acc = accuracy(all_outputs>threshold, all_labels)
            epoch_precision = precision(all_outputs>threshold , all_labels)
            epoch_recall = recall(all_outputs>threshold , all_labels)
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
                #if val_metrics['f1'][-1] > best_f1:
                 #   best_f1 = val_metrics['f1'][-1]
                  #  best_model_wts = model.state_dict()
                   # checkpoint['threshold'] = threshold
                    #torch.save(checkpoint, 'checkpoint.pth')
               
        # cant be formated in strin g
        tr_loss, tr_acc, tr_f1, tr_prec, tr_rec, tr_auc = train_metrics['loss'][-1], train_metrics['acc'][-1],  train_metrics['f1'][-1], train_metrics['precision'][-1], train_metrics['recall'][-1], train_metrics['auc'][-1]
        val_loss, val_acc, val_f1, val_prec, val_rec, val_auc = val_metrics['loss'][-1], val_metrics['acc'][-1], val_metrics['f1'][-1], val_metrics['precision'][-1], val_metrics['recall'][-1], val_metrics['auc'][-1]
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{num_epochs}, learning rate: {lr}')
        print(f'Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}, Train f1: {tr_f1:.4f}, Train Precision: {tr_prec:.4f}, Train Recall: {tr_rec:.4f}, Train AUC: {tr_auc:.4f}')
        print(f'Valitadion Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}, Vall f1: {val_f1:.4f}, Val Precision: {val_prec:.4f}, Val Recall: {val_rec:.4f}, Val AUC: {val_auc:.4f}')
        if (fold==2)|(fold==3): 
            if (tr_f1>.73)&(val_f1 >=.75):
                break
            elif (epoch>40)&(val_f1 >=.6):
                break                
        else:
            if (tr_f1>.8)&(val_f1 >=.82):
                break
            elif (epoch>40)&(val_f1 >=.75):
                break



  

        # if (tr_f1>.82)&(val_f1 >=.85):
           # break
        # elif (epoch>19)&(val_f1 >=.8):
           # break
        # elif (epoch>30)&(val_f1 >=.75):
           # break
        # elif (epoch>38)&(val_f1 >=.7):
           # break
        # elif (epoch>50)&(val_f1 >=.65):
           # break
        # elif (epoch>60)&(val_f1 >=.6):
           # break
    
        
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
    #print(f'Best val auc: {best_f1:4f}')
    # load best model weights
    #model.load_state_dict(best_model_wts)
    return train_metrics, val_metrics
    
    
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
    threshold = 0.5
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
    # wandb.log({"test_acc": epoch_acc, "test_loss": epoch_loss, "test_f1": epoch_f1, "test_precision": epoch_precision
                         # , "test_recall": epoch_recall, "test_auc": epoch_auc})
    print(f'Inference complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'F1 Score = : {epoch_f1:4f}')
    print(f'AUC Score = : {epoch_auc:4f}')
    print(f'Acc Score = : {epoch_acc:4f}')
    return test_metrics
    
    
import wandb
import timm
import numpy as np
model_scores=[]

for i in range(39):
  if i>=15:
    #if i<20:
        # metadata_list=['mass_case_description_test_set','calc_case_description_train_set']
        # im_set = "300x500_v6"
    # elif i <12:
        # metadata_list=['calc_case_description_test_set','calc_case_description_train_set']
        # im_set = "300x500_v6"
    # elif i<18:
        # metadata_list=['mass_case_description_test_set','mass_case_description_train_set']
        # im_set = "600x1000_v6"
    # elif i<24:
        # metadata_list=['calc_case_description_test_set','calc_case_description_train_set']
        # im_set = "600x1000_v6"
    # elif i<30:
        # metadata_list=['mass_case_description_test_set','mass_case_description_train_set']
        # im_set = "600x1000_v6"
    # elif i<36:
        # metadata_list=['calc_case_description_test_set','calc_case_description_train_set']
        # im_set = "600x1000_v6"
    # elif i<42:
        # metadata_list=['mass_case_description_test_set','mass_case_description_train_set']
        # im_set = "600x1000_v6"
    # elif i<48:
        # metadata_list=['calc_case_description_test_set','calc_case_description_train_set']
        # im_set = "600x1000_v6"
    # elif i<54:
        # metadata_list=['mass_case_description_test_set','mass_case_description_train_set']
        # im_set = "600x1000_v6"
    # elif i<60:
        # metadata_list=['calc_case_description_test_set','calc_case_description_train_set']
        # im_set = "600x1000_v6"
    # elif i<66:
        # metadata_list=['mass_case_description_test_set','mass_case_description_train_set']
        # im_set = "300x500_v6"
    # elif i<72:
        # metadata_list=['calc_case_description_test_set','calc_case_description_train_set']
        # im_set = "300x500_v6"
    # elif i<78:
        # metadata_list=['mass_case_description_test_set','mass_case_description_train_set']
        # im_set = "300x500_v6"
    # elif i<84:
        # metadata_list=['calc_case_description_test_set','calc_case_description_train_set']
        # im_set = "300x500_v6"
    # elif i<90:
        # metadata_list=['mass_case_description_test_set','mass_case_description_train_set']
        # im_set = "300x500_v6"
    # elif i<96:
        # metadata_list=['calc_case_description_test_set','calc_case_description_train_set']
        # im_set = "300x500_v6"
    # elif i<102:
        # metadata_list=['mass_case_description_test_set','mass_case_description_train_set']
        # im_set = "300x500_v6"
    # else:
        # metadata_list=['calc_case_description_test_set','calc_case_description_train_set']
        # im_set = "300x500_v6"


    import pandas as pd
    top_dir='/home/mbadhan/Desktop/mberghouse/inbreast/'
    df=pd.read_csv(top_dir+'inbreast/INbreast Release 1.0/INbreast.csv', delimiter=';')

    filenames_test=[]
    labels_test=[]
    filenames=[]
    labels=[]

    filenames_train=[]
    labels_train=[]
    for m in range(len(df)):
        sname = str(df['File Name'].iloc[m])+'__MG_'
        fname = top_dir+'inbreast/INbreast Release 1.0/600x1000/'+sname+df.Laterality.iloc[m]+'_'+df.View.iloc[m][0:2]+'_ANON.png'
        try:
            birad=float(df['Bi-Rads'].iloc[m][-2])
        except:
            birad=float(df['Bi-Rads'].iloc[m][-1])
        if birad<5:
            label = 0
        else:
            label = 1
        filenames.append(fname)
        labels.append(label)
    
    
        #if m%6!=0:
        #    filenames_train.append(fname)
        #    labels_train.append(label)
        #else:
        #    filenames_test.append(fname)
        #    labels_test.append(label)
        
    import torchvision
    import numpy.matlib as np_mlb
    from sklearn.model_selection import train_test_split
    
    from sklearn.model_selection import KFold

    # Assuming filenames and labels are already loaded
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    batch_size=10

# Convert your list of filenames and labels to numpy arrays
    filenames = np.array(filenames)
    labels = np.array(labels)
    total_train_metrics = {'loss': 0, 'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0}
    total_val_metrics = {'loss': 0, 'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0}


    for fold, (train_idx, valid_idx) in enumerate(kf.split(filenames, labels)):
        print(f"Processing Fold {fold + 1}")
    
        train_filenames, valid_filenames = filenames[train_idx], filenames[valid_idx]
        train_labels, valid_labels = labels[train_idx], labels[valid_idx]
        
        # Create datasets for this fold
        train_dataset = CBISDataset(train_labels, train_filenames, transform=augmentator)
        valid_dataset = CBISDataset(valid_labels, valid_filenames, transform=small_aug)
        
        # Create data loaders for this fold
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=batch_size)
        val_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=batch_size)

        # dataloaders dictionary for this fold
        dataloaders = {'train': train_dataloader, 'val': val_dataloader}

        # Train the model for this fold
        

        # No need to test the model separately for each fold as the validation set in cross-validation 
        # serves the same purpose. But if you have a completely separate test set, you can test the model here.

    
    #filenames_train, filenames_test, labels_train, labels_test = train_test_split(np.array(filenames), np.array(labels), stratify=np.array(labels), test_size=.2)
            
    #dataset = CBISDataset(labels_train,filenames_train,transform=augmentator)

    #val_pct = 0.01
    #val_size = int(val_pct * len(dataset))
    #train_size = len(dataset) - val_size
    #train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    #test_dataset = CBISDataset(labels_test,filenames_test, transform=small_aug)
    #val_size =  len(val_dataset)
    #test_size =  len(test_dataset)
    #train_size = len(train_dataset)

    
    # if i <48:
        # batch_size = 10
    # else:
        # batch_size = 16
    

    #test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
    #train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, pin_memory = True,num_workers=batch_size)
    #val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle = True, pin_memory = True,num_workers=batch_size )
    


        
    #if i%2 == 0:
   # 	epochs = 6
   # else:
    #epochs=36
    
        #if i%2 == 0:
         #   epochs = 12
        #else:
         #   epochs = 8
        #epochs =14
        
        epochs = 80

        #epochs = np.random.randint(35,40)
        hidden_size = 512
        dropout = .00
        lr = 1e-4#np.random.uniform(2e-5,4e-5)
        #if i<1:
        #    model_name = 'CECAresnet50'
        #    model = timm.create_model(model_name, pretrained=True)
        #    lr = np.random.uniform(4e-5,8e-5)
        #    dataset = "300x500-v6-calc"  
        
    #Hi, do you need to use the computer right now?

           
          
    #################   
         ### Regnetx ###
         #################  
     ###########################              
     ######### Mass ############
     ###########################      
     
        if i<2:
            model_name = 'CBAMregnetx_064'
            #baseline_name
            epochs = 65
            #lr = 2e-5
            dropout = .0#np.random.uniform(0.05,.25)
            model = timm.create_model(model_name, pretrained=True)
            dataset ="full-data"   
            
                
        elif i<4:
            #epochs = 35
            model_name = 'CBAMresnet50'
            #epochs=12
            #lr = 2e-5
            model = timm.create_model(model_name, pretrained=True)
            #lr = np.random.uniform(3e-5,5e-5)
            #dropout = np.random.uniform(.01,.03)
            dataset ="full-data"   
            
            

            
        elif i<6:
            model_name = 'CBAMdensenet169'
            #epochs = np.random.randint(24,30)
            model = timm.create_model(model_name, pretrained=True)
            #lr = 2e-5
            dataset ="full-data"   
            #epochs=12


           
            
         ###########################              
     ######### Calc ############
     ###########################        

           
        elif i<8:
            model_name = 'CBAMregnetx_064'
            #epochs=12
            #lr = 2.5e-5#np.random.uniform(2.5e-5,5.5e-5)
            #dropout = np.random.uniform(.05,.15)
            model = timm.create_model(model_name, pretrained=True)
            #lr = np.random.uniform(3e-5,5e-5)
            dataset ="full-data"   

            
        elif i<10:
            #epochs=12
            #dropout = .01
            model_name = 'CBAMresnet50'
            #epochs = np.random.randint(24,30)

            model = timm.create_model(model_name, pretrained=True)
            #lr = 2e-5
            dataset ="full-data"   
            

            
        elif i<11:
            #epochs=12
            #dropout = .01
            model_name = 'CBAMdensenet169'
            model = timm.create_model(model_name, pretrained=True)
            #lr = 2e-5
            dataset ="full-data"   
    #################   
         ### CBAM 600x1000 ###
         #################  
         
        elif i<12:
            model_name = 'CBAMregnetx_064'
            #dropout = np.random.uniform(.01,.03)
            model = timm.create_model(model_name, pretrained=True)
            # lr = 2e-5
            dataset ="full-data"   
            #epochs=12
                
        elif i<13:
            #epochs=12
            model_name = 'CBAMresnet50'
            #epochs = np.random.randint(24,30)
            model = timm.create_model(model_name, pretrained=True)
            #lr = 2e-5
            #dropout = np.random.uniform(.01,.03)
            dataset ="full-data"   
            
            

            
        elif i<14:
            model_name = 'CBAMdensenet169'
            #epochs=12
            model = timm.create_model(model_name, pretrained=True)
           # lr = 2e-5
            dataset ="full-data"   


           
            
         ###########################              
     ######### Calc ############
     ###########################        

           
        elif i<17:
            model_name = 'CBAMregnetx_064'
            #lr = 2e-5
            #dropout = .01#np.random.uniform(.01,.03)
            model = timm.create_model(model_name, pretrained=True)
            #epochs = 12
            dataset ="full-data"   

            
        elif i<19:
            #epochs = 12
            #dropout = .01
            model_name = 'CBAMresnet50'
            model = timm.create_model(model_name, pretrained=True)
            #lr = 2e-5
            dataset ="full-data"   
            

            
        elif i<21:
            #epochs=12
            #dropout = .01
            model_name = 'CBAMdensenet169'
            model = timm.create_model(model_name, pretrained=True)
          #  lr = 2e-5
            dataset ="full-data"   
            
         #################   
         ### Densenets ###
         #################  
     ###########################              
     ######### Mass ############
     ###########################       
            
           
        elif i<23:
            model_name = 'ECAregnetx_064'
            #epochs = 12
            model = timm.create_model(model_name, pretrained=True, global_pool='max')
          #  lr = 2e-5
            dataset ="full-data"
           # print (model)
            
        elif i<25:
           model_name = 'ECAdensenet169'
           #epochs = 12
           model = timm.create_model(model_name, pretrained=True)
           lr = 2e-5
           dataset ="full-data"   
            
       
            
        elif i<27:
            model_name = 'ECAresnet50'
            #epochs = 12
            #epochs = np.random.randint(31,32)
            model = timm.create_model(model_name, pretrained=True)
          #  lr = 2e-5
            dataset ="full-data"   
            
            

     
     ###########################              
     ######### Calc ############
     ###########################       
            
            
         
        

            
         #################   
         ### Resnets ###
         #################  
     ###########################              
     ######### Mass ############
     ###########################       
                  
        elif i<29:
            model_name = 'SEresnet50'
            model = timm.create_model(model_name, pretrained=True)
            #lr = 2e-5
            dataset ="full-data"   
            #epochs=12
            

           
        elif i<31:
            model_name = 'SEdensenet169'
            #lr = 2e-5
            model = timm.create_model(model_name, pretrained=True)
            dataset ="full-data"   
            #epochs=12
            
        elif i<33:
            model_name = 'SEregnetx_064'
            #lr = 3e-5
            model = timm.create_model(model_name, pretrained=True)
            dataset ="full-data"   
            #epochs=12
        

           
            
     ###########################              
     ######### Calc ############
     ###########################       
                    

     # ###########################              
     # ########## Mass ############
     # ###########################       
                  
        elif i<35:
            model_name = 'tv_resnet50'
            model = timm.create_model(model_name, pretrained=True)
            #lr = 3e-5
            dataset ="full-data"   
            #epochs=12

           
        elif i<37:
            model_name = 'densenet169'
            model = timm.create_model(model_name, pretrained=True)
            #lr = 3e-5
            dataset ="full-data"   
            #epochs=12
            
        elif i<39:
            model_name = 'regnetx_064'
            #epochs=14
            model = timm.create_model(model_name, pretrained=True)
           # lr = 3e-5
            dataset ="full-data"   
          #  epochs=12
        

           
            
     ###########################              
     ######### Calc ############
     ###########################       
                    
        elif i<56:
            model_name = 'tv_resnet50'
            #baseline_name
            #epochs=8
            #dropout = .01
            model = timm.create_model(model_name, pretrained=True)
          #  lr = 4e-5
            dataset ="full-data"   

            
        elif i<58:
            model_name = 'densenet169'
            #epochs=8
            #dropout = .01
            model = timm.create_model(model_name, pretrained=True)
            #lr = 4e-5
            dataset ="full-data"   
            
            
        elif i<60:
            model_name = 'regnetx_064'
            #epochs=8
            #dropout = .01
            model = timm.create_model(model_name, pretrained=True)
           # lr = 4e-5
            dataset ="full-data"   
            
            
             # ###########################              
     # ######### Mass ############
     # ###########################       
                  
        elif i<62:
            model_name = 'tv_resnet50'
            model = timm.create_model(model_name, pretrained=True)
           # lr = 3e-5
            dataset ="full-data"   
            #epochs=12

           
        elif i<64:
            model_name = 'densenet169'
            model = timm.create_model(model_name, pretrained=True)
           # lr = 3e-5
            dataset ="full-data"   
            #epochs=12
            
        elif i<66:
            model_name = 'regnetx_064'
            #epochs=14
            model = timm.create_model(model_name, pretrained=True)
           # lr = 3e-5
            dataset ="full-data"   
            #epochs=12
        

           
            
     ###########################              
     ######### Calc ############
     ###########################       
                    
        elif i<68:
            model_name = 'tv_resnet50'
            #baseline_name
            #epochs=8
            #dropout = .01
            model = timm.create_model(model_name, pretrained=True)
           # lr = 5e-5
            dataset ="full-data"   

            
        elif i<70:
            model_name = 'densenet169'
            #epochs=8
            #dropout = .01
            model = timm.create_model(model_name, pretrained=True)
          #  lr = 5e-5
            dataset ="full-data"   
            
            
        elif i<72:
            model_name = 'regnetx_064'
            #epochs=8
           # dropout = .01
            model = timm.create_model(model_name, pretrained=True)
           # lr = 5e-5
            dataset ="full-data"   
         # #################   
         # ### Regnetx ###
         # #################  
     # ###########################              
     # ######### Mass ############
     # ###########################      
     
        elif i<74:
            model_name = 'ECAresnet50'
            #epochs =12
            #baseline_name
            model = timm.create_model(model_name, pretrained=True)
          #  lr = 3e-5
            dataset ="full-data"   
            
        elif i<76:
            model_name = 'ECAresnet50'
            #epochs =12
            #baseline_name
            model = timm.create_model(model_name, pretrained=True)
           # lr = 3e-5
            dataset ="full-data"   
            
            
            
        elif i<78:
            model_name = 'SEresnet50'
            #epochs =12
            model = timm.create_model(model_name, pretrained=True)
         #   lr = 3e-5
            dataset ="full-data"   

           
            
         ###########################              
     ######### Calc ############
     ###########################        

           
        elif i<80:
            model_name = 'ECAresnet50'
            #epochs =8
            model = timm.create_model(model_name, pretrained=True)
         #   lr = 3e-5
            dataset ="full-data"   
            
        elif i<82:
            model_name = 'ECAresnet50'
            #epochs =8
            model = timm.create_model(model_name, pretrained=True)
            #epochs = 10
        #    lr = 3e-5
            dataset ="full-data"   
            
        elif i<84:
            model_name = 'SEresnet50'
            model = timm.create_model(model_name, pretrained=True)
          #  lr = 3e-5
            #epochs =8
            dataset ="full-data"   

        
         

            
         #################   
         ### Densenets ###
         #################  
     ###########################              
     ######### Mass ############
     ###########################       
            
           

            
        elif i<86:
            model_name = 'SEdensenet169'
            #epochs =12
            model = timm.create_model(model_name, pretrained=True)
         #   lr = 3e-5
            dataset ="full-data"   
            
        elif i<88:
            model_name = 'ECAdensenet169'
            #epochs =12
            model = timm.create_model(model_name, pretrained=True)
       #     lr = 3e-5
            dataset ="full-data"   
            #epochs=10
            
            
        elif i<90:
            model_name = 'SEdensenet169'
            model = timm.create_model(model_name, pretrained=True)
          #  lr = 3e-5
            dataset ="full-data"   
            #epochs =12
           

     
     ###########################              
     ######### Calc ############
     ###########################       
            

        
           
        elif i<92:
            model_name = 'SEdensenet169'
            #epochs =8
            #epochs = np.random.randint(22,24)
            model = timm.create_model(model_name, pretrained=True)
           # lr = 3e-5
            dataset ="full-data"   
            
        elif i<94:
            model_name = 'ECAdensenet169'
            #epochs =8
            #epochs = np.random.randint(22,24)
            model = timm.create_model(model_name, pretrained=True)
           #lr = 3e-5
            dataset ="full-data"   
            #epochs=10
        elif i<96:
            model_name = 'SEdensenet169'
            #epochs =8
         #   lr = 3e-5
            model = timm.create_model(model_name, pretrained=True)
            #lr = np.random.uniform(6e-5,9e-5)
            dataset ="full-data"   
            

            
         #################   
         ### Resnets ###
         #################  
     ###########################              
     ######### Mass ############
     ###########################       
                  
        elif i<98:
            model_name = 'SEregnetx_064'
            #baseline_name
            #epochs =12
            model = timm.create_model(model_name, pretrained=True)
         #   lr = 3e-5
            dataset ="full-data"   
        
        elif i<100:
            model_name = 'ECAregnetx_064'
            #baseline_name
            model = timm.create_model(model_name, pretrained=True)
          #  lr = 3e-5
            dataset ="full-data"   
            #epochs =12
            # if i == 87:
                # epochs = 10
            # else:
                # epochs = np.random.randint(19,22)
            
        elif i<102:
            model_name = 'ECAregnetx_064'
          #  lr = 3e-5
            #epochs = np.random.randint(22,24)
            model = timm.create_model(model_name, pretrained=True)
            #lr = np.random.uniform(3e-5,4e-5)
            dataset ="full-data"   
            
           
            
     ###########################              
     ######### Calc ############
     ###########################       
                    
        elif i<104:
            model_name = 'SEregnetx_064'
            #baseline_name
            #epochs =8
            model = timm.create_model(model_name, pretrained=True)
           # lr = 3e-5
            dataset ="full-data"   
            

        elif i<106:
            model_name = 'ECAregnetx_064'
           # epochs =8
            model = timm.create_model(model_name, pretrained=True)
          #  lr = 3e-5
            dataset ="full-data"   
            
            
        else:
            model_name = 'ECAregnetx_064'
            #epochs =8
            #epochs = np.random.randint(16,22)
            model = timm.create_model(model_name, pretrained=True)
         #  lr = 3e-5
            dataset ="full-data"   
 


        print (model_name)
        #print (model.classifier)

        print (dataset)
            
        
        #Initiate Weights and Biases Tracking
        wandb.init(
            # set the wandb project where this run will be logged
            project="Model Testing cross-val 10-21_600x1000",
            # track hyperparameters and run metadata
            config={
            "learning_rate": lr,
            "architecture": model_name,
            "dataset": dataset,
            "epochs": epochs,
            "batch size": batch_size,
            "fold number": fold, 
            "optimizer_name": 'Adam',
            "dropout_rate": dropout,
            "weight decay": 1e-3,       
            }
        )
            
        dropout_rate = 0
        num_in_features = model.get_classifier().in_features
        import torch.nn as nn


# Flatten the output
        flatten = nn.Flatten(start_dim=1, end_dim=-1)

# Combine the adaptive pooling and flattening
        

        
        for name, param in model.named_parameters():
            ijk=0
            print (name)
            
        #Replace the existing classifier. It's named: classifier
        if "head.fc" in name:
            model.head.fc = nn.Sequential(
            #nn.AdaptiveAvgPool2d(output_size=1),

            #nn.Dropout(dropout_rate),
            nn.Linear(in_features=num_in_features, out_features=512, bias=False),
            nn.LeakyReLU(.1),
            nn.Linear(in_features=512, out_features=256, bias=False),
            nn.LeakyReLU(.1),
            nn.Linear(in_features=256, out_features=64, bias=False),
            nn.LeakyReLU(.1),
            nn.Linear(in_features=64, out_features=1, bias=False),
            #nn.ReLU(),
            nn.Sigmoid()
            )
        elif "fc" in name:
            model.fc = nn.Sequential(
            # nn.Dropout(dropout_rate),
            nn.Linear(in_features=num_in_features, out_features=512, bias=False),
            nn.LeakyReLU(.1),
            nn.Linear(in_features=512, out_features=256, bias=False),
            nn.LeakyReLU(.1),
            nn.Linear(in_features=256, out_features=64, bias=False),
            nn.LeakyReLU(.1),
            nn.Linear(in_features=64, out_features=1, bias=False),
            #nn.ReLU(),
            nn.Sigmoid()
            )
        elif "classifier" in name:
            model.classifier = nn.Sequential(
            nn.Linear(in_features=num_in_features, out_features=512, bias=False),
            nn.LeakyReLU(.1),
            nn.Linear(in_features=512, out_features=256, bias=False),
            nn.LeakyReLU(.1),
            nn.Linear(in_features=256, out_features=64, bias=False),
            nn.LeakyReLU(.1),
            nn.Linear(in_features=64, out_features=1, bias=False),
            #nn.ReLU(),
            nn.Sigmoid()
            )
        elif "head" in name:
            model.head = nn.Sequential(
            nn.Linear(in_features=num_in_features, out_features=512, bias=False),
            nn.LeakyReLU(.1),
            nn.Linear(in_features=512, out_features=256, bias=False),
            nn.LeakyReLU(.1),
            nn.Linear(in_features=256, out_features=64, bias=False),
            nn.LeakyReLU(.1),
            nn.Linear(in_features=64, out_features=1, bias=False),
            #nn.ReLU(),
            nn.Sigmoid()
            )
        # #elif "neck" in name:
        # #    model.head = nn.Sequential(
        # #    nn.AdaptiveAvgPool2d((64,1)),
        # #    nn.Dropout(dropout_rate),
        # #    nn.Linear(in_features=1, out_features=64, bias=False),
            # #nn.LeakyReLU(.1,inplace=True),
            # #nn.Dropout(dropout_rate),
            # #nn.Linear(in_features=hidden_size, out_features=1, bias=False),
            # #nn.Sigmoid())
        #print (name)
        print (model)
        

# Verify which layers are trainable
        

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        model.to(device)

        from torch.optim import lr_scheduler
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
        #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1980, 28010, 29075], gamma=0.1)
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=.333, total_iters=20)
        
        #scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 2600, T_mult=2, eta_min=1e-12, last_epoch=- 1, verbose=False)
        criterion = nn.BCELoss().to(device)
        #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(device))
#def train_model(model, model_name,criterion, optimizer, scheduler, num_epochs=25):

        train_metrics, val_metrics = train_model(model, model_name, criterion, optimizer, scheduler, dataloaders, fold,num_epochs=epochs)
        #for key in total_train_metrics.keys():
         #   total_train_metrics[key] += train_metrics[key][-1]  # Assuming you want the last epoch's metrics
          #  total_val_metrics[key] += val_metrics[key][-1]
        
        
        #model = train_model(model, model_name, criterion, optimizer, scheduler, num_epochs=epochs)

        #if fold <=4:
         #   wandb.finish() 
        #else:
         #   for key in total_train_metrics.keys():
          #      total_train_metrics[key] /= kf.get_n_splits()
           #     total_val_metrics[key] /= kf.get_n_splits()

    # Log the average metrics to wandb
            #wandb.log({"avg_train_acc": total_train_metrics['acc'], "avg_train_loss": total_train_metrics['loss'], 
              # "avg_train_f1": total_train_metrics['f1'], "avg_train_precision": total_train_metrics['precision'],
               #"avg_train_recall": total_train_metrics['recall'], "avg_train_auc": total_train_metrics['auc']})

            #wandb.log({"avg_val_acc": total_val_metrics['acc'], "avg_val_loss": total_val_metrics['loss'], 
             #  "avg_val_f1": total_val_metrics['f1'], "avg_val_precision": total_val_metrics['precision'],
               #"avg_val_recall": total_val_metrics['recall'], "avg_val_auc": total_val_metrics['auc']})
        wandb.finish() 
 

        gc.collect()
        torch.cuda.empty_cache()  
        #print (val_metrics['f1'][-1])
        #if val_metrics['f1'][-1]>=.75:
            #print (val_metrics.f1)
         #   PATH='/home/mbadhan/Desktop/mberghouse/inbreast/inbreast/INbreast Release 1.0/weights_300x500/'+'run3_'+model_name+'_600x1000_'+str(fold)+'_'+str(i)
         #   torch.save(model.state_dict(), PATH)
          #  PATH='/home/mbadhan/Desktop/mberghouse/inbreast/inbreast/INbreast Release 1.0/models_300x500/'+'run3_'+model_name+'_600x1000_'+str(fold)+'_'+str(i)
           # torch.save(model, PATH)
    
 
    
    

