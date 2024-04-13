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


def train_model(model, model_name,criterion, optimizer, scheduler, num_epochs=25):
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

        if val_auc>.82:
            break
        # elif (tr_loss<.41)&(val_loss < .42):
            # break
        # if (tr_loss<.35):
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
    print(f'Best val auc: {best_f1:4f}')
    # load best model weights
    #model.load_state_dict(best_model_wts)
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

for i in range(20):
  if i>=0:
    if i<20:
        metadata_list=['mass_case_description_test_set','mass_case_description_train_set']
        im_set = "600x1000_v6"
    # elif i <12:



    df_list=[]
    for j in range(len(metadata_list)):
        print (metadata_list[j])
        df=pd.read_csv('../'+metadata_list[j]+'.csv')
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
            df.filename.iloc[k]='../CBIS-DDSM-preprocessed/'+im_set+'/'+df.filename.iloc[k].rsplit('/',3)[0]+'/1-1.png'
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
    test_size =  len(test_dataset)
    train_size = len(train_dataset)

    
    # if i <48:
        # batch_size = 10
    # else:
        # batch_size = 16
    batch_size=13

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, pin_memory = True,num_workers=0 )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle = True, pin_memory = True,num_workers=0 )
    


        
    #if i%2 == 0:
   # 	epochs = 6
   # else:
    #epochs=36
    
    if i%2 == 0:
        epochs = 12
    else:
        epochs = 8
    #epochs =14
    
  
    #epochs = np.random.randint(35,40)
    hidden_size = 64
    dropout = .01
    lr = 4.0e-6#np.random.uniform(2e-5,4e-5)
    #if i11:
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
 
    if i<20:
        model_name = 'davit_tiny.msft_in1k'
        #baseline_name
        epochs = 100
        #lr = 4e-5
        dropout = .2#np.random.uniform(0.05,.25)
        model = nn.Sequential(timm.create_model('davit_tiny.msft_in1k', drop_path_rate=.0,num_classes=1,pretrained=True), nn.Sigmoid())
        dataset = "600x1000-v6-mass"   


    print (model_name)
    print (im_set)
    print (dataset)
        
    
    #Initiate Weights and Biases Tracking
    wandb.init(
        # set the wandb project where this run will be logged
        project="Model Testing 8_31",
        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "architecture": model_name,
        "dataset": dataset,
        "epochs": epochs,
        "batch size": batch_size,
        "hidden size": hidden_size, 
        "optimizer_name": 'Adam',
        "dropout_rate": dropout,
        "weight decay": 1e-3,       
        }
    )
        
    dropout_rate = dropout
    #num_in_features = model.get_classifier().in_features
    
    # for name, param in model.named_parameters():
        # ijk=0
        # print (name)
        
    # # Replace the existing classifier. It's named: classifier
    # if "head.fc" in name:
        # model.head.fc = nn.Sequential(
        # nn.Dropout(dropout_rate),
        # nn.Linear(in_features=num_in_features, out_features=hidden_size, bias=False),
        # nn.GELU(),
        # #nn.Dropout(dropout_rate),
        # nn.Linear(in_features=hidden_size, out_features=1, bias=False),
        # nn.Sigmoid())
    # elif "fc" in name:
        # model.fc = nn.Sequential(
        # nn.Dropout(dropout_rate),
        # nn.Linear(in_features=num_in_features, out_features=hidden_size, bias=False),
        # nn.GELU(),
        # #nn.LeakyReLU(.1,inplace=True),
        # #nn.Dropout(dropout_rate),
        # nn.Linear(in_features=hidden_size, out_features=1, bias=False),
        # nn.Sigmoid())
    # elif "classifier" in name:
        # model.classifier = nn.Sequential(
        # nn.Dropout(dropout_rate),
        # nn.Linear(in_features=num_in_features, out_features=hidden_size, bias=False),
        # nn.GELU(),
        # #nn.Dropout(dropout_rate),
        # nn.Linear(in_features=hidden_size, out_features=1, bias=False),
        # nn.Sigmoid())
    # elif "head" in name:
        # model.head = nn.Sequential(
        # nn.Dropout(dropout_rate),
        # nn.Linear(in_features=num_in_features, out_features=hidden_size, bias=False),
        # nn.GELU(),
        # #nn.LeakyReLU(.1,inplace=True),
        # #nn.Dropout(dropout_rate),
        # nn.Linear(in_features=hidden_size, out_features=1, bias=False),
        # nn.Sigmoid())
    # #elif "neck" in name:
    # #    model.head = nn.Sequential(
    # #    nn.AdaptiveAvgPool2d((64,1)),
    # #    nn.Dropout(dropout_rate),
    # #    nn.Linear(in_features=1, out_features=64, bias=False),
        # #nn.LeakyReLU(.1,inplace=True),
        # #nn.Dropout(dropout_rate),
        # #nn.Linear(in_features=hidden_size, out_features=1, bias=False),
        # #nn.Sigmoid())
    # print (name)
    # print (model)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-4,lr=lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 2600, T_mult=2, eta_min=1e-12, last_epoch=- 1, verbose=False)
    # if i <48:
        # scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1700, T_mult=2, eta_min=1e-12, last_epoch=- 1, verbose=False)
    # else:
        # scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1200, T_mult=2, eta_min=1e-12, last_epoch=- 1, verbose=False)


    criterion = nn.BCELoss()
    test_size=len(test_dataset)
    dataloaders = {'train' : train_dataloader, 'val' : test_dataloader}
    dataset_sizes = {'train': train_size, 'val' : test_size}
    checkpoint = {'model': 'abc',
              'state_dict': model.state_dict(),
              'optimizer' : optimizer.state_dict(),
                 'threshold' : 0.5}
    
    
    
    model = train_model(model, model_name, criterion, optimizer, scheduler, num_epochs=epochs)

    test_metrics = test_model(model,  criterion,test_dataloader, threshold = .4)
    wandb.finish() 
    #model_scores.append(test_metrics.auc)
    
    # rem = i%3
    
    # if rem == 0:
    	# best_f1 = test_metrics['f1']
    	# best_model = model
    # else:
    	# if best_f1 < test_metrics['f1']:
    	   # best_f1 = test_metrics['f1']
    	   # best_model = model
 

    gc.collect()
    torch.cuda.empty_cache()  
    
    
 
    # if i%3 == 2:
    #PATH='/home/mbadhan/Desktop/mberghouse/pytorch_models/weights/6_29/'+model_name+metadata_list[0][0:3]+dataset[0:6]+str(i)+'_weights'
    #torch.save(model.state_dict(), PATH)
    #PATH='/home/mbadhan/Desktop/mberghouse/pytorch_models/models/6_29/'+model_name+metadata_list[0][0:3]+dataset[0:6]+str(i)+'_model'
   # torch.save(model, PATH)
    
    

