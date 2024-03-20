import matplotlib.pyplot as plt
import torch
from torch.nn.functional import softmax, interpolate
from torchvision.io.image import read_image
from torchvision.io import ImageReadMode
from torchvision.models import resnet18
from torchvision.transforms.functional import normalize, resize, to_pil_image

from torchcam.methods import SmoothGradCAMpp, LayerCAM
from torchcam.utils import overlay_mask
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


small_aug = transforms.Compose([
    # input for augmentator is always PIL image
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

class CBISDataset(Dataset):
    """CBIS-DDSM dataset."""

    def __init__(self, labels, filenames, transform=None):

        self.labels = labels
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fname = self.filenames[idx]
        image = cv2.imread(fname)#, torchvision.io.ImageReadMode.RGB)
        label = self.labels[idx]    
        if self.transform:
            image = self.transform(image)

        return image, label, fname

import os
import pydicom
import numpy as np
from PIL import Image
import cv2
from pydicom.pixel_data_handlers import apply_windowing, apply_voi_lut

def _fix_photometric_interpretation(self, img, scan):
    if scan.PhotometricInterpretation == 'MONOCHROME1':
        return img.max() - img
    elif scan.PhotometricInterpretation == 'MONOCHROME2':
        return img - img.min()
    else:
        raise ValueError("Invalid Photometric Interpretation: {}"
                           .format(scan.PhotometricInterpretation))

# Cast into 8-bits for saving
def normalize_to_255(img):
    if img.max() != 0:
        img = img / img.max()
    img *= 255
    return img.astype(np.uint8)

# Flip the breast horizontally on the chosen side 
def _flip_breast_side(self, img):
    img_breast_side = self._determine_breast_side(img)
    if img_breast_side == self.breast_side:
        return img
    else:
        return np.fliplr(img)    

# Determine the current breast side
def _determine_breast_side(self, img):
    col_sums_split = np.array_split(np.sum(img, axis=0), 2)
    left_col_sum = np.sum(col_sums_split[0])
    right_col_sum = np.sum(col_sums_split[1])
    if left_col_sum > right_col_sum:
        return 'L'
    else:
        return 'R'

# Crop the useless background of the image
def _crop(self, img):
    try:
        bin_img = self._binarize(img, threshold=1)
        contour = self._extract_contour(bin_img)
        img = self._erase_background(img, contour)
        x1, x2 = np.min(contour[:, :, 0]), np.max(contour[:, :, 0])
        y1, y2 = np.min(contour[:, :, 1]), np.max(contour[:, :, 1])
#             x1, x2 = int(0.99 * x1), int(1.01 * x2)
#             y1, y2 = int(0.99 * y1), int(1.01 * y2)
        return img[y1:y2, x1:x2]
    except:
        print ('crop didnt work')
        return img

# Binarize the image at the threshold
def _binarize(self, img, threshold):
    return (img > threshold).astype(np.uint8)

# Get contour points of the breast
def _extract_contour(self, bin_img):
    contours, _ = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    return contour

# Set to background pixels of the image to zero
def _erase_background(self, img, contour):
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    output = cv2.bitwise_and(img, mask)
    return output

# Resize the image to the preprocessor size
def _resize(self, img):
    return cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)

# Get the save path of a given dicom file
def _get_save_path(self, path, png, save_dir):
    patient = path.split('/')[-2]
    filename = path.split('/')[-1]
    if png:
        filename = filename.replace('dcm', 'png')
    else:
        filename = filename.replace('dcm', 'jpeg')
    if save_dir:
        save_path = os.path.join(self.save_root, save_dir, patient, filename)
    else:
        save_path = os.path.join(self.save_root, patient, filename)
    return save_path

# Save the preprocessed image
def _save_image(self, img, path, png, save_dir):
    save_path = self._get_save_path(path, png, save_dir)
    patient_folder = os.path.split(save_path)[0]
    os.makedirs(patient_folder, exist_ok=True)
    cv2.imwrite(save_path, img)
        
def OpenMask(mask, ksize=(23, 23), operation="open"):
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=ksize)    
    if operation == "open":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)    
    # Then dilate
    edited_mask = cv2.morphologyEx(edited_mask, cv2.MORPH_DILATE, kernel)
    return edited_mask

def SortContoursByArea(contours, reverse=True):   

    # Sort contours based on contour area.
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)    
    # Construct the list of corresponding bounding boxes.
    bounding_boxes = [cv2.boundingRect(c) for c in sorted_contours]
    return sorted_contours, bounding_boxes

def DrawContourID(img, bounding_box, contour_id):    

    # Center of bounding_rect.
    x, y, w, h = bounding_box
    center = ( ((x + w) // 2), ((y + h) // 2) )
    # Draw the countour number on the image
    cv2.putText(img=img,
                text=f"{contour_id}",
                org=center, # Bottom-left corner of the text string in the image.
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=10, 
                color=(255, 255, 255),
                thickness=40)
    return img

def XLargestBlobs(mask, top_X=None):
    

    # Find all contours from binarised image.
    # Note: parts of the image that you want to get should be white.
    contours, hierarchy = cv2.findContours(image=mask,
                                           mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_NONE)   
    n_contours = len(contours)    
    # Only get largest blob if there is at least 1 contour.
    if n_contours > 0:        
        # Make sure that the number of contours to keep is at most equal 
        # to the number of contours present in the mask.
        if n_contours < top_X or top_X == None:
            top_X = n_contours        
        # Sort contours based on contour area.
        sorted_contours, bounding_boxes = SortContoursByArea(contours=contours,
                                                             reverse=True)        
        # Get the top X largest contours.
        X_largest_contours = sorted_contours[0:top_X]        
        # Create black canvas to draw contours on.
        to_draw_on = np.zeros(mask.shape, np.uint8)        
        # Draw contours in X_largest_contours.
        X_largest_blobs = cv2.drawContours(image=to_draw_on, # Draw the contours on `to_draw_on`.
                                           contours=X_largest_contours, # List of contours to draw.
                                           contourIdx=-1, # Draw all contours in `contours`.
                                           color=1, # Draw the contours in white.
                                           thickness=-1) # Thickness of the contour lines.        
    return n_contours, X_largest_blobs


import cv2
import numpy as np

def select_largest_obj(img_bin, lab_val=255, fill_holes=False, 
                       smooth_boundary=False, kernel_size=15):

    n_labels, img_labeled, lab_stats, _ = \
        cv2.connectedComponentsWithStats(img_bin, connectivity=8, 
                                         ltype=cv2.CV_32S)
    largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1
    largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
    largest_mask[img_labeled == largest_obj_lab] = lab_val
    # import pdb; pdb.set_trace()
    if fill_holes:
        bkg_locs = np.where(img_labeled == 0)
        bkg_seed = (bkg_locs[0][0], bkg_locs[1][0])
        img_floodfill = largest_mask.copy()
        h_, w_ = largest_mask.shape
        mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
        cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed, 
                      newVal=lab_val)
        holes_mask = cv2.bitwise_not(img_floodfill)  # mask of the holes.
        largest_mask = largest_mask + holes_mask
    if smooth_boundary:
        kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN, 
                                        kernel_)
    return largest_mask


def max_pix_val(dtype):
    if dtype == np.dtype('uint8'):
        maxval = 2**8 - 1
    elif dtype == np.dtype('uint16'):
        maxval = 2**16 - 1
    else:
        raise Exception('Unknown dtype found in input image array')
    return maxval


def suppress_artifacts( img, global_threshold=.05, fill_holes=False, 
                       smooth_boundary=True, kernel_size=15):

    maxval = max_pix_val(img.dtype)
    if global_threshold < 1.:
        low_th = int(img.max()*global_threshold)
    else:
        low_th = int(global_threshold)
    _, img_bin = cv2.threshold(img, low_th, maxval=maxval, 
                               type=cv2.THRESH_BINARY)
    breast_mask = select_largest_obj(img_bin, lab_val=maxval, 
                                          fill_holes=True, 
                                          smooth_boundary=True, 
                                          kernel_size=kernel_size)
    img_suppr = cv2.bitwise_and(img, breast_mask)

    return img_suppr

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
    
    
def segment_breast(img, low_int_threshold=.05, crop=True):

    # Create img for thresholding and contours.
    img_8u = (img.astype('float32')/img.max()*255).astype('uint8')
    if low_int_threshold < 1.:
        low_th = int(img_8u.max()*low_int_threshold)
    else:
        low_th = int(low_int_threshold)
    _, img_bin = cv2.threshold(
        img_8u, low_th, maxval=255, type=cv2.THRESH_BINARY)
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        contours,_ = cv2.findContours(
            img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours,_ = cv2.findContours(
            img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [ cv2.contourArea(cont) for cont in contours ]
    idx = np.argmax(cont_areas)  # find the largest contour, i.e. breast.
    breast_mask = cv2.drawContours(
        np.zeros_like(img_bin), contours, idx, 255, -1)  # fill the contour.
#     # segment the breast.
    img_breast_only = cv2.bitwise_and(img, img, mask=breast_mask)
    x,y,w,h = cv2.boundingRect(contours[idx])
    if crop:
        img_breast_only = img_breast_only[y:y+h, x:x+w]
    return  breast_mask

def crop( img, mask):
    try:
        bin_img = binarize(img, threshold=1)
        contour = extract_contour(bin_img)
        img = erase_background(img, contour)
        x1, x2 = np.min(contour[:, :, 0]), np.max(contour[:, :, 0])
        y1, y2 = np.min(contour[:, :, 1]), np.max(contour[:, :, 1])
        return mask[y1:y2, x1:x2]
    except:
        print ('crop didnt work')
        return mask
    
def binarize( img, threshold):
    return (img > threshold).astype(np.uint8)
    
    # Get contour points of the breast
def extract_contour( bin_img):
    contours, _ = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    return contour
    
# Set to background pixels of the image to zero
def erase_background( img, contour):
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    output = cv2.bitwise_and(img, mask)
    return output

def flip_breast( img, mask):
    col_sums_split = np.array_split(np.sum(img, axis=0), 2)
    left_col_sum = np.sum(col_sums_split[0])
    right_col_sum = np.sum(col_sums_split[1])
    if left_col_sum > right_col_sum:
        return img, mask
    else:
        return np.fliplr(img),np.fliplr(mask)
     

def preprocess_new(image,mask,size):
    dim=(image.shape[0],image.shape[1])
    # print ('1',mask, mask.shape)
    n=.00
    n2=.00
    image = image[int(0+dim[0]*n):int(dim[0]-dim[0]*n),int(0+dim[1]*n2):int(dim[1]-dim[1]*n2)].astype('uint8')
    mask = mask[int(0+dim[0]*n):int(dim[0]-dim[0]*n),int(0+dim[1]*n2):int(dim[1]-dim[1]*n2)].astype('uint8')
    image=suppress_artifacts(image)
    mask_img = OpenMask(mask=image, ksize=(14, 14), operation="open")
    blob_img=select_largest_obj(mask_img)
    image[blob_img==0]=0
    # print (mask)
    image, mask=flip_breast(image, mask)
    # print ('2',mask)
    mask=crop(image, mask)
    # print (mask)
    mask=cv2.resize(mask,size,interpolation = cv2.INTER_LANCZOS4)
    #image=cv2.resize(image,size,interpolation = cv2.INTER_LANCZOS4)
    return  mask

def apply_windowing(img2d,ds):
    img2d = apply_voi_lut(img2d, ds)
    # convert to float to avoid overflow or underflow losses.
    if ds.PhotometricInterpretation == 'MONOCHROME1':
        img2d = np.amax(img2d)-img2d
    # min-max scale
    img2d = (img2d - img2d.min()) / (img2d.max() - img2d.min())
    # convert to uint8
    img2d = (img2d * 255).astype(np.uint8)
    img = cv2.cvtColor(img2d, cv2.COLOR_GRAY2BGR)
    
    return img

def apply_rescale(pixel_data, slope, intercept):
    return pixel_data * slope + intercept

def delete_existing_images(dest_dir, filename_without_extension):
    for ext in ['.png', '.jpg']:
        file_path = os.path.join(dest_dir, filename_without_extension + ext)
        if os.path.exists(file_path):
            os.remove(file_path)
        
def dicom2png(dcm_path):
    # Ensure destination directory exists
    #wc=1600
    from scipy import stats
    ww=1500
    # Read the DICOM file
    ds = pydicom.dcmread(dcm_path)
    if stats.mode(ds.pixel_array, axis=None)[0]!=0:
        im = np.max(ds.pixel_array)-ds.pixel_array
    else:
        im=ds.pixel_array
    return im
    #im = preprocess_new(im,(300,500))
    
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
    for inputs, labels, fname in tqdm(dataloader):
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
    print(f'Inference complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'F1 Score = : {epoch_f1:4f}')
    print(f'AUC Score = : {epoch_auc:4f}')
    print(f'Acc Score = : {epoch_acc:4f}')
    return test_metrics, threshold

 
def find_cam_thresh(cam, masks):
    iou=[]
    threshes=[]
    
    for j in range(4,45):
        #print (i)
        cam_tensor = torch.clone(cam)
        thresh = j*.02
        cam_tensor[cam_tensor>=thresh]=1
        cam_tensor[cam_tensor<thresh]=0
        cam_tensor=torch.Tensor(cam_tensor).to(torch.long)
        try:
            if cam_tensor[0][0]<-1000:
                cam_tensor = (cam_tensor/torch.max(cam_tensor)) 
        except:
            print ('exception')
        final_mask = masks
#         for i in range(len(masks)):
#             #print (masks[i].shape)
#             mask_bin=np.copy(masks[i][:,:,0])
#             mask_bin[mask_bin>0]=1
#             final_mask = final_mask + mask_bin
        final_mask = torch.tensor(final_mask)
        final_mask[final_mask>0]=1
        iou.append(jaccard(cam_tensor,final_mask))
        threshes.append(thresh)
        #print (thresh)
        #print(jaccard(cam_tensor,final_mask))
    best_iou=np.max(iou)
    idx = iou.index(max(iou))
    best_thresh = threshes[idx]
    # cam_tensor = torch.clone(cam)
    # cam_tensor[cam_tensor>=best_thresh]=1
    # cam_tensor[cam_tensor<best_thresh]=0
    # cam_tensor=torch.Tensor(cam_tensor).to(torch.long)
    # try:
        # if cam_tensor[0][0]<-1000:
            # cam_tensor = (cam_tensor/torch.max(cam_tensor)) 
    # except:
        # print ('exception')


    return best_iou, best_thresh,  final_mask



def save_overlay(cam_tensor, inputs, cam_path, final_mask):
    cam_im=np.array(cam_tensor.squeeze(0).cpu())
    mask = np.array(final_mask.squeeze(0).cpu())*260
    #print (cam_im.shape)
    alpha=.7
    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = cv2.resize(cam_im,(inputs.shape[3],inputs.shape[2]))
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    #print (np.max(mask))
    
    im=(np.asarray(inputs.squeeze(0).permute(1, 2, 0).to('cpu'))*255).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img =(alpha * im + (1 - alpha) * overlay).astype(np.uint8)
    #print (overlayed_img.shape)
    # for i in range(len(masks)):
        # print (masks[i].shape)
        # mask = masks[i]#cv2.resize(masks[i], [inputs.shape[3],inputs.shape[2]])
    overlayed_img[:,:,0] = ((overlayed_img[:,:,0]+mask)/2).astype('uint8')
    overlayed_img[:,:,1] = ((overlayed_img[:,:,1]+mask)/2).astype('uint8')
    overlayed_img[:,:,2] = ((overlayed_img[:,:,2]+mask)/2).astype('uint8')
    im_final = pil.fromarray(overlayed_img, "RGB")
    im_final.save(cam_path)
    
from skimage.draw import polygon
import numpy as np
import plistlib    
def load_inbreast_mask(mask_path, image):
    """
    Hàm này dùng để đọc file XML
    @mask_path : Path tới file XML
    @imshape : shape của ảnh đầu ra vd. [4084, 3328]
    return: numpy array where positions in the roi are assigned a value of 1.
    """

    def load_point(point_string):
        x, y = tuple([float(num) for num in point_string.strip('()').split(',')])
        return y, x
    imshape = image.shape
    mask = np.zeros(imshape)
    with open(mask_path, 'rb') as mask_file:
        plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML)['Images'][0]
        numRois = plist_dict['NumberOfROIs']
        rois = plist_dict['ROIs']
        assert len(rois) == numRois
        for roi in rois:
            numPoints = roi['NumberOfPoints']
            points = roi['Point_px']
            assert numPoints == len(points)
            points = [load_point(point) for point in points]
            if len(points) <= 2:
                for point in points:
                    mask[int(point[0]), int(point[1])] = 1
            else:
                x, y = zip(*points)
                x, y = np.array(x), np.array(y)
                poly_x, poly_y = polygon(x, y, shape=imshape)
                mask[poly_x, poly_y] = 1
    return mask        
        
        
from torchcam.methods import GradCAMpp, LayerCAM

device='cuda'

from torchmetrics import JaccardIndex
import os
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from tensorflow import keras
from PIL import Image as pil
colormap='jet'


metricf1 = BinaryF1Score()
precision = BinaryPrecision()
recall = BinaryRecall()
accuracy = BinaryAccuracy()
roc = BinaryROC()
auc = BinaryAUROC()

mask_folders_path='/home/mbadhan/Desktop/mberghouse/inbreast/inbreast/INbreast Release 1.0/AllXML/'
path_dcm= '/home/mbadhan/Desktop/mberghouse/inbreast/inbreast/INbreast Release 1.0/AllDICOMs/'
dcm_list = os.listdir(path_dcm)

mask_folders=os.listdir(mask_folders_path)
# folders_path = "/home/mbadhan/Desktop/mberghouse/CBIS-DDSM-png/"
# folders=os.listdir(folders_path)
model_names = os.listdir('/home/mbadhan/Desktop/mberghouse/inbreast/inbreast/INbreast Release 1.0/models_300x500/')
size=(300,500)
final_stats=[]
print (model_names)
count2 = 0


for model_name in model_names:
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
        if '600' in model_name:
            fname = top_dir+'inbreast/INbreast Release 1.0/600x1000/'+sname+df.Laterality.iloc[m]+'_'+df.View.iloc[m][0:2]+'_ANON.png'
        else:
            fname = top_dir+'inbreast/INbreast Release 1.0/300x500/'+sname+df.Laterality.iloc[m]+'_'+df.View.iloc[m][0:2]+'_ANON.png'
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
        
    import torchvision
    import timm
    import numpy.matlib as np_mlb
    from sklearn.model_selection import train_test_split
    
    from sklearn.model_selection import KFold

    # Assuming filenames and labels are already loaded
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    batch_size=1

# Convert your list of filenames and labels to numpy arrays
    filenames = np.array(filenames)
    labels = np.array(labels)
    total_train_metrics = {'loss': 0, 'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0}
    total_val_metrics = {'loss': 0, 'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0}


    for fold, (train_idx, valid_idx) in enumerate(kf.split(filenames, labels)):
        print(f"Processing Fold {fold + 1}")
        ##print (model_name)
        #print (model_name.split('_')[3])
        if str(fold) == model_name.split('_')[3]:
   
            valid_filenames = filenames[valid_idx]
            valid_labels = labels[valid_idx]
            
            # Create datasets for this fold
            #train_dataset = CBISDataset(train_labels, train_filenames, transform=augmentator)
            valid_dataset = CBISDataset(valid_labels, valid_filenames, transform=small_aug)
            
            # Create data loaders for this fold
            #train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=batch_size)
            val_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=batch_size)
    
            PATH_m='/home/mbadhan/Desktop/mberghouse/inbreast/inbreast/INbreast Release 1.0/models_300x500/'+model_name
            PATH_w='/home/mbadhan/Desktop/mberghouse/inbreast/inbreast/INbreast Release 1.0/weights_300x500/'+model_name
            model = torch.load(PATH_m).eval()
            model.load_state_dict(torch.load(PATH_w),strict=False)
            #, map_location=torch.device('cpu')
            #optimizer = torch.optim.Adam(model.parameters(), lr=4e-5, weight_decay=.0001)
            #scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, T_mult=2, eta_min=1e-12, last_epoch=- 1, verbose=False)
            criterion = nn.BCELoss()
            test_metrics, threshold = test_model(model, criterion,val_dataloader, threshold = .45)
            #print (test_metrics)
            threshold = .5
            if 'dense' in model_name:
                layer_list=['features.denseblock1.denselayer6.conv2',
                'features.denseblock2.denselayer12.conv2',
                'features.denseblock3.denselayer24.conv2','features.denseblock4.denselayer16.conv2']
            elif 'regnetx' in model_name:
                layer_list=['s2.b1.conv3','s3.b1.conv3',
                's4.b1.conv3','s1.b1.conv3']
            elif 'resnet' in model_name:
                layer_list=['layer2.2.conv3','layer3.2.conv3',
                'layer1.2.conv3','layer4.2.conv3']
            # Get your input
            #n=.1
            print (fname[80:88])
            #print (fname[0][76:84])
            #cam_path = '
            count=0
            alpha=.5
            TPc=0
            TNc=0
            FNc=0
            FPc=0
            iou_score=[]
            designation=[]
            count=0
            masks=None
            mask=None
            #cam_extractor = LayerCAM(model, layer_list)
        #('C:/Users/marcb/Desktop/CBIS-DDSM-preprocessed/300x500_v6/Calc-Test_P_00038_LEFT_CC/1-1.png',)
            for inputs,labels,fname in val_dataloader:
                mask_num = 1
                
                if count%20==0:
                    print (fname)
                count=count+1
                # if count>0:
                no_jacc = 0
                
                for k in dcm_list:
                    if fname[0][80:88] in k:
                        dcm_path = path_dcm+k
                
                #print (folders_path+fname[0][67:-8]+'/1-1.png')
                image=dicom2png(dcm_path)
                mask_path=mask_folders_path+fname[0][80:88]+'.xml'
                #print (fname[0][81:89])
                try:
                    mask = load_inbreast_mask(mask_path, image)
                    #print ('MASK FOUND')
                except:
                    print ('mask not found')
                    mask_num=0
                if mask_num!=0:
                    if len(mask[mask==0]) >100:
                        mask = preprocess_new(image, mask,size)
                    else:
                        mask_num=0
                

                inputs = inputs.float()
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                ##threshold=.4
                ##with torch.no_grad():
                    ##fpr, tpr, thresholds = roc(outputs.squeeze(0).cpu(), labels.cpu())
                    #threshold, _ = find_optim_thres(fpr, tpr, thresholds)

                preds = (outputs > threshold).long()

                if labels == 1 :
                    if preds!=labels:
                        val='Benign Prediction for Malignant Truth'
                        v='_FN'
                        FNc=FNc+1
                    else:
                        val='Malignant Prediction for Malignant Truth'
                        v='_TP'
                        TPc=TPc+1
                else:
                    if preds==labels:
                        val='Benign Prediction for Benign Truth'
                        v='_TN'
                        TNc=TNc+1
                    else:
                        val='Malignant Prediction for Benign Truth'
                        v='_FP'
                        FPc=FPc+1
                        
                #print (fname)
                #cam_path=os.path.join('/home/mbadhan/Desktop/mberghouse/grad_cam/5_31',model_name,str(fname[0][57:-8]),v+'.jpg')
                #try:
                #    os.makedirs(os.path.join('/home/mbadhan/Desktop/mberghouse/grad_cam/5_31/',model_name,str(fname[0][57:-8])))
                #except:
                #    continue
                gc.collect()
                torch.cuda.empty_cache()
                cam_extractor = LayerCAM(model, layer_list)
                scores = model(inputs)
                
                cams=cam_extractor(scores.squeeze(0).argmax().item(), scores)
                jaccard = JaccardIndex(task='binary')
                fused_cam = cam_extractor.fuse_cams(cams)
                #
                if mask_num==0:
                    iou_score.append(1)
                    designation.append(v)
                else:
                    cam_im_bin = np.copy(fused_cam.cpu())
                    cam_im_bin = cv2.resize(cam_im_bin.squeeze(0),(inputs.shape[3],inputs.shape[2]))
                    cam_tensor=torch.Tensor(cam_im_bin)
                    cam_tensor = (cam_tensor/torch.max(cam_tensor))
                    iou , thresh, final_mask= find_cam_thresh(cam_tensor,mask)
                    #save_overlay(cam_tensor, inputs, cam_path, final_mask)
                    
                    iou_score.append(iou)
                    designation.append(v)
                    #print ('IOU: '+str(iou))
                    #print ('Thresh: '+str(thresh))
                    cam_extractor.remove_hooks()
            #print (iou)
            # gc.collect()
            # torch.cuda.empty_cache()

                    
                        
    mean_fp = np.nanmean(np.array(iou_score)[np.array(designation)=='_FP'])
    mean_fn = np.nanmean(np.array(iou_score)[np.array(designation)=='_FN'])
    mean_tp = np.nanmean(np.array(iou_score)[np.array(designation)=='_TP'])
    mean_tn = np.nanmean(np.array(iou_score)[np.array(designation)=='_TN'])
    iou_arr=np.array(iou_score)
    mean_total = np.nanmean(iou_arr[iou_arr!=1])
    print ('FP: ' + str(mean_fp))
    print ('FN: ' + str(mean_fn))
    print ('TP: ' + str(mean_tp))
    print ('TN: ' + str(mean_tn))
    print ('total: '+str(mean_total))

    final_stats.append([model_name,mean_fp,mean_fn,mean_tp,mean_tn,mean_total,test_metrics['auc']])
pd.DataFrame(final_stats).to_csv("/home/mbadhan/Desktop/mberghouse/inbreast/final_stats_11_8_300x500.csv")
