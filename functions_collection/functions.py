import numpy as np
import glob 
import os
from PIL import Image
import math
from scipy import ndimage
import SimpleITK as sitk
from scipy.spatial.distance import directed_hausdorff
from scipy.interpolate import RegularGridInterpolator
from nibabel.affines import apply_affine
from scipy.spatial import ConvexHull
from skimage.draw import polygon2mask
import re
import cv2 
from skimage.measure import label, regionprops
import torch
import torch.nn as nn
import torch.nn.functional as F

## important: DICE loss
def customized_dice_loss(pred, mask, num_classes, exclude_index = 10):
    # set valid mask to exclude pixels either equal to exclude index 
    valid_mask = (mask != exclude_index) 

    # one-hot encode the mask
    one_hot_mask = F.one_hot(mask, num_classes = exclude_index + 1).permute(0,3,1,2)


    # softmax the prediction
    pred_softmax = F.softmax(pred,dim = 1)

    pred_probs_masked = pred_softmax[:,1:num_classes,...] * valid_mask.unsqueeze(1)  # Exclude background class
    ground_truth_one_hot_masked = one_hot_mask[:,1:num_classes,...] * valid_mask.unsqueeze(1)
        
    # Calculate intersection and union, considering only the included pixels
    intersection = torch.sum(pred_probs_masked * ground_truth_one_hot_masked, dim=(0,2, 3))
    union = torch.sum(pred_probs_masked, dim = (0,2,3)) + torch.sum(ground_truth_one_hot_masked, dim=(0,2, 3))
        
    # Compute Dice score
    dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)  # Adding a small epsilon to avoid division by zero
        
    # Dice loss is 1 minus Dice score
    dice_loss = 1 - dice_score

    return torch.mean(dice_loss)



# function: define base, mid and apex segments given slice number N
def define_three_segments(N):
    arr = np.arange(0, N)

    # Calculate the lengths of the segments
    # Ensuring the middle segment has the most elements if the array can't be equally divided
    segment_length = N // 3
    segment_extra = N % 3

    if segment_extra == 1 or segment_extra == 0:
        # then assign this extra to mid
        base_segment = arr[:segment_length]
        mid_segment = arr[segment_length:2 * segment_length + segment_extra]
        apex_segment = arr[2 * segment_length + segment_extra:]

    else:
        # then assign 1 to base and assign 1 to mid
        base_segment = arr[:segment_length + 1]
        mid_segment = arr[segment_length + 1: 2 * segment_length + 2*1]
        apex_segment = arr[2 * segment_length + 2*1 :]

    return base_segment, mid_segment, apex_segment


# function: normalize the CMR image
def normalize_image(x, axis=(0,1,2)):
    # normalize per volume (x,y,z) frame
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x-mu)/(sd+1e-8)

# function: match nii img orientation to nrrd/dicom img orientation
def nii_to_nrrd_orientation(nii_data):
    return np.flip(np.rollaxis(np.flip(nii_data, axis=2),1,0), axis = 0)

# function: match nrrd/dicom img orientation to nii img orientation
def nrrd_to_nii_orientation(nrrd_data, format = 'nrrd'):
    if format[0:4] == 'nrrd':
        nrrd_data = np.rollaxis(nrrd_data,0,3)
    return np.rollaxis(np.flip(np.rollaxis(np.flip(nrrd_data, axis=0), -2, 2), axis = 2),1,0)

# function: remove disconnected regions in a binary image
def remove_scatter(img,target_label):
    new_img = np.copy(img)
    new_img[new_img == target_label] = 100
    need_to_remove = False
    for i in range(0,img.shape[2]):
        a = img[:,:,i]
        # check if there's label 1 in this slice
        if np.sum(a==target_label) == 0:
            continue
        labeled_image = label(a == target_label)
        regions = regionprops(labeled_image)
        region_sizes = [region.area for region in regions]
      
        if len(region_sizes) > 1:
            need_to_remove = True
        
          # Step 3: Find Largest Region Label
        largest_region_label = np.argmax(region_sizes) + 1  # Adding 1 because labels start from 1

        # Step 4: Create Mask for Largest Region
        largest_region_mask = (labeled_image == largest_region_label)

        # Step 5: Apply Mask to Original Image
        result_image = a.copy()
        result_image[~largest_region_mask] = 0
        new_slice = new_img[:,:,i]
        new_slice[result_image==target_label] = target_label
        new_slice[new_slice == 100] = 0
        new_img[:,:,i] = new_slice
    return new_img, need_to_remove

def remove_scatter3D(img,target_label):
    new_img = np.copy(img)
    new_img[new_img == target_label] = 100
    need_to_remove = False
   
    a = np.copy(img)
    
    labeled_image = label(a == target_label)
    regions = regionprops(labeled_image)
    region_sizes = [region.area for region in regions]
      
    if len(region_sizes) > 1:
        need_to_remove = True
        
    # Step 3: Find Largest Region Label
    largest_region_label = np.argmax(region_sizes) + 1  # Adding 1 because labels start from 1

    # Step 4: Create Mask for Largest Region
    largest_region_mask = (labeled_image == largest_region_label)

    # Step 5: Apply Mask to Original Image
    result_image = a.copy()
    result_image[~largest_region_mask] = 0
    new_img[result_image==target_label] = target_label
    new_img[new_img == 100] = 0
    return new_img, need_to_remove

def remove_scatter3D_smallest(img,target_label): # only remove the smallest 
    need_to_remove = False
    new_img = np.copy(img)

    labeled_image = label(new_img == target_label)
    regions = regionprops(labeled_image)
    region_sizes = np.asarray([region.area for region in regions])

    print('region_sizes:', region_sizes)

    if len(region_sizes) > 1:
        need_to_remove = True
        
    smallest_region_label = np.argmin(region_sizes) + 1
    smallest_region_mask = (labeled_image == smallest_region_label)
    new_img[smallest_region_mask] = 0
    
    return new_img, need_to_remove

def remove_scatter3D_minsize(img,target_label, min_size = 50): # only remove any regions with pixel number < min_size
    need_to_remove = False
    new_img = np.copy(img)
    labeled_image = label(new_img == target_label)
    regions = regionprops(labeled_image)
    region_sizes = np.asarray([region.area for region in regions])
        
    # Identify regions to be removed: Any region with size less than min_size
    remove_labels = np.where(region_sizes < min_size)[0] + 1

    if len(remove_labels) > 0:
        need_to_remove = True
    
        # Create a mask for regions to be removed
        remove_mask = np.isin(labeled_image, remove_labels)
        
        # Remove specified regions
        new_img = np.where(remove_mask, 0, new_img)

    return new_img, need_to_remove


# function: get first X numbers
# if we have 1000 numbers, how to get the X number of every interval numbers?
def get_X_numbers_in_interval(total_number, start_number, end_number , interval = 10):
    n = []
    for i in range(0, total_number, interval):
      n += [i + a for a in range(start_number,end_number)]
    n = np.asarray(n)
    return n

# function: save itk
def save_itk(img, save_file_name, previous_file, new_voxel_dim = None, new_affine = None):
    image = sitk.GetImageFromArray(img)

    image.SetDirection(previous_file.GetDirection())
    image.SetOrigin(previous_file.GetOrigin())

    if new_voxel_dim is None:
        image.SetSpacing(previous_file.GetSpacing())
    else:
        image.SetSpacing(new_voxel_dim)

    if new_affine is None:
        image.SetMetaData("TransformMatrix", previous_file.GetMetaData("TransformMatrix"))
    else:
        affine_matrix_str = np.array2string(new_affine, separator=',')
        image.SetMetaData("TransformMatrix", affine_matrix_str)


    sitk.WriteImage(image, save_file_name)


# function: find all files under the name * in the main folder, put them into a file list
def find_all_target_files(target_file_name,main_folder):
    F = np.array([])
    for i in target_file_name:
        f = np.array(sorted(glob.glob(os.path.join(main_folder, os.path.normpath(i)))))
        F = np.concatenate((F,f))
    return F

# function: find time frame of a file
def find_timeframe(file,num_of_dots,start_signal = '/',end_signal = '.'):
    k = list(file)

    if num_of_dots == 0: 
        num = [i for i,e in enumerate(k) if e== start_signal][-1]
        if end_signal is not None:
            num1 = [i for i,e in enumerate(k) if e== end_signal][-1]
        else:
            num1 = len(k)
        kk = k[num+1:num1]
    
    else:
        if num_of_dots == 1: #.png
            num1 = [i for i, e in enumerate(k) if e == end_signal][-1]
        elif num_of_dots == 2: #.nii.gz
            num1 = [i for i, e in enumerate(k) if e == end_signal][-2]
        num2 = [i for i,e in enumerate(k) if e== start_signal][-1]
        kk=k[num2+1:num1]


    total = 0
    for i in range(0,len(kk)):
        total += int(kk[i]) * (10 ** (len(kk) - 1 -i))
    return total

# function: sort files based on their time frames
def sort_timeframe(files,num_of_dots,start_signal = '/',end_signal = '.'):
    time=[]
    time_s=[]
    
    for i in files:
        a = find_timeframe(i,num_of_dots,start_signal,end_signal)
        time.append(a)
        time_s.append(a)
    time_s.sort()
    new_files=[]
    for i in range(0,len(time_s)):
        j = time.index(time_s[i])
        new_files.append(files[j])
    new_files = np.asarray(new_files)
    return new_files

# function: make folders
def make_folder(folder_list):
    for i in folder_list:
        os.makedirs(i,exist_ok = True)

# function: set window level
def set_window(image,level,width):
    if len(image.shape) == 3:
        image = image.reshape(image.shape[0],image.shape[1])
    new = np.copy(image)
    high = level + width // 2
    low = level - width // 2
    # normalize
    unit = (1-0) / (width)
    new[new>high] = high
    new[new<low] = low
    new = (new - low) * unit 
    return new

# function: save grayscale image
def save_grayscale_image(a,save_path,normalize = True, WL = 50, WW = 100): 
    I = np.zeros((a.shape[0],a.shape[1],3))
    # normalize
    if normalize == True:
        a = set_window(a, WL, WW)

    for i in range(0,3):
        I[:,:,i] = a
    
    Image.fromarray((I*255).astype('uint8')).save(save_path) 

# function: save grayscale image with mask
def save_grayscale_image_w_mask(img, mask, save_path,normalize = True, WL = 50, WW = 100):
    I = np.zeros((img.shape[0],img.shape[1],3))
    # normalize
    if normalize == True:
        img = set_window(img, WL, WW)

    grey_image = (img * 255).astype(np.uint8)
    rgb_image = cv2.cvtColor(grey_image, cv2.COLOR_GRAY2RGB)

    rgb_image[mask == 1] = [255, 0, 0]
    rgb_image[mask == 2] = [0, 255, 0]

    cv2.imwrite(save_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

# function: remove nan from list:
def remove_nan(l, show_row_index = False):
    l_new = []
    a = np.sum(np.isnan(l),axis = 1)
    non_nan_row_index = []
    for i in range(0,a.shape[0]):
        if a[i] == 0:
            l_new.append(l[i])
            non_nan_row_index.append(i)
    l_new = np.asarray(l_new)
    non_nan_row_index = np.asarray(non_nan_row_index)
    if show_row_index == True:
        return l_new, non_nan_row_index
    else:
        return l_new

# function: eucliean distance excluding nan:
def ED_no_nan(pred, gt):
    ED = []
    for row in range(0,gt.shape[0]):
        if np.isnan(gt[row,0]) == 1:
            continue
        else:
            ED.append(math.sqrt((gt[row,0] - pred[row,0]) ** 2 +  (gt[row,1] - pred[row,1]) ** 2))
    return sum(ED) / len(ED)

        
# function: normalize a vector:
def normalize(x):
    x_scale = np.linalg.norm(x)
    return np.asarray([i/x_scale for i in x])

# function: count pixels belonged to one class
def count_pixel(seg,target_val):
    index_list = np.where(seg == target_val)
    count = index_list[0].shape[0]
    pixels = []
    for i in range(0,count):
        p = []
        for j in range(0,len(index_list)):
            p.append(index_list[j][i])
        pixels.append(p)
    return count,pixels


# Dice calculation
def np_categorical_dice(pred, truth, target_class, exclude_class = None):
    if exclude_class is not None:
        valid_mask = (truth != exclude_class)
        pred = pred * valid_mask
        truth = truth * valid_mask

    """ Dice overlap metric for label k """
    A = (pred == target_class).astype(np.float32)
    B = (truth == target_class).astype(np.float32)
    return (2 * np.sum(A * B) + 1e-8) / (np.sum(A) + np.sum(B) + 1e-8)


def np_mean_dice(pred, truth, k_list = [1,2]):
    """ Dice mean metric """
    dsc = []
    for k in k_list:
        dsc.append(np_categorical_dice(pred, truth, k))
    return np.mean(dsc)


def HD(pred, gt, pixel_size, target_class, exclude_class = None, max_or_mean = 'max'):
    if exclude_class is not None:
        valid_mask = (gt != exclude_class)
        pred = pred * valid_mask
        gt = gt * valid_mask
    
    pred = (pred == target_class).astype(np.float32)
    gt = (gt == target_class).astype(np.float32)

    gt_points = np.argwhere(gt)
    pred_points = np.argwhere(pred)

    hd1 = directed_hausdorff(gt_points, pred_points)[0]
    hd2 = directed_hausdorff(pred_points, gt_points)[0]

    if max_or_mean == 'max':
        return max(hd1, hd2) * pixel_size
    else:
        return (hd1 + hd2) / 2 * pixel_size

# function: accuracy, sensitivity, specificity:
def quantitative(y_pred, y_true):
    accuracy = np.sum(y_pred == y_true) / len(y_true)

    # also calculate sensitivity and specificity, write the code please
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    # Sensitivity (True Positive Rate)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    # Specificity (True Negative Rate)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    return accuracy, sensitivity, specificity, TP, TN, FP, FN

# function: pick slices:
def pick_slices(all_slices, heart_slices, target_num):
    a = np.asarray(all_slices)

    b = np.asarray(heart_slices)

    need_more_num = target_num - b.shape[0]

    ahead_reach_end = False
    behind_reach_end = False

    ahead_num = need_more_num // 2
    if b[0] - ahead_num < 0:
        ahead_num = b[0]
        ahead_reach_end = True

    behind_num = need_more_num - ahead_num
    if b[-1] + behind_num >= a.shape[0]:
        behind_num = a.shape[0] -1 - b[-1]
        if ahead_reach_end == False:
            ahead_num = need_more_num - behind_num
            if b[0] - ahead_num < 0:
                ahead_num = b[0]
                ahead_reach_end = True

    final = np.concatenate((a[b[0]-ahead_num:b[0]], b, a[b[-1]+1 : b[-1]+behind_num+1]))
    return final


# function: coordinate conversion according to the affine matrix
def coordinate_convert(grid_points, target_affine, original_affine):
    return apply_affine(np.linalg.inv(target_affine).dot(original_affine), grid_points)

# function: interpolation
def define_interpolation(data,Fill_value=0,Method='nearest'):
    shape = data.shape
    [x,y,z] = [np.linspace(0,shape[0]-1,shape[0]),np.linspace(0,shape[1]-1,shape[1]),np.linspace(0,shape[-1]-1,shape[-1])]
    interpolation = RegularGridInterpolator((x,y,z),data,method=Method,bounds_error=False,fill_value=Fill_value)
    return interpolation


# function: reslice a mpr
def reslice_mpr(mpr_data,plane_center,x,y,x_s,y_s,interpolation):
    # plane_center is the center of a plane in the coordinate of the whole volume
    mpr_shape = mpr_data.shape
    new_mpr=[]
    centerpoint = np.array([(mpr_shape[0]-1)/2,(mpr_shape[1]-1)/2,0])
    for i in range(0,mpr_shape[0]):
        for j in range(0,mpr_shape[1]):
            delta = np.array([i,j,0])-centerpoint
            v = plane_center + (x*x_s)*delta[0]+(y*y_s)*delta[1]
            new_mpr.append(v)
    new_mpr=interpolation(new_mpr).reshape(mpr_shape)
    return new_mpr


# switch two classes in one image:
def switch_class(img, class1, class2):
    new_img = np.where(img == class1, class2, np.where(img == class2, class1, img))
    return new_img


# function: write txt file
def txt_writer(save_path,parameters,names):
    t_file = open(save_path,"w+")
    for i in range(0,len(parameters)):
        t_file.write(names[i] + ': ')
        for ii in range(0,len(parameters[i])):
            t_file.write(str(round(parameters[i][ii],2))+' ')
        t_file.write('\n')
    t_file.close()

def txt_writer2(save_path,record):
    t_file = open(save_path,"w+")
    for i in range(0,len(record)):
        r = record[i]
        t_file.write('slice '+ str(r[0]) + ', total_distance: ' + str(round(r[1],2)) + 'mm, vector mm: ' 
                        + str(round(r[2][0],2)) + ' ' + str(round(r[2][1],2)) + ' vector pixel: ' + str(round(r[3][0],2)) + ' ' + str(round(r[3][1],2))
                        + ' rotation: '+str(r[4]) + ' degree' )
        if i != (len(record) - 1):
            t_file.write('\n')
    t_file.close()


# function: from ID_00XX to XX:
def ID_00XX_to_XX(input_string):
    # Find the first non-zero number in the string
    match = re.search(r'\d+', input_string)
    if match:
        number_string = match.group()
        return int(number_string)
    else:
        return None
    
# function: from XX to ID_00XX:
def XX_to_ID_00XX(num):
    if num < 10:
        return 'ID_000' + str(num)
    elif num>= 10 and num< 100:
        return 'ID_00' + str(num)
    elif num>= 100 and num < 1000:
        return 'ID_0' + str(num)
    elif num >= 1000:
        return 'ID_' + str(num)


# function: split train and val data
def split_train_val(X,Y, cross_val_batch_num, val_batch_index, save_split_file = None):
    '''X and Y first dimension is the number of cases'''
    num_of_cases_in_each_batch = int(X.shape[0] / cross_val_batch_num)

    if os.path.isfile(save_split_file):
        batches = np.load(save_split_file)
    
    else:
        # find out which Y is 1 and which Y is 0
        Y_1_index = np.where(Y == 1)[0]
        Y_0_index = np.where(Y == 0)[0]
        # split these indexes into 10 groups with similar size
        Y_1_index_split = np.array_split(Y_1_index,cross_val_batch_num)

        batches = []; start = 0
        for b in range(0, cross_val_batch_num):
            current_num = Y_1_index_split[b].shape[0]
            end = start + (num_of_cases_in_each_batch - current_num)
            Y_0_batch = Y_0_index[start:end]
  
            batch = np.concatenate((Y_0_batch, Y_1_index_split[b]))
            batches.append(batch)
            start = end
        batches = np.asarray(batches); np.save(save_split_file, batches)

    # split the data into train and val, with the val batch index (Axis = 0) as validation dataset
    val_idx = batches[val_batch_index,:]
    train_idx = np.delete(batches, val_batch_index, 0).flatten()
    X_train = X[train_idx,:]; Y_train = Y[train_idx]
    X_val = X[val_idx,:]; Y_val =Y[val_idx]
    return X_train, Y_train,  train_idx, X_val, Y_val, val_idx


def erode_and_dilate(img_binary, kernel_size, erode = None, dilate = None):
    img_binary = img_binary.astype(np.uint8)

    kernel = np.ones(kernel_size, np.uint8)  

    if dilate is True:
        img_binary = cv2.dilate(img_binary, kernel, iterations = 1)

    if erode is True:
        img_binary = cv2.erode(img_binary, kernel, iterations = 1)
    return img_binary

# function: sample 15 time frames
def sample_temporal_series(X, L, es_tf):
    '''it samaples a temporal series of length L from a series of 1 - X, it will always include 1, es_tf, and X'''
    '''it returns a list of indices of the sampled series, starting from 0'''

    assert 1 < es_tf < X, "es_tf must be between 1 and X"
    assert L > 0, "L must be a positive integer"

    # If X is less than L, create a series from 1 to X and pad with X to reach L elements
    if X < L:
        series = list(range(1, X + 1)) + [X] * (L - X)
    else:
        # Initialize the series with 1, es_tf, and X
        series = [1, es_tf, X]

        # Calculate step size for the rest of the series
        remaining_slots = L - 3  # As 1, es_tf, and X are already included
        step = max(int(round((X - 1) / remaining_slots)), 1)

        # Add numbers to the series with the calculated step, avoiding duplicates
        current = 1
        while len(series) < L:
            current += step
            if current not in series and current < X:
                series.append(current)
            elif current >= X:
                # Decrease step size when overshooting
                step -= 1
                current = 1  # Reset current to start

        # Sort the series since 1, es_tf, and X could be anywhere
        series.sort()

    return [int(x - 1) for x in series]

# Example usage
X = 16  # Example X value
L = 15  # Desired length of the new series
es_tf = 15  # Example es_tf value
result = sample_temporal_series(X, L, es_tf)
print(result)

# function: convert zero start to ED
def convert_zerostart_to_ED(tf_segments, actual_ed_index, current_ed_index, total_num):
    # for situation where ED is not the first frame
    new_list = []
    for i in tf_segments:
        ii = i + (actual_ed_index - current_ed_index )
        if ii >= total_num:
            ii = ii - total_num
        new_list.append(ii)
    return new_list


def find_LV_enclosed_by_myo(image):
    image = (image ).astype(np.uint8) * 255

    x_indices, y_indices = np.where(image == 255)

    # Create a set of points from the white pixel positions
    points = np.vstack((x_indices, y_indices)).T

    # Compute the convex hull
    hull = ConvexHull(points)

    # Extract the vertices making up the convex hull
    hull_vertices = points[hull.vertices]

    # Use skimage.draw.polygon2mask to create a mask from the convex hull vertices
    image_shape = image.shape
    convex_hull_mask = polygon2mask(image_shape, hull_vertices)

    # differntiate the mask from the original image
    diff = convex_hull_mask - image
        
    LV = diff.astype(np.int32)

    return LV