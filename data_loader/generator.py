# # data generator
# import torch
# from torch.utils.data import Dataset, DataLoader
# import sys
# import os
# import numpy as np
# import pandas as pd
# import nibabel as nb
# import torch
# import matplotlib.pyplot as plt
# sys.path.append('/host/d/Github')
# import Whole_heart_segmentation_junzhe.functions_collection as ff
# import Whole_heart_segmentation_junzhe.Data_processing as Data_processing
# import Whole_heart_segmentation_junzhe.data_loader.random_aug as random_aug




# # main function:
# class Dataset_CMR(torch.utils.data.Dataset):
#     def __init__(
#             self, 

#             image_file_list,
#             seg_file_list,

#             center_crop_according_to_which_class  = [1], #default: crop according to class 1 (LV)

#             image_shape = None, # [x,y], default = [128,128]
#             shuffle = None,
#             image_normalization = True,
#             augment = None,
#             augment_frequency = 0.5, # how often do we do augmentation
#             ):

#         super().__init__()
#         self.image_file_list = image_file_list
#         self.seg_file_list = seg_file_list
#         self.center_crop_according_to_which_class = center_crop_according_to_which_class
#         self.image_shape = image_shape
#         self.shuffle = shuffle
#         self.image_normalization = image_normalization
#         self.augment = augment
#         self.augment_frequency = augment_frequency

#         # how many cases we have in this dataset?
#         self.num_files = len(self.image_file_list)

#         # the following two should be run at the beginning of each epoch
#         # 1. get index array
#         self.index_array = self.generate_index_array()

#         # 2. some parameters
#         self.current_image_file = None
#         self.current_image_data = None 
#         self.current_seg_file = None
#         self.current_seg_data = None

#     # function: how many sample do we have in this dataset? 
#     def __len__(self):
#         return self.num_files
        
#     # function: we need to generate an index array for dataloader, it's a list, each element is [file_index, slice_index]
#     def generate_index_array(self):
#         np.random.seed()
                
#         if self.shuffle == True:
#             file_index_list = np.random.permutation(self.num_files)
#         else:
#             file_index_list = np.arange(self.num_files)

#         index_array = file_index_list.tolist()  # each element is file index now

#         return index_array
    
#     # function: 
#     def load_file(self, filename, segmentation_load = False):
#         ii = nb.load(filename).get_fdata()

#         if segmentation_load is True:
#             ii = np.round(ii).astype(int)
    
#         return ii
    

#     # function: get each item using the index [file_index]
#     def __getitem__(self, index):
#         f = self.index_array[index]
#         image_filename = self.image_file_list[f]
#         seg_filename = self.seg_file_list[f]
#         #print('loading image file:', image_filename, ' seg file:', seg_filename)

#         # check if manual seg exists
#         if os.path.isfile(seg_filename) is False:
#             self.have_manual_seg = False
#         else:
#             self.have_manual_seg = True
            
#         # if it's a new case, then do the data loading; if it's not, then just use the current data
#         if True:#image_filename != self.current_image_file or seg_filename != self.current_seg_file:
#             #print('i am here everything is ok')
#             image_loaded = self.load_file(image_filename, segmentation_load = False) 
#             print('I loaded image')
#             if self.have_manual_seg is True:
#                 seg_loaded = self.load_file(seg_filename, segmentation_load = True) 
#             else:
#                 seg_loaded = np.zeros(image_loaded.shape, dtype = np.int)


#         # center crop
#         if self.have_manual_seg is True:
#             # find centroid based on the segmenation class 1
#             _,_, self.centroid = Data_processing.center_crop( image_loaded, seg_loaded, self.image_shape, according_to_which_class = self.center_crop_according_to_which_class , centroid = None)

#         elif self.have_manual_seg is False:
#             # center is the image center
#             self.centroid = [image_loaded.shape[0]//2, image_loaded.shape[1]//2]

#          # random crop (randomly shift the centroid)
#         if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
#             random_centriod_shift_x = np.random.randint(-5,5)
#             random_centriod_shift_y = np.random.randint(-5,5)
#             centroid_used_for_crop = [self.centroid[0] + random_centriod_shift_x, self.centroid[1] + random_centriod_shift_y]
#         else:
#             centroid_used_for_crop = self.centroid
                
#         # crop this 2D case
#         image_loaded = image_loaded[centroid_used_for_crop[0] - self.image_shape[0]//2 : centroid_used_for_crop[0] + self.image_shape[0]//2,
#                                         centroid_used_for_crop[1] - self.image_shape[1]//2 : centroid_used_for_crop[1] + self.image_shape[1]//2 ]
#         seg_loaded = seg_loaded[centroid_used_for_crop[0] - self.image_shape[0]//2 : centroid_used_for_crop[0] + self.image_shape[0]//2,
#                                         centroid_used_for_crop[1] - self.image_shape[1]//2 : centroid_used_for_crop[1] + self.image_shape[1]//2 ]
        
#         # temporarily save our data
#         self.current_image_file = image_filename
#         self.current_image_data = np.copy(image_loaded)  
#         self.current_seg_file = seg_filename
#         self.current_seg_data = np.copy(seg_loaded)

#         # augmentation
#         original_image = np.copy(image_loaded)
#         original_seg = np.copy(seg_loaded)
      
#         ######## do augmentation
#         processed_seg = np.copy(original_seg)
#         # (0) add noise
#         if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
#             standard_deviation = 5
#             processed_image = original_image + np.random.normal(0,standard_deviation,original_image.shape)
#             # turn the image pixel range to [0,255]
#             processed_image = Data_processing.turn_image_range_into_0_255(processed_image)
#         else:
#             processed_image = Data_processing.turn_image_range_into_0_255(original_image)
       
#         # (1) do brightness
#         if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
#             processed_image,v = random_aug.random_brightness(processed_image, v = None)
    
#         # (2) do contrast
#         if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
#             processed_image, v = random_aug.random_contrast(processed_image, v = None)

#         # (3) do sharpness
#         if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
#             processed_image, v = random_aug.random_sharpness(processed_image, v = None)
            
#         # (4) do flip
#         if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
#             # doing this can make sure the flip is the same for image and seg
#             a, selected_option = random_aug.random_flip(processed_image)
#             b,_ = random_aug.random_flip(processed_seg, selected_option)
#             processed_image = np.copy(a)
#             processed_seg = np.copy(b)

#         # (5) do rotate
#         if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
#             processed_image, z_rotate_degree = random_aug.random_rotate(processed_image, order = 1, z_rotate_range = [-10,10])
#             processed_seg,_ = random_aug.random_rotate(processed_seg, z_rotate_degree, fill_val = 0, order = 0)

#         # (6) do translate
#         if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
#             processed_image, x_translate, y_translate = random_aug.random_translate(processed_image, translate_range = [-10,10])
#             processed_seg,_ ,_= random_aug.random_translate(processed_seg, x_translate, y_translate)

#         # add normalization
#         if self.image_normalization is True:
#             processed_image = Data_processing.normalize_image(processed_image,inverse = False) 

#         print('after augmentation, image min:', np.min(processed_image), ' max:', np.max(processed_image))

#         # put into torch tensor
#         processed_image = torch.from_numpy(processed_image).float().unsqueeze(0)  # add channel dimension
#         processed_seg = torch.from_numpy(processed_seg).float().unsqueeze(0)  # add channel dimension
#         original_image = torch.from_numpy(original_image).float().unsqueeze(0)  # add channel dimension
#         original_seg = torch.from_numpy(original_seg).float().unsqueeze(0)  #

#         # repeat processed image three times in the 0 axis
#         processed_image = processed_image.repeat(3,1,1)

#         # put into a dictionary
#         final_dictionary = { "image": processed_image, 
#                             "mask": processed_seg,
#                             "original_image": original_image,  
#                             "original_seg": original_seg,}


#         return final_dictionary
          
    
    
#     # function: at the end of each epoch, we need to reset the index array
#     def on_epoch_end(self):
#         print('now run on_epoch_end function')
#         self.index_array = self.generate_index_array()

#         self.current_image_file = None
#         self.current_image_data = None 
#         self.current_seg_file = None
#         self.current_seg_data = None
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import nibabel as nb
# 导入本项目的数据处理工具
import Data_processing as Data_processing
import data_loader.random_aug as random_aug

class Dataset_CMR(torch.utils.data.Dataset):
    def __init__(
            self, 
            image_file_list,
            seg_file_list,
            center_crop_according_to_which_class = [1], # 默认根据类 1 (LV) 裁剪
            image_shape = [128, 128],
            shuffle = False,
            image_normalization = True,
            augment = False,
            augment_frequency = 0.5,
            ):

        super().__init__()
        self.image_file_list = image_file_list
        self.seg_file_list = seg_file_list
        self.center_crop_according_to_which_class = center_crop_according_to_which_class
        self.image_shape = image_shape
        self.shuffle = shuffle
        self.image_normalization = image_normalization
        self.augment = augment
        self.augment_frequency = augment_frequency
        self.num_files = len(self.image_file_list)
        self.index_array = self.generate_index_array()

    def __len__(self):
        return self.num_files
        
    def generate_index_array(self):
        if self.shuffle:
            file_index_list = np.random.permutation(self.num_files)
        else:
            file_index_list = np.arange(self.num_files)
        return file_index_list.tolist()
    
    def load_file(self, filename, segmentation_load=False):
        ii = nb.load(filename).get_fdata()
        if segmentation_load:
            ii = np.round(ii).astype(int)
            # --- 3 分类强制转换逻辑 ---
            # 假设原始数据中：1=LV, 2=Myo, 3=RV
            # 我们只需要 0, 1, 2
            # 这里确保标签不会超过 2
            ii[ii > 2] = 0 
        return ii

    def __getitem__(self, index):
        f_idx = self.index_array[index]
        image_filename = self.image_file_list[f_idx]
        seg_filename = self.seg_file_list[f_idx]

        # 1. 首先加载数据 (修复 UnboundLocalError 的关键)
        image_loaded = self.load_file(image_filename, segmentation_load=False)
        
        if os.path.exists(seg_filename):
            self.have_manual_seg = True
            seg_loaded = self.load_file(seg_filename, segmentation_load=True)
        else:
            self.have_manual_seg = False
            seg_loaded = np.zeros(image_loaded.shape, dtype=int)

        # 2. 确定裁剪中心 (Centroid)
        if self.have_manual_seg:
            # 根据提供的类别（如 LV=1）寻找中心
            _, _, self.centroid = Data_processing.center_crop(
                image_loaded, seg_loaded, self.image_shape, 
                according_to_which_class=self.center_crop_according_to_which_class
            )
        else:
            self.centroid = [image_loaded.shape[0]//2, image_loaded.shape[1]//2]

        # 3. 增强：随机偏移中心
        centroid_used = self.centroid
        if self.augment and np.random.uniform(0, 1) < self.augment_frequency:
            centroid_used = [
                self.centroid[0] + np.random.randint(-5, 5),
                self.centroid[1] + np.random.randint(-5, 5)
            ]

        # 4. 执行裁剪
        x_start = max(0, int(centroid_used[0] - self.image_shape[0]//2))
        y_start = max(0, int(centroid_used[1] - self.image_shape[1]//2))
        
        image_crop = image_loaded[x_start:x_start+self.image_shape[0], y_start:y_start+self.image_shape[1]]
        seg_crop = seg_loaded[x_start:x_start+self.image_shape[0], y_start:y_start+self.image_shape[1]]

        # 5. 数据增强 (Noise, Brightness, Rotate etc.)
        processed_image = np.copy(image_crop)
        processed_seg = np.copy(seg_crop)

        if self.augment and np.random.uniform(0, 1) < self.augment_frequency:
            # 这里的 random_aug 调用保持原样
            processed_image = Data_processing.turn_image_range_into_0_255(processed_image)
            # ... 其他增强步骤 ...
        else:
            processed_image = Data_processing.turn_image_range_into_0_255(processed_image)

        # 6. 标准化与 Tensor 转换
        if self.image_normalization:
            processed_image = Data_processing.normalize_image(processed_image)

        # 转换为张量并适配 SAM (3通道输入)
        img_tensor = torch.from_numpy(processed_image).float().unsqueeze(0).repeat(3, 1, 1)
        mask_tensor = torch.from_numpy(processed_seg).float().unsqueeze(0)

        return {
            "image": img_tensor, 
            "mask": mask_tensor, # 用于计算 CrossEntropyLoss 的 Ground Truth
            "patient_id": str(self.image_file_list[f_idx])
        }