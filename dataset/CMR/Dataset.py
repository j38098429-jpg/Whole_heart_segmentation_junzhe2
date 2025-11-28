#!/usr/bin/env python

import sys
sys.path.append('/workspace/Documents')
import torch
import numpy as np
import os
import pandas as pd
import nibabel as nb


from torch.utils.data import Dataset, DataLoader

import original_SAM.Data_processing as Data_processing
import original_SAM.functions_collection as ff
import original_SAM.dataset.CMR.random_aug as random_aug

# main function:
class Dataset_CMR(torch.utils.data.Dataset):
    def __init__(
            self, 
            patient_list_spreadsheet_file,

            image_file_list,
            seg_file_list,

            return_arrays_or_dictionary = 'dictionary', # "arrays" or "dictionary"
            center_crop_according_to_which_class = None,

            image_shape = None, # [x,y], channel =  tf always 15 
            shuffle = None,
            image_normalization = True,
            augment = None,
            augment_frequency = 0.5, # how often do we do augmentation
            ):

        super().__init__()
     
        self.patient_list_spreadsheet = pd.read_excel(patient_list_spreadsheet_file)

        self.image_file_list = image_file_list
        self.seg_file_list = seg_file_list

        self.image_shape = image_shape
        self.shuffle = shuffle
        self.image_normalization = image_normalization
        self.augment = augment
        self.augment_frequency = augment_frequency
        self.return_arrays_or_dictionary = return_arrays_or_dictionary
        self.center_crop_according_to_which_class = center_crop_according_to_which_class

        # how many cases we have in this dataset?
        self.num_files = len(self.image_file_list)

        # the following two should be run at the beginning of each epoch
        # 1. get index array
        self.index_array = self.generate_index_array()
        # 2. some parameters
        self.current_image_file = None
        self.current_image_data = None 
        self.current_seg_file = None
        self.current_seg_data = None

    # function: how many cases do we have in this dataset?
    def __len__(self):
        return  self.num_files
        

    # function: we need to generate an index array for dataloader, it's a list, each element is [file_index, slice_index]
    def generate_index_array(self):
        np.random.seed()
        index_array = []
                
        if self.shuffle == True:
            file_index_list = np.random.permutation(self.num_files)
        else:
            file_index_list = np.arange(self.num_files)

        index_array = file_index_list.tolist()
        return index_array
    
    # function: 

    def load_file(self, filename, segmentation_load = False):
        ii = nb.load(filename).get_fdata()
        if segmentation_load is True:
            ii = np.round(ii).astype(int)
        
        return ii
    

    # function: get each item using the index [file_index, slice_index]
    def __getitem__(self, index):
        f = self.index_array[index]
    
        image_filename = self.image_file_list[f]
        seg_filename = self.seg_file_list[f]

        if os.path.isfile(seg_filename) is False:
            self.have_manual_seg = False
        else:
            self.have_manual_seg = True
            
        # if it's a new case, then do the data loading; if it's not, then just use the current data
        if image_filename != self.current_image_file or seg_filename != self.current_seg_file:
            image_loaded = self.load_file(image_filename, segmentation_load = False) 
          
            if self.have_manual_seg is True:
                seg_loaded = self.load_file(seg_filename, segmentation_load=True) 

            self.original_shape = image_loaded.shape
            h,w = image_loaded.shape[0], image_loaded.shape[1]
    
            # now we need to do the center crop for both image and seg
            if self.have_manual_seg is True: # center crop regarding the manual segmentation
              
                _,_, self.centroid = Data_processing.center_crop( image_loaded, seg_loaded, self.image_shape, according_to_which_class = self.center_crop_according_to_which_class, centroid = None)
                # random crop (randomly shift the centroid)
                if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
                    random_centriod_shift_x = np.random.randint(-5,5)
                    random_centriod_shift_y = np.random.randint(-5,5)
                    centroid_used_for_crop = [self.centroid[0] + random_centriod_shift_x, self.centroid[1] + random_centriod_shift_y]
                else:
                    centroid_used_for_crop = self.centroid
                # then for each dim in slice_num and tf, we do the center crop
                image_loaded_tem, seg_loaded_tem, _ = Data_processing.center_crop( image_loaded, seg_loaded, self.image_shape, according_to_which_class = None , centroid = centroid_used_for_crop)
                image_loaded = np.copy(image_loaded_tem)
                seg_loaded = np.copy(seg_loaded_tem)

            elif self.have_manual_seg is False:
                # crop the image regarding the image center, just find the center and take 128x128 ROI
                self.centroid = np.array([h//2, w//2])
                image_loaded_tem = np.copy(image_loaded[self.centroid[0]-self.image_shape[0]//2:self.centroid[0]+self.image_shape[0]//2, self.centroid[1]-self.image_shape[1]//2:self.centroid[1]+self.image_shape[1]//2,:,:])
                image_loaded = np.copy(image_loaded_tem)
                seg_loaded = np.zeros(image_loaded.shape) # segmentation is all zeros

            self.current_image_file = image_filename
            self.current_image_data = np.copy(image_loaded)  
            self.current_seg_file = seg_filename
            self.current_seg_data = np.copy(seg_loaded)
        
        # pick the slice
        original_image = np.copy(self.current_image_data)
        original_seg = np.copy(self.current_seg_data)
      
        ######## do augmentation
        processed_seg = np.copy(original_seg)
        # (0) add noise
        if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
            standard_deviation = 5
            processed_image = original_image + np.random.normal(0,standard_deviation,original_image.shape)
            # turn the image pixel range to [0,255]
            processed_image = Data_processing.turn_image_range_into_0_255(processed_image)
        else:
            processed_image = Data_processing.turn_image_range_into_0_255(original_image)
       
        # (1) do brightness
        if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
            processed_image,v = random_aug.random_brightness(processed_image, v = None)
    
        # (2) do contrast
        if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
            processed_image, v = random_aug.random_contrast(processed_image, v = None)

        # (3) do sharpness
        if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
            processed_image, v = random_aug.random_sharpness(processed_image, v = None)
            
        # (4) do flip
        if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
            # doing this can make sure the flip is the same for image and seg
            a, selected_option = random_aug.random_flip(processed_image)
            b,_ = random_aug.random_flip(processed_seg, selected_option)
            processed_image = np.copy(a)
            processed_seg = np.copy(b)

        # (5) do rotate
        if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
            processed_image, z_rotate_degree = random_aug.random_rotate(processed_image, order = 1, z_rotate_range = [-10,10])
            processed_seg,_ = random_aug.random_rotate(processed_seg, z_rotate_degree, fill_val = 0, order = 0)

        # (6) do translate
        if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
            processed_image, x_translate, y_translate = random_aug.random_translate(processed_image, translate_range = [-10,10])
            processed_seg,_ ,_= random_aug.random_translate(processed_seg, x_translate, y_translate)

        # add normalization
        if self.image_normalization is True:
            processed_image = Data_processing.normalize_image(processed_image,inverse = False) 


        # also add infos from patient list spread sheet
        row = self.patient_list_spreadsheet.loc[self.patient_list_spreadsheet['img_file'] == image_filename]
        
        # now it's time to turn numpy into tensor and collect as a dictionary (this is the final return)
        processed_image_torch = torch.from_numpy(processed_image).unsqueeze(0).float() 
        processed_seg_torch = torch.from_numpy(processed_seg).unsqueeze(0)  

        # also need to return the original image and seg without the augmentation (with center crop done)
        original_image_torch = torch.from_numpy(original_image).unsqueeze(0).float()
        original_seg_torch = torch.from_numpy(original_seg).unsqueeze(0).float()

        final_dictionary = { "image": processed_image_torch, 
                            "mask": processed_seg_torch, 
                            "original_image": original_image_torch,  
                            "original_seg": original_seg_torch,
                            
                            "image_file_name" : image_filename, "seg_file_name": seg_filename,
                            "original_shape" : self.original_shape,
                            "centroid": self.centroid,
                          
                            # copy infos from patient list spreadsheet
                            "patient_id": row.iloc[0]['patient_id'],
                            "img_file": row.iloc[0]['img_file'],
                            "seg_file": row.iloc[0]['seg_file'],
                            "slice_index": row.iloc[0]['slice_index'],
                            }

        if self.return_arrays_or_dictionary == 'dictionary':
            return final_dictionary
        elif self.return_arrays_or_dictionary == 'arrays':
            return processed_image_torch, processed_seg_torch # model input and label
        else:
            raise ValueError('return_arrays_or_dictionary should be "arrays" or "dictionary"')
    
    # function: at the end of each epoch, we need to reset the index array
    def on_epoch_end(self):
        print('now run on_epoch_end function')
        self.index_array = self.generate_index_array()

        self.current_image_file = None
        self.current_image_data = None 
        self.current_seg_file = None
        self.current_seg_data = None