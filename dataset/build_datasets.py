#!/usr/bin/env python
# build dataloader for CMR SAX slices, you can build your own

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import original_SAM.dataset.CMR.Dataset as Dataset

def from_excel_file(file, index_list = None): 
    data = pd.read_excel(file)
    if index_list == None:
        c = data
    else:
        c = data.iloc[index_list]

    patient_id_list = np.asarray(c['patient_id'])
    img_file_list = np.asarray(c['img_file'])
    seg_file_list = np.asarray(c['seg_file'])
    slice_index_list = np.asarray(c['slice_index'])

    return patient_id_list, img_file_list, seg_file_list, slice_index_list
       

def build_dataset(args,  patient_list_file, index_list, shuffle = False, augment = False):

    _, img_file_list, seg_file_list, _ = from_excel_file(patient_list_file, index_list)
 
    dataset = Dataset.Dataset_CMR(
                                  patient_list_file,
                                  image_file_list = img_file_list,
                                  seg_file_list = seg_file_list,

                                  return_arrays_or_dictionary = 'dictionary',
                                  center_crop_according_to_which_class = 1,

                                  image_shape = [args.img_size,args.img_size],
                                  image_normalization = True,
                                  shuffle = shuffle, 
                                  augment = augment,
                                  augment_frequency = args.augment_frequency)
    return dataset

