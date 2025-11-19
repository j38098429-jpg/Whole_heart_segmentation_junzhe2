import numpy as np
import os
import pandas as pd


class Build():
    def __init__(self,file_list):
        self.file_list = file_list
        self.data = pd.read_excel(file_list, dtype = {'Patient_ID': str})

    def __build__(self,batch_list):
        for b in range(len(batch_list)):
            cases = self.data.loc[self.data['batch'] == batch_list[b]]
            if b == 0:
                c = cases.copy()
            else:
                c = pd.concat([c,cases])

        batch_list = np.asarray(c['batch'])
        patient_id_list = np.asarray(c['case_id'])
        slice_index_list = np.asarray(c['slice_index'])
        img_file_list = np.asarray(c['img_file'])
        seg_file_list = np.asarray(c['seg_file'])
        
        return batch_list, patient_id_list, slice_index_list, img_file_list, seg_file_list