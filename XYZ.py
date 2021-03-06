# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 00:47:39 2021

@author: Hanan
"""

import os
import numpy as np
import nibabel as nib
import h5py
from preprocess import ExtractProcessedData
from original import generate_train_validate_test_set
from config import conf

example_filename = 'patient047_4d.nii.gz' # file path
img = nib.load(example_filename)
print(img.shape)

#data = h5py.File('P_050_ES_09_MINF.hdf5','r')
path1='P_050_ES_09_MINF.hdf5'
conf = conf()
patch_img, patch_gt, patch_wmap, file_name = ExtractProcessedData(path1, mode='train', 
                                                                  transformation_params=conf.transformation_params)

path2='patient050_frame12_gt.nii.gz'
dest_path='/ok'
generate_train_validate_test_set(path2,dest_path)
