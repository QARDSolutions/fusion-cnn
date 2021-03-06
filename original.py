# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 16:31:16 2021

@author: Hanan
"""
import numpy as np
import os, sys, shutil, time, re
import h5py
import skimage.morphology as morph
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import pickle
# For ROI extraction
import skimage.transform
from scipy.fftpack import fftn, ifftn
from skimage.feature import peak_local_max, canny
from skimage.transform import hough_circle 
# Nifti processing
import nibabel as nib
from collections import OrderedDict
# print sys.path
# sys.path.append("..") 
import errno
np.random.seed(42)


def generate_train_validate_test_set(src_path, dest_path):
  """
  Split the data into 70:15:15 for train-validate-test set
  arg: path: input data path
  """
  SPLIT_TRAIN = 0.7
  SPLIT_VALID = 0.15

  dest_path = os.path.join(dest_path,'dataset')
  if os.path.exists(dest_path):
    shutil.rmtree(dest_path)
  os.makedirs(os.path.join(dest_path, 'train_set'))  
  os.makedirs(os.path.join(dest_path, 'validation_set'))  
  os.makedirs(os.path.join(dest_path, 'test_set'))  
  # print (src_path)
  groups = next(os.walk(src_path))[1]
  for group in groups:
    group_path = next(os.walk(os.path.join(src_path, group)))[0]
    patient_folders = next(os.walk(group_path))[1]
    np.random.shuffle(patient_folders)
    train_ = patient_folders[0:int(SPLIT_TRAIN*len(patient_folders))]
    valid_ = patient_folders[int(SPLIT_TRAIN*len(patient_folders)): 
                 int((SPLIT_TRAIN+SPLIT_VALID)*len(patient_folders))]
    test_ = patient_folders[int((SPLIT_TRAIN+SPLIT_VALID)*len(patient_folders)):]
    for patient in train_:
      folder_path = os.path.join(group_path, patient)
      copy(folder_path, os.path.join(dest_path, 'train_set', patient))

    for patient in valid_:
      folder_path = os.path.join(group_path, patient)
      copy(folder_path, os.path.join(dest_path, 'validation_set', patient))

    for patient in test_:
      folder_path = os.path.join(group_path, patient)
      copy(folder_path, os.path.join(dest_path, 'test_set', patient))
