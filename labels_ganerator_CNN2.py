# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 19:23:51 2021

@author: Hanan
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 10:53:22 2021

@author: Hanan
"""

import os

label_dict={
            '001': 'DCM','002': 'DCM','003': 'DCM','004': 'DCM','005': 'DCM',
            '006': 'DCM','007': 'DCM','008': 'DCM','009': 'DCM','010': 'DCM',
            '011': 'DCM','012': 'DCM','013': 'DCM','014': 'DCM','015': 'DCM',
            '016': 'DCM','017': 'DCM','018': 'DCM','019': 'DCM','020': 'DCM',
            '021': 'HCM','022': 'HCM', '023': 'HCM','024': 'HCM', '025': 'HCM',
            '026': 'HCM', '027': 'HCM','028': 'HCM', '029': 'HCM','030': 'HCM',
            '031': 'HCM', '032': 'HCM','033': 'HCM', '034': 'HCM','035': 'HCM',
            '036': 'HCM', '037': 'HCM','038': 'HCM', '039': 'HCM','040': 'HCM',
        	'041': 'MINF', '042': 'MINF','047': 'MINF', '048': 'MINF','049': 'MINF',
            '050': 'MINF'
            }

classes = {
    'DCM' : 0,
    'HCM' : 1,
    'MINF' : 2,
    'ARV' : 3,
    'NOR' : 4
    }

# training

path = 'hdf5_files_CNN2/train_set'
k = os.listdir(path)
a = []
b = []


for p in os.listdir(path):
    if p=='desktop.ini':
        pass
    else:
        a.append(p)
        b.append(p.split('_')[1])
        
labels_CNN2 = []
    
for p in os.listdir(path):
   if p=='desktop.ini':
      pass
   else:
      id_ = p.split('_')[1]
      class_ = label_dict[id_]
      labels_CNN2.append(classes[class_])
# validation set
    
path = 'hdf5_files_CNN2/validation_set'
k = os.listdir(path)
a = []
b = []

for p in os.listdir(path):
    if p=='desktop.ini':
        pass
    else:
        a.append(p)
        b.append(p.split('_')[1])
        
labels_CNN2_val = []
    
for p in os.listdir(path):
   if p=='desktop.ini':
      pass
   else:
      id_ = p.split('_')[1]
      class_ = label_dict[id_]
      labels_CNN2_val.append(classes[class_])
    
# Test Set

path = 'hdf5_files_CNN2/test_set'
k = os.listdir(path)
a = []
b = []

for p in os.listdir(path):  
    if p=='desktop.ini':
        pass
    else:
        a.append(p)
        b.append(p.split('_')[1])
    
labels_CNN2_test = []

for p in os.listdir(path):
    if p=='desktop.ini':
        pass
    else:
        id_ = p.split('_')[1]
        class_ = label_dict[id_]
        labels_CNN2_test.append(classes[class_])