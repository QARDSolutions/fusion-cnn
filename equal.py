# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 21:40:04 2021

@author: Hanan
"""

import os

path1 = 'hdf5_files_CNN1/test_set'
path2 = 'hdf5_files_CNN2/test_set'
CNN1 = os.listdir(path1) # cnn1
CNN2 = os.listdir(path2) # cnn2

one_not_two = [ x for x in CNN1 if x not in CNN2 ]
two_not_one = [ x for x in CNN2 if x not in CNN1 ]

Final = one_not_two + two_not_one

print (len(Final))

for i in one_not_two:
    path = os.path.join(path1,i)
    os.remove(path)
    
for i in two_not_one:
    path = os.path.join(path2,i)
    os.remove(path)
    
CNN1 = os.listdir('hdf5_files_CNN1/test_set')
CNN2 = os.listdir('hdf5_files_CNN2/test_set')