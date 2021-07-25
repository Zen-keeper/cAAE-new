# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:58:01 2020

@author: jsc
"""
import os

path = "D:/python_file/cAAE-master/cAAE-master/Results/c5_2/NC_ORI"
igs = os.listdir(path)
list1 = [img.split("_")[-2] for img in igs]
d = "D:/python_file/cAAE-master/cAAE-master/Results/c5_2/NC_gen"
ps = os.listdir(d)
for i in ps:
    if(i.split("_")[-2]  not in list1):
        os.remove(os.path.join(d,i))