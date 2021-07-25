# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 23:20:25 2020

@author: jsc
"""
import os,shutil
#from nipype.interfaces.dcm2nii import Dcm2niix
import dicom2nifti

def transform(input_path, out_path, name):
    converter = Dcm2niix()
    converter.inputs.source_dir = input_path
    converter.inputs.compression = 5
    converter.inputs.out_filename = name
    converter.inputs.output_dir = out_path
    # converter.inputs.merge_imgs = True
    converter.cmdline
    converter.run()

if __name__=="__main__":
    input_path = "D:/work/research/01_SCD/data_raw/MRI_SCD"
#    os.mkdir(input_path+"temp")
    paths = os.listdir(input_path)
    for p in paths:
        pp = os.path.join(input_path,p)
        in_ps = os.listdir(pp)
        for child in in_ps:
            if(child.endswith(".zip")):
                os.remove(os.path.join(pp,child))
#            transform(pp, pp, child)
            dicom2nifti.dicom_series_to_nifti(pp, pp+"/"+p, reorient_nifti=True)
            break
                
                
    