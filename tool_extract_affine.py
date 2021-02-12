#!/usr/bin/env python

## this script is used for extracting the affine matrix and thus the translation vector + directional vector for cardiac imaging planes 
## this script can also save a "padding_coordinate_conversion.npy" to show the math relation of coordinate system before the padding/cropping and that after the padding/cropping.

import os
import numpy as np
import glob
import nibabel as nib
from nibabel.affines import apply_affine
import supplements
import math
import sympy as sym
import function_list as ff

np.set_printoptions(precision=5,suppress=True)
cg = supplements.Experiment()

#center point method. easy method
def calculate_conversion(shape):
    volume_center=np.array([int((shape[0]-1)/2),int((shape[1]-1)/2),int((shape[-1]-1)/2)])
    delta=np.array([int((cg.dim[0]-1)/2),int((cg.dim[1]-1)/2),int((cg.dim[-1]-1)/2)])-volume_center
    return delta

o = [0,0,0]
x1 = [1,0,0]
y1 = [0,1,0]
z1 = [0,0,1]

# main function
def get_vectors(i,j,i_affine,j_affine): #i=image,j=imaging planes
    x=nib.load(i)
    y=nib.load(j)
    s1=x.shape
    s2=y.shape
    A = np.linalg.inv(i_affine).dot(j_affine)

    # translation of center
    mpr_center=np.array([(s2[0]-1)/2,(s2[1]-1)/2,0])
    img_center=np.array([(s1[0]-1)/2,(s1[1]-1)/2,(s1[-1]-1)/2])
    mpr_center_img = ff.convert_coordinates(i_affine,j_affine,mpr_center)
    translation_c = mpr_center_img - img_center
        
    # normalize translation into padding coordinate system
    x_pad = ut.in_adapt(i)
    p_size = x_pad.shape
    translation_c_n = np.array([translation_c[0]/p_size[0]*2,translation_c[1]/p_size[1]*2,translation_c[2]/p_size[2]*2])

    
    # translation of origin
    translation_o = ff.convert_coordinates(i_affine,j_affine,[0,0,0]) - [0,0,0]
    translation_o_n = np.array([translation_o[0]/p_size[0]*2,translation_o[1]/p_size[1]*2,translation_o[2]/p_size[2]*2])


    # x_direction 
    x_d = ff.convert_coordinates(i_affine,j_affine,x1) - ff.convert_coordinates(i_affine,j_affine,o)
    x_scale = np.linalg.norm(x_d)
    x_n = np.asarray([i/x_scale for i in x_d])
        
        
    # y_direction
    y_d = ff.convert_coordinates(i_affine,j_affine,y1) - ff.convert_coordinates(i_affine,j_affine,o)
    y_scale = np.linalg.norm(y_d)
    y_n = np.asarray([i/y_scale for i in y_d])

    # z_direction
    z_d = ff.convert_coordinates(i_affine,j_affine,z1) - ff.convert_coordinates(i_affine,j_affine,o)
    z_scale = np.linalg.norm(z_d)
    z_n = np.asarray([i/z_scale for i in z_d])

    # scale
    x_s = ff.length(x_d)/1
    y_s = ff.length(y_d)/1
    z_s = ff.length(z_d)/1
    scale = np.array([x_s,y_s,z_s])

 
    # put all into a nparray
    vectors=np.array([0,0,0, translation_o, translation_o_n, x_d, x_n, y_d, y_n, z_d, z_n, scale,translation_c,translation_c_n,img_center])
    return vectors

chamber_list = ['2C','3C','4C','BASAL']
chamber_choice = [0]
patient_list = ff.find_all_target_files(['ucsd_*/*'],cg.data_dir)
img_fld = 'img-nii-sm'
mpr_fld = 'mpr-new-nii-sm-1.5'
save_fld = 'affine_standard'
suffix = '_MR'

for p in patient_list:
    patient_id = os.path.basename(p)
    patient_class = os.path.basename(os.path.dirname(p))
    print(patient_class,patient_id)

    # get affine matrix for planes
    for c in chamber_choice:
        chamber = chamber_list[c]

        # define save file
        save_dir=os.path.join(cg.data_dir,patient_class,patient_id,save_fld)
        save_file = os.path.join(save_dir,chamber+suffix)

        if os.path.exists(os.path.join(save_dir,chamber+suffix+'.npy')):
            print('already get the matrix for ', chamber)
            continue

        #image 1.5mm
        i = os.path.join(p,img_fld,'0.nii.gz')
        i_affine = ff.check_affine(i)
    
        #mpr 1.5mm
        j = os.path.join(p,mpr_fld,chamber,'0.nii.gz')
        j_affine = nib.load(j).affine
        
        matrix=get_vectors(i,j,i_affine,j_affine)
        
        #save matrix

        os.makedirs(save_dir,exist_ok = True)
        np.save(save_file,matrix)

    # get padding_coordinate_conversion.npy
    image_file = os.path.join(p,'img-nii-sm','0.nii.gz')
    save_file_padding= os.path.join(save_dir,'padding_coordinate_conversion.npy')
    if os.path.exists(save_file_padding):
        print('already done this patient')
        continue

    volume=nib.load(image_file)
    volume_data=volume.get_fdata()
    shape=volume_data.shape
    delta=calculate_conversion(shape)
    np.save(save_file_padding,delta)

       