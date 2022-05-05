#!/usr/bin/env python

# this script uses the predicted plane vectors to generates a cine movie of cardiac imaging planes
# including three LAX planes + A SAX stack (9 planes from mitral valve plane to LV apex) 
# see an example of this movie in Example_plane_cine.mp4 

import function_list as ff
import os
import math
import numpy as np
import nibabel as nib 
import supplements
from PIL import Image
from Make_plane_movies import *
cg = supplements.Experiment()

WL = 500
WW = 800  

######### Define patient_list
main_folder = cg.predict_dir
patient_list = ff.find_all_target_files(['ucsd_ccta/CVC*'],main_folder)

######## Define folders
seg_folder = 'seg-pred-try' # where you save predicted segmentation
vector_folder = 'vector-pred-try'  # where you save predicted plane vectors



# main script (usually no need to change):
for patient in patient_list:
    patient_id = os.path.basename(patient)
    patient_class = os.path.basename(os.path.dirname(patient))
    print(patient_class,patient_id)

    save_folder = os.path.join(patient,'planes_pred')
    ff.make_folder([os.path.dirname(save_folder),save_folder, os.path.join(save_folder,'pngs')])

    # check whether already done
    if os.path.isfile(os.path.join(save_folder,patient_id+'_planes.mp4')) == 1:
        print('already done for this patient')
        continue
        
    # load (manual/predicted) segmentation (you are going to use segmentation to determine the range of LV, thus determine the range of SAX stack)
    seg = nib.load(os.path.join(patient,seg_folder,'pred_s_0.nii.gz'))
    seg_LV = seg.get_fdata()

    # load plane vectors
    prepare = Prepare(main_folder,patient_class,patient_id, cg.spacing, [cg.dim[0],cg.dim[1],1], os.path.join(patient,'img-nii-sm'))
    image_center, vector_2C, vector_3C, vector_4C, vector_SA, normal_vector_SA = prepare.load_plane_vectors(vector_folder)

    # get affine matrices
    volume_affine, _,_,_ = prepare.obtain_affine_matrix(vector_2C, vector_3C, vector_4C)
        
    # define the range of SAX stack using the LV segmentation
    # we will obtain two numbers "a" and "b", which means the LV SAX stack should start "a" sclices ahead of the predicted BASAL plane and ends "b" slices after the BASAL plane
    a , b = prepare.define_SAX_range(vector_SA,image_center, seg_LV,  False)

    # get a center list of 9-plane SAX stack
    _, center_list9, gap = prepare.define_SAX_planes_center_list(vector_SA, image_center, a, b, normal_vector_SA)
    if gap < 1:
        print('predicted LV segmentation is failed')
        continue

    # get the image list
    img_list = ff.sort_timeframe(ff.find_all_target_files(['img-nii-sm/*.nii.gz'],patient),2)

    # make the plane images for each time frame
    for img in img_list:
        volume_data = nib.load(img).get_fdata()
        time = ff.find_timeframe(img,2)
        save_path = os.path.join(save_folder,'pngs',str(time) +'.png')
        make = Make_Planes(main_folder, patient_class, patient_id, volume_data,[WL,WW], image_center,[vector_2C, vector_3C, vector_4C, vector_SA], volume_affine, center_list9 ,cg.spacing, [cg.dim[0],cg.dim[1], 1], zoom_factor_LAX = 1.2, zoom_factor_SAX = 1.4)
        make.plane_image(save_path, draw_lines = False) # draw_lines means use lines to represent SAX/LAX planes on LAX/SAX planes
        print('finish time frame ',time)

    # make cine movie
    pngs = ff.sort_timeframe(ff.find_all_target_files(['*.png'],os.path.join(save_folder,'pngs')),1)
    save_movie_path = os.path.join(save_folder,patient_id+'_planes.mp4')
    ff.make_movies(save_movie_path,pngs)

    





    
    