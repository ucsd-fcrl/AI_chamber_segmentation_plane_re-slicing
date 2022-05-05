#!/usr/bin/env python

## this script obtains plane vectors (translation vector + directional vector) used to generate cardiac imaging planes via reading the affine matrix
## this script can also save a "padding_coordinate_conversion.npy" to show the math relation of coordinate system before and after the padding/cropping.

import os
import numpy as np
import nibabel as nib
import supplements
import function_list as ff
from Extract_vector_from_affines import *

np.set_printoptions(precision=5,suppress=True)
cg = supplements.Experiment()


##### Define patient_list
main_folder = os.path.join(cg.main_dir,'CNN/all-classes-all-phases-1.5')
patient_list = ff.find_all_target_files(['ucsd_ccta/CVC*'],main_folder)

##### Define folders
img_fld = 'img-nii-sm'
mpr_fld = 'mpr-new-nii-sm-1.5'
save_fld = 'affine_standard'

##### Define Plane views
plane_list = ['2C','3C','4C','BASAL']
plane_choice = [0,1,2]



# main script (usually no need to change)
for p in patient_list:
    patient_id = os.path.basename(p)
    patient_class = os.path.basename(os.path.dirname(p))
    print(patient_class,patient_id)

    # extract plane vectors
    vector = Vector(main_folder,patient_class,patient_id,plane_list,plane_choice)
    vector.save_vectors(img_fld,mpr_fld,save_fld)


    # get padding_coordinate_conversion.npy
    vector.save_padding_coordinate_conversion(save_fld)

       