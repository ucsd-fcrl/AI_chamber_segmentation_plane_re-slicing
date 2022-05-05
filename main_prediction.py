#!/usr/bin/env python

# this script uses trained deep learning (DL) model to predict:
# (1) chamber segmentation (LV+LA) 
# (2) vectors (translation vector t + directional vectors r) to reslice the cardiac imaging planes 

import os
import numpy as np
import nibabel as nb
import supplements
import supplements.utils as ut
import function_list as ff
from Build_model import *
cg = supplements.Experiment()


######### in total we have 9 tasks:
# chamber segmentation + predict (1) translation vecotr "t" and (2) directional vector "r" for 4 planes (2CH, 3CH, 4CH and SAX): 1 + 4 x 2 = 9 tasks
task_list = ['s','2C_t','2C_r','3C_t','3C_r','4C_t','4C_r','BASAL_t','BASAL_r'] 
# define which tasks you want to do:
task_num_list = [0,1,2,3,4,5,6,7,8]  # want to do all of them

######### Define patient list
patient_list = ff.find_all_target_files(['ucsd_ccta/CVC*'],cg.predict_dir)

######### Define save folder
save_folder = cg.predict_dir
save_folder_seg = 'seg-pred-try'
save_folder_vector = 'vector-pred-try'



# Main script (usually no need to change):

# build models
build_model = Build_Model()
model = build_model.build_blank_model # model architecture (no model compile and weight loading)
MODEL_list = build_model.model_list # trained model list (you need to update with your own trained model list)

# do tasks one by one
for task_num in task_num_list:
  print('current task is: ', task_list[task_num])

  [view,vector] = build_model.generator_parameters(task_list[task_num])
 
  # load saved weights
  model_files = ff.find_all_target_files([MODEL_list[task_num][0]],cg.model_dir)
  assert len(model_files) == 1
  print('finish loading saved weights: ',model_files[0])
  model.load_weights(model_files[0],by_name = True)

  # predict patietns one by one
  for p in patient_list:
    patient_class = os.path.basename(os.path.dirname(p))
    patient_id = os.path.basename(p)
    print(patient_class, patient_id)
      
    # if already done:
    if task_list[task_num] == 's':
      if os.path.isfile(os.path.join(save_folder,patient_class,patient_id,save_folder_seg,'pred_s_0.nii.gz')) == 1:
        print('already done segmentation')
        continue
    else:
      if os.path.isfile(os.path.join(save_folder,patient_class,patient_id,save_folder_vector,'pred_'+task_list[task_num]+'.npy')) == 1:
        print('already done ', task_list[task_num])
        continue

    # find the input images for time frames:
    if task_list[task_num] == 's':
      img_list = ff.sort_timeframe(ff.find_all_target_files(['img-nii-sm/*.nii.gz'],p),2) # predict segmentation for all time frames
    else:
      img_list = ff.find_all_target_files(['img-nii-sm/0.nii.gz'],p) # only need one time frame to predict planes


    for img in img_list:
      # predict:
      valgen = build_model.image_generator()
      seg_pred,t_pred,x_pred,y_pred= model.predict_generator(valgen.predict_flow(np.asarray([img]),
            batch_size = 1,
            view = view,
            relabel_LVOT = True, # default - True
            input_adapter = ut.in_adapt,
            output_adapter = ut.out_adapt,
            shape = cg.dim,
            input_channels = 1,
            output_channels = cg.num_classes,),
            verbose = 1,
            steps = 1,)

      # save u_net segmentation
      if task_list[task_num] == 's':
        save_path = os.path.join(save_folder,patient_class,patient_id,save_folder_seg, 'pred_s_'+os.path.basename(img))
        ff.make_folder([os.path.join(save_folder,patient_class), os.path.join(save_folder,patient_class,patient_id), os.path.dirname(save_path)])
        build_model.save_segmentation(img,seg_pred,save_path)
      
      # save vectors
      if task_list[task_num] != 's':
        save_path = os.path.join(save_folder,patient_class,patient_id, save_folder_vector, 'pred_'+task_list[task_num])
        ff.make_folder([os.path.join(save_folder,patient_class), os.path.join(save_folder,patient_class,patient_id), os.path.dirname(save_path)])
        build_model.save_vector(t_pred,x_pred,y_pred,save_path)