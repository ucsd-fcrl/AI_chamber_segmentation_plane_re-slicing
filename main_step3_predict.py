#!/usr/bin/env python

# this script can predict chamber segmentation + translation&direction vectors by pre-trained model
# You can define your case list as well as task list in the script.
# Run python main_step3_predict in the terminal

# System
import argparse
import os

# Third Party
import numpy as np
import math
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.models import Model
from keras.layers import Input, \
                         Conv1D, Conv2D, Conv3D, \
                         MaxPooling1D, MaxPooling2D, MaxPooling3D, \
                         UpSampling1D, UpSampling2D, UpSampling3D, \
                         Reshape, Flatten, Dense
from keras.initializers import Orthogonal
from keras.regularizers import l2
import nibabel as nb

import supplements
import supplements.utils as ut
import dvpy as dv
import dvpy.tf
import function_list as ff

cg = supplements.Experiment()


# Define pre-trained model list (segmentation model and vector models for different planes)
# model_s stands for segmentation model
# model_2C_t stands for model to predict translation (t) vector for 2 chamber (2C) plane
# model_2C_r stands for model to predict direction (r) vector for 2 chamber plane
[model_s, model_2C_t, model_2C_r, model_3C_t, model_3C_r, model_4C_t, model_4C_r, model_BASAL_t, model_BASAL_r] = ['*batch0/model-batch0_s-040-*', 
   '*batch0/2C_t/model-batch0_2C_t-001-*',  # pick the one with lowest validation loss
   '*batch0/2C_r/model-batch0_2C_r-001-*',
   '*batch0/3C_t/model-batch0_3C_t-001-*',
   '*batch0/3C_r/model-batch0_3C_r-001-*',
   '*batch0/4C_t/model-batch0_4C_t-001-*',
   '*batch0/4C_r/model-batch0_4C_r-001-*',
   '*batch0/BASAL_t/model-batch0_BASAL_t-001-*',
   '*batch0/BASAL_r/model-batch0_BASAL_r-001-*']
MODEL = [model_s, model_2C_t, model_2C_r, model_3C_t, model_3C_r, model_4C_t, model_4C_r, model_BASAL_t, model_BASAL_r]

# Define the prediction tasks you want to do
# prediction task list
task_list = ['s','2C_t','2C_r','3C_t','3C_r','4C_t','4C_r','BASAL_t','BASAL_r'] 
task_num_list = [0,3]

# define patient CT image list
patient_list = ff.find_all_target_files(['ucsd_ccta/*'],cg.data_dir)
print(patient_list.shape)
print('finish finding all patients')

# build the model
shape = cg.dim + (1,)
model_inputs = [Input(shape)]
model_outputs=[]
ds_layer, _, unet_output = dvpy.tf.get_unet(cg.dim,cg.num_classes,cg.conv_depth,layer_name='unet',
                                      dimension =len(cg.dim),unet_depth = cg.unet_depth,)(model_inputs[0])
ds_flat = Flatten()(ds_layer)
Loc= Dense(384,kernel_initializer=Orthogonal(gain=1.0),kernel_regularizer = l2(1e-4),
                                      activation='relu',name='Loc')(ds_flat)
translation = Dense(3,kernel_initializer=Orthogonal(gain=1e-1),kernel_regularizer = l2(1e-4),
                                      name ='t')(Loc)
x_direction = Dense(3,kernel_initializer=Orthogonal(gain=1e-1),kernel_regularizer = l2(1e-4),
                                      name ='x')(Loc)
y_direction = Dense(3,kernel_initializer=Orthogonal(gain=1e-1),kernel_regularizer = l2(1e-4),
                                      name ='y')(Loc)
model_outputs += [unet_output]
model_outputs += [translation]
model_outputs += [x_direction]
model_outputs += [y_direction]
model = Model(inputs = model_inputs,outputs = model_outputs)

# define the generator
valgen = dv.tf.ImageDataGenerator(3,input_layer_names=['input_1'],output_layer_names=['unet','t','x','y'],)
print('finish building the model')


# predict per patient
for task_num in task_num_list:
  print('current task is: ', task_list[task_num])
  if task_list[task_num] == 's':
      view = '2C'
      print(view)
  else:
    view = task_list[task_num].split('_')[0]
    vector = task_list[task_num].split('_')[1]
    print(view)
    ##### load saved weight
  model_file_name = MODEL[task_num]
  model_files = ff.find_all_target_files([model_file_name],cg.model_dir)
  assert len(model_files) == 1
  model.load_weights(model_files[0],by_name = True)
  print('=======\n=======\n')
  print('finish loading saved weights: ',model_files[0])

    # predict
  for p in patient_list:
    patient_class = os.path.basename(os.path.dirname(p))
    patient_id = os.path.basename(p)
    print(patient_class, patient_id)

    # find the input images for all time frames:
    if task_list[task_num] == 's':
      img_list = ff.sort_timeframe(ff.find_all_target_files(['img-nii-sm/*.nii.gz'],p),2)
    else:
      img_list = ff.find_all_target_files(['img-nii-sm/0.nii.gz'],p)

    for img in img_list:
      time_frame = ff.find_timeframe(img,2)
      print(time_frame)

    # build the predict_generator
      u_pred,t_pred,x_pred,y_pred= model.predict_generator(valgen.predict_flow(np.asarray([img]),
            batch_size = cg.batch_size,
            view = view,
            relabel_LVOT = False,
            input_adapter = ut.in_adapt,
            output_adapter = ut.out_adapt,
            shape = cg.dim,
            input_channels = 1,
            output_channels = cg.num_classes,),
            verbose = 1,
            steps = 1,)

      # save u_net segmentation
      if task_list[task_num] == 's':
        u_gt_nii = nb.load(img)
        u_pred = np.argmax(u_pred[0], axis = -1).astype(np.uint8)
        u_pred = dv.crop_or_pad(u_pred, u_gt_nii.get_data().shape)
        u_pred = nb.Nifti1Image(u_pred, u_gt_nii.affine)
        save_path = os.path.join(cg.predict_dir,patient_class,patient_id,'seg-pred','pred_'+task_list[task_num]+'_'+os.path.basename(img))
        ff.make_folder([os.path.dirname(os.path.dirname(os.path.dirname(save_path))), os.path.dirname(os.path.dirname(save_path)), os.path.dirname(save_path)])
        nb.save(u_pred, save_path)
        
        # save vectors
      if task_list[task_num] != 's':
        print(x_pred)
        x_n = ff.normalize(x_pred)
        y_n = ff.normalize(y_pred)
        matrix = np.concatenate((t_pred.reshape(1,3),x_n.reshape(1,3),y_n.reshape(1,3)))
        save_path = os.path.join(cg.predict_dir,patient_class,patient_id,'vector-pred','pred_'+task_list[task_num])
        ff.make_folder([os.path.dirname(os.path.dirname(os.path.dirname(save_path))), os.path.dirname(os.path.dirname(save_path)), os.path.dirname(save_path)])
        np.save(save_path,matrix)
