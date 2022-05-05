#!/usr/bin/env python

# this script trains the AI to predict directional vector/translational vector. 
# The AI model's weights are initialized by the pre-trained segmentation model.
# in terminal, type ./main_step2B_train_vectors--batch N to run. again, N here is for cross-validation.

# System
import argparse
import os

# Third Party
import numpy as np
import tensorflow as tf

# Internal
import supplements
import supplements.utils as ut
import dvpy as dv
import dvpy.tf
import function_list as ff
from Build_model import *

cg = supplements.Experiment()

##### Define which vector you are going to train the model on
# in total there are 8 choices from 4 plnaes (2C, 3C, 4C and BASAL), each of which has 2 kinds of vectors (translation vector "t" and directional vector "r")
# 8 = 4 x 2
view_list = ['2C', '3C', '4C', 'BASAL']
vector_list = ['t','r']

which_view = 0
which_vector = 1

##### define the filepath of your pre-trained model (model trained on segmentation with the highest validation accuracy)
epoch = '035' # you just need to point out which epoch has the highest validation accuracy




# main script (usually no need to change)
def get_pre_trained_model(batch,epoch): 
  if batch == None:
    pre_trained_model = 'model-batchall_s-' + epoch +'*'
    pre_model_path = ff.find_all_target_files([pre_trained_model],os.path.join(cg.model_dir,'model_batchall/s'))
  else:
    pre_trained_model = 'model-batch' + str(batch) +'_s-' + epoch +'*'
    pre_model_path = ff.find_all_target_files([pre_trained_model],os.path.join(cg.model_dir,'model_batch'+str(batch),'s'))
  assert len(pre_model_path) == 1
  
  print('the pre-trained model is: ', pre_model_path[0])
  return pre_model_path[0]
  

def retrain(batch,pre_trained_model):
  # define the task
  vector = vector_list[which_vector] 
  view = view_list[which_view] 
  print('view is: ',view,'vector is',vector)
  current_task = view+'_'+vector

  #===========================================
  dv.section_print('Calculating Image Lists...')

  imgs_list_trn=[np.load(os.path.join(cg.model_dir,'partitions_dir/img_list_'+str(p)+'.npy'),allow_pickle = True) for p in range(cg.num_partitions)]
  segs_list_trn=[np.load(os.path.join(cg.model_dir,'partitions_dir/seg_list_'+str(p)+'.npy'),allow_pickle = True) for p in range(cg.num_partitions)]
    
  if batch is None:
    print('No batch was provided: training on all images.')
    batch = 'all'

    imgs_list_trn = np.concatenate(imgs_list_trn)
    segs_list_trn = np.concatenate(segs_list_trn)
      
    imgs_list_tst = imgs_list_trn
    segs_list_tst = segs_list_trn
   
  else:
    imgs_list_tst = imgs_list_trn.pop(batch)
    segs_list_tst = segs_list_trn.pop(batch)
    
    imgs_list_trn = np.concatenate(imgs_list_trn)
    segs_list_trn = np.concatenate(segs_list_trn)

  len_list=[len(imgs_list_trn),len(segs_list_trn),len(imgs_list_tst),len(segs_list_tst)]
  print(len_list)

  #===========================================
  dv.section_print('Creating and compiling model...')
  losses = {'unet':'categorical_crossentropy','t':'mse','x':'cosine_proximity','y':'cosine_proximity',}
  if vector == 't':
    weights = {'unet':1,'t':1,'x':0,'y':0}  
  elif vector == 'r':
    weights = {'unet':1,'t':0,'x':1,'y':1}  
  else:
    raise ValueError('wrong vector definition')
  
  build = Build_Model()
  model = build.get_model(losses,weights,load_weights = True, load_model_path = pre_trained_model)

  #===========================================
  dv.section_print('Set callbacks')
  callbacks = build.set_callbacks(batch, task = current_task)

  #======================
  dv.section_print('Fitting model...')
  datagen_flow,valgen_flow = build.fit_models(imgs_list_trn,segs_list_trn,imgs_list_tst, segs_list_tst)

  model.fit_generator(datagen_flow,
                      steps_per_epoch = imgs_list_trn.shape[0] // cg.batch_size,
                      epochs = cg.epochs,
                      workers = 1,
                      validation_data = valgen_flow,
                      validation_steps = imgs_list_tst.shape[0] // cg.batch_size,
                      callbacks = callbacks,
                      verbose = 1,
                      )
    
if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--batch', type=int)
  args = parser.parse_args()

  if args.batch is not None:
    assert(0 <= args.batch < cg.num_partitions)

  pre_trained_model = get_pre_trained_model(args.batch,epoch)
  retrain(args.batch,pre_trained_model)
