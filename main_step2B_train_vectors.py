#!/usr/bin/env python

# this script trains the AI to predict directional vector/translational vector. The AI model's weights are initialized by the pre-trained segmentation model.
# in terminal, type ./main_step2B_train_vectors--batch X to run

# System
import argparse
import os

# Third Party
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import CSVLogger
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.layers.core import Activation
from keras.layers import Input, \
                         Conv1D, Conv2D, Conv3D, \
                         MaxPooling1D, MaxPooling2D, MaxPooling3D, \
                         UpSampling1D, UpSampling2D, UpSampling3D, \
                         Reshape, Flatten, Dense
from keras.layers.merge import concatenate, multiply
from keras.initializers import Orthogonal
from keras.regularizers import l2
from keras.layers.merge import concatenate, multiply

import tensorflow as tf
## https://github.com/avolkov1/keras_experiments
#from keras_exp.multigpu import get_available_gpus, make_parallel

# Internal
import supplements
import supplements.utils as ut
import dvpy as dv
import dvpy.tf
import function_list as ff

cg = supplements.Experiment()

K.set_image_dim_ordering('tf')  # Tensorflow dimension ordering in this code

# Allow Dynamic memory allocation.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def train(batch):
    # define your task
    vector = 't' # 't' or 'r'
    view = '3C' # '2C', '3C', '4C' or 'BASAL'
    print('view is: ',view,'vector is',vector)

    # define pre-trained model (model trained on segmentation)
    pre_trained_model = 'model-batch0_s-040-*'

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
    shape = cg.dim + (1,)
    model_inputs = [Input(shape)]
    model_outputs=[]
    ds_layer, _, unet_output = dvpy.tf.get_unet(cg.dim,
                                    cg.num_classes,
                                    cg.conv_depth,
                                    layer_name='unet',
                                    dimension =len(cg.dim),
                                    unet_depth = cg.unet_depth,
                                   )(model_inputs[0])
    model_outputs += [unet_output]
    
    ds_flat = Flatten()(ds_layer)
    
    # Loc Networks.

    Loc= Dense(384,
                    kernel_initializer=Orthogonal(gain=1.0),
                    kernel_regularizer = l2(1e-4),
                    activation='relu',name='Loc')(ds_flat)


    translation = Dense(3, #output dimension
                        kernel_initializer=Orthogonal(gain=1e-1),
                        kernel_regularizer = l2(1e-4),
                        name ='t')(Loc)

    x_direction = Dense(3, #output dimension
                        kernel_initializer=Orthogonal(gain=1e-1),
                        kernel_regularizer = l2(1e-4),
                        name ='x')(Loc)
    
    y_direction = Dense(3, #output dimension
                        kernel_initializer=Orthogonal(gain=1e-1),
                        kernel_regularizer = l2(1e-4),
                        name ='y')(Loc)

    
    model_outputs += [translation]
    model_outputs += [x_direction]
    model_outputs += [y_direction]
  
    
    model = Model(inputs = model_inputs,outputs = model_outputs)
    
    # load the previously trained model
    if batch is None:
      pre_model_path = ff.find_all_target_files([pre_trained_model],os.path.join(cg.model_dir,'model_batchall'))
    else:
      pre_model_path = ff.find_all_target_files([pre_trained_model],os.path.join(cg.model_dir,'model_batch'+str(batch)))
    assert len(pre_model_path) == 1
    print('the pre-trained model is: ', pre_model_path[0])
    model.load_weights(pre_model_path[0])

    # define loss and regularization
    opt = Adam(lr = 1e-4)
    losses={'unet':'categorical_crossentropy','t':'mse','x':'cosine_proximity','y':'cosine_proximity',}
    if vector == 't':
      weight={'unet':1,'t':1,'x':0,'y':0}  
    elif vector == 'r':
      weight={'unet':1,'t':0,'x':1,'y':1}  
    else:
      raise ValueError('wrong vector definition')

    model.compile(optimizer= opt, 
                 loss= losses,
                 metrics= {'unet':'acc',},
                loss_weights = weight)
  
    #======================
    dv.section_print('Fitting model...')
    
    if batch is None:
      model_name = 'model-batchall_'+view+'_'+vector
      model_fld = 'model_batchall'
    else:
      model_name = 'model-batch'+str(batch)+'_'+view+'_'+vector
      model_fld = 'model_batch'+str(batch)
    filename = model_name + '-{epoch:03d}-{loss:.3f}-{val_loss:.3f}-{val_unet_loss:.4f}-{val_unet_acc:.3f}-{val_t_loss:.4f}-{val_x_loss:.4f}-{val_y_loss:.4f}.hdf5'
    ff.make_folder([os.path.join(cg.model_dir,model_fld),os.path.join(cg.model_dir,model_fld,view+'_'+vector)])
    filepath=os.path.join(cg.model_dir,model_fld,view+'_'+vector,filename)
  
    # set callbacks
    callbacks = [ModelCheckpoint(filepath,          
                                 monitor='val_loss',
                                 save_best_only=False,
                                 ),
                 LearningRateScheduler(dv.learning_rate_step_decay),   
                ]
    
   
    datagen = dv.tf.ImageDataGenerator(
        3,  
        input_layer_names = ['input_1'],
        output_layer_names = ['unet','t','x','y'],
        translation_range=cg.xy_range,  
        rotation_range=cg.rt_range,  
        scale_range=cg.zm_range,
        flip=cg.flip,)
    
    datagen_flow = datagen.flow(imgs_list_trn,
      segs_list_trn,
      batch_size = cg.batch_size,
      view = view,
      relabel_LVOT = True,
      input_adapter = ut.in_adapt,
      output_adapter = ut.out_adapt,
      shape = cg.dim,
      input_channels = 1,
      output_channels = cg.num_classes,
      augment = True,
      )

    valgen = dv.tf.ImageDataGenerator(
        3, 
        input_layer_names=['input_1'],
        output_layer_names=['unet','t','x','y'],
        )

    valgen_flow = valgen.flow(imgs_list_tst,
      segs_list_tst,
      batch_size = cg.batch_size,
      view = view,
      relabel_LVOT = True,
      input_adapter = ut.in_adapt,
      output_adapter = ut.out_adapt,
      shape = cg.dim,
      input_channels = 1,
      output_channels = cg.num_classes,
      )
    
    # Fit the model on the batches generated by datagen.flow().
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

  train(args.batch)
