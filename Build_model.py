#!/usr/bin/env python

# System
import argparse
import os

import numpy as np
import nibabel as nb
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

import supplements
import supplements.utils as ut
import dvpy as dv
import dvpy.tf
import function_list as ff

cg = supplements.Experiment()

K.set_image_dim_ordering('tf')  

# Allow Dynamic memory allocation.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class Build_Model():
    def __init__(self):
        self.build_blank_model = self.get_model(0,0,False,False,False)
        self.model_list = self.get_trained_model_list()
    
    def get_model(self,losses, weights, load_weights = False, load_model_path = False, compile = True):
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

    
        if load_weights == True:
            model.load_weights(load_model_path)

        if compile == True:
            opt = Adam(lr = 1e-4) #change
            model.compile(optimizer= opt, 
                        loss= losses,
                        metrics= {'unet':'acc',},
                        loss_weights = weights)

        return model

    def set_callbacks(self,batch, task = 's'):
        if batch is None:
            model_name = 'model-batchall_'+task
            model_fld = 'model_batchall'
        else:
            model_name = 'model-batch'+str(batch)+'_'+task
            model_fld = 'model_batch'+str(batch)
        filename = model_name +'-{epoch:03d}.hdf5'
        filepath=os.path.join(cg.model_dir,model_fld, task, filename)  
        ff.make_folder([os.path.dirname(os.path.dirname(filepath)), os.path.dirname(filepath)])

        csv_logger = CSVLogger(os.path.join(cg.model_dir, 'logs',  model_name + '_training-log' + '.csv')) # log will automatically record the train_accuracy/loss and validation_accuracy/loss in each epoch
        ff.make_folder([os.path.join(cg.model_dir, 'logs')])

        # set callbacks
        callbacks = [csv_logger, ModelCheckpoint(filepath,          
                                    monitor='val_loss',
                                    save_best_only=False,
                                    ),
                    LearningRateScheduler(dv.learning_rate_step_decay2),   
                    ]
        return callbacks

    def fit_models(self,imgs_list_trn,segs_list_trn,imgs_list_tst, segs_list_tst):

        datagen = dv.tf.ImageDataGenerator(
        3,  # Dimension of input image
        input_layer_names = ['input_1'],
        output_layer_names = ['unet','t','x','y'],
        translation_range=cg.xy_range,  # randomly shift images vertically (fraction of total height)
        rotation_range=cg.rt_range,  # randomly rotate images in the range (degrees, 0 to 180)
        scale_range=cg.zm_range,
        flip=cg.flip,)
    
        datagen_flow = datagen.flow(imgs_list_trn,
        segs_list_trn,
        batch_size = cg.batch_size,
        view = '2C', # default
        relabel_LVOT = True, # feature developed for lvot segmentation
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
        view = '2C', # default
        relabel_LVOT = True,
        input_adapter = ut.in_adapt,
        output_adapter = ut.out_adapt,
        shape = cg.dim,
        input_channels = 1,
        output_channels = cg.num_classes,
        )

        return datagen_flow, valgen_flow

    def get_trained_model_list(self):
        # update the file path of your own trained models with highest validation accuracy

        # model to do segmentation
        model_s = ['*batch0/model-U2_batch0_s-050-*']

        # model to predict translation vector of 2-chamber plane
        model_2C_t = ['*batch0/model-U2_batch0_2C_t3-005-*']

        # model to predict directional vector of 2-chamber plane
        model_2C_r = ['*batch0/model-U2_batch0_2C_r-032-*']

        # model to predict translation vector of 3-chamber plane
        model_3C_t = ['*batch0/model-U2_batch0_3C_t-037-*']

        # model to predict directional vector of 3-chamber plane
        model_3C_r = ['*batch0/model-U2_batch0_3C_r-040-*']

        # model to predict translation vector of 4-chamber plane
        model_4C_t = ['*batch0/model-U2_batch0_4C_t-032-*']

        # model to predict directional vector of 4-chamber plane
        model_4C_r = ['*batch0/model-U2_batch0_4C_r-018-*']

        # model to predict translation vector of short-axis plane
        model_BASAL_t = ['*batch0/model-U2_batch0_BASAL_t2-026-*']

        # model to predict directional vector of short-axis plane
        model_BASAL_r = ['*batch1/model-U2_batch1_BASAL_r-018-*']

        MODEL_list = [model_s,model_2C_t,model_2C_r,model_3C_t,model_3C_r,model_4C_t,model_4C_r,model_BASAL_t,model_BASAL_r]

        return MODEL_list

    def generator_parameters(self,task_name):
        if task_name == 's':
            view = '2C'
            return [view,'']
        else:
            view = task_name.split('_')[0]
            vector = task_name.split('_')[1]
            return [view,vector]

    def image_generator(self):
        return dv.tf.ImageDataGenerator(3,input_layer_names=['input_1'],output_layer_names=['unet','t','x','y'],)

    def save_segmentation(self,img,seg_pred,save_path):
        u_gt_nii = nb.load(img) # get affine matrix from image
        u_pred = np.argmax(seg_pred[0], axis = -1).astype(np.uint8)
        u_pred = dv.crop_or_pad(u_pred, u_gt_nii.get_data().shape)
        u_pred[u_pred == 3] = 4 # relabel LVOT
        u_pred = nb.Nifti1Image(u_pred, u_gt_nii.affine)
        nb.save(u_pred, save_path)

    def save_vector(self,t_pred,x_pred,y_pred,save_path):
        x_n = ff.normalize(x_pred)
        y_n = ff.normalize(y_pred)
        matrix = np.concatenate((t_pred.reshape(1,3),x_n.reshape(1,3),y_n.reshape(1,3)))
        np.save(save_path,matrix)