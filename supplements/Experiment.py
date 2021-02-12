# System
import os

class Experiment():

  def __init__(self):

    # # folder
    self.main_dir = os.environ['CG_MAIN_DATA_DIR']
    self.data_dir = os.environ['CG_DATA_DIR']
    self.predict_dir = os.environ['CG_PREDICT_DIR']
    self.model_dir = os.environ['CG_MODEL_DIR']

    # Dimension of padded input, for training.
    self.dim = (int(os.environ['CG_CROP_X']), int(os.environ['CG_CROP_Y']), int(os.environ['CG_CROP_Z']))
  
    # Seed for randomization.
    self.seed = 1
  
    # Number of Classes (Including Background)
    self.num_classes = int(os.environ['CG_NUM_CLASSES'])
      
    # UNet Depth
    self.unet_depth = 5
  
    # Depth of convolutional feature maps
    self.conv_depth_multiplier = 1
    self.ii = int(os.environ['CG_FEATURE_DEPTH'])
    self.conv_depth = [2**(self.ii-4),2**(self.ii-3),2**(self.ii-2),2**(self.ii-1),2**(self.ii),2**(self.ii),2**(self.ii-1),2**(self.ii-2),
                      2**(self.ii-3),2**(self.ii-4),2**(self.ii-4)]
  
    self.conv_depth = [self.conv_depth_multiplier*x for x in self.conv_depth]
  
    assert(len(self.conv_depth) == (2*self.unet_depth+1))
  
    # How many images should be processed in each batch?
    self.batch_size = int(os.environ['CG_BATCH_SIZE'])
  
    # SPACING:
    self.spacing = float(os.environ['CG_SPACING'])

    # number of partitions
    self.num_partitions = int(os.environ['CG_NUM_PARTITIONS'])

    # data augmentation
    # Translation Range
    self.xy_range = float(os.environ['CG_XY_RANGE'])
  
    # Scale Range
    self.zm_range = float(os.environ['CG_ZM_RANGE'])

    # Rotation Range
    self.rt_range=float(os.environ['CG_RT_RANGE'])
  
    # Should Flip
    self.flip = False

     # Total number of epochs to train
    self.epochs = int(os.environ['CG_EPOCHS'])

    # Number of epochs to train before decreasing learning rate
    self.lr_epochs = int(os.environ['CG_LR_EPOCHS'])
