## Define Parameters
# GPU usage
export CUDA_VISIBLE_DEVICES="1"

# Define number of classes predicted, 2 for LV only, 3 for LV+LA.
export CG_NUM_CLASSES=3

# Define the resampled pixel size and resampled dimension
export CG_SPACING=1.5
export CG_CROP_X=160
export CG_CROP_Y=160
export CG_CROP_Z=96

# Define the number of feature channels in the last down-sampled stage
export CG_FEATURE_DEPTH=8

# Define the training epochs 
export CG_EPOCHS=40
export CG_LR_EPOCHS=26 # learning rate decays exponetionally every 26 epochs

# Define batch size 
export CG_BATCH_SIZE=1

# Define data augmentation parameters:
export CG_XY_RANGE="0.1"   
export CG_ZM_RANGE="0.1"  
export CG_RT_RANGE="10"   

# Define the number of cross-validation folder
export CG_NUM_PARTITIONS=5

## Define folders
export CG_MAIN_DIR="/Data/McVeighLabSuper/projects/Zhennong/AI_publish_test/"
export CG_DATA_DIR="/Experiment/Documents/Data_1.5/" # data saved in the remote local machine where ML is run
export CG_PREDICT_DIR="/Data/McVeighLabSuper/projects/Zhennong/AI_publish_test/Predictions/" # folder where the predictions are saved
export CG_MODEL_DIR="/Data/McVeighLabSuper/projects/Zhennong/AI_publish_test/Models/" # folder to save models


