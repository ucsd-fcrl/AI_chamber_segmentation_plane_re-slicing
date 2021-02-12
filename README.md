# AI_chamber_segmentation_plane_re-slicing

=======

Authors: Zhennong Chen, Davis Vigneault, Marzia Rogolli and Francisco Contijoch
Citation: Paper under review by European Heart Journal - Digital Health.

## Description
We developed a convolutional neural network to provide automatic, accurate and fast chamber segmentation (Left ventricle and Left atrium) + cardiac imaging planes re-slicing (two-chamber, three-chamber, four-chamber planes + a short-axis stack) from cardiac CT images. 
This convolutional neural network is a variant of conventional U-Net. We modified the U-Net architecuture so that (1) it can take the down-sampled 3D CT image directly as the input and (2) it can predict vectors that can be used to re-slice different cardiac imaging planes.

## User Guideline
### Environment Setup
The entire code is [containerized](https://www.docker.com/resources/what-container). This makes setting up NMF swift and easy. Make sure you have nvidia-docker and Docker CE [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) on your machine before going further. 

You can build your own docker from provided dockerfile (Dockerfile_AI_plane). This Docker includes the installation of a github python package "dvpy", which is a toolkit for tensorflow U-Net well-developed by Davis Vigneault and Zhennong Chen.

### Data preparation
Here is a list of things you need to prepare:
1. NIFTI files (.nii.gz) of CT volumes + ground-truth segmentation (for training purpose). Both volumes and segmentation should be pre-down-sampled to 1.5mm isotropically.
2. NIFTI files of ground truth cardiac imaging planes (for training purpose)
3. run tool_extract_affine.py to get ground truth vectors for plane re-slicing
4. run tool_partition.py to split the data for n-fold cross-validation

### Steps:
Follow the steps indicated by file-names to run the code

step 1: define default parameters by running ./main_step1_defaults.sh

step 2A：train the segmentation model by running ./main_step2A_train_segmentation.py --batch N

step 2B: pick the pre-trained segmentation model for weight initialzation for the vector training (run ./main_step2B_trian_vectors.py --batch N

step 3: predict by running python main_step3_predict.py

step 4: make predicted cine image of cardiac imaging planes by running python main_step4_generate_predicted_plane_image.py

### Additional guidelines
see comments in the script

Please contact zhc043@eng.ucsd.edu for any further questions.




