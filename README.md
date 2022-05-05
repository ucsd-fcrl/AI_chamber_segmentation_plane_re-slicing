# A DL Pipieline to Simultaneously Predict Multi-Chamber Segmentation and Cardiac Imaging Planes
**Author: Zhennong Chen, PhD**<br />

This is the GitHub repo for published paper: <br />
*Automated Cardiac Volume Assessment and Cardiac Long- and Short-Axis Imaging Plane Prediction from ECG-gated CT Volumes Enabled By Deep Learning.*<br />
Authors: Zhennong Chen, Davis Vigneault, Marzia Rogolli, Francisco Contijoch<br />
Citation: Zhennong Chen, Marzia Rigolli, Davis Marc Vigneault, Seth Kligerman, Lewis Hahn, Anna Narezkina, Amanda Craine, Katherine Lowe, Francisco Contijoch, Automated cardiac volume assessment and cardiac long- and short-axis imaging plane prediction from electrocardiogram-gated computed tomography volumes enabled by deep learning, European Heart Journal - Digital Health, Volume 2, Issue 2, June 2021, Pages 311–322, https://doi.org/10.1093/ehjdh/ztab033

## Description
We developed a convolutional neural network to provide automatic, accurate and fast chamber segmentation (Left ventricle and Left atrium) + cardiac imaging planes re-slicing (two-chamber, three-chamber, four-chamber planes + a short-axis stack) from cardiac CT images. 
This convolutional neural network is a variant of conventional U-Net. We modified the U-Net architecuture so that (1) it can take the down-sampled 3D CT image directly as the input and (2) it can predict vectors that can be used to re-slice different cardiac imaging planes.

## User Guideline
### Environment Setup
The entire code is [containerized](https://www.docker.com/resources/what-container). This makes setting up environment swift and easy. Make sure you have nvidia-docker and Docker CE [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) on your machine before going further. 

You can build your own docker from provided dockerfile (Dockerfile_AI_plane). This Docker includes the installation of a github python package "dvpy", which is a toolkit for tensorflow U-Net well-developed by Davis Vigneault and Zhennong Chen.

### Data preparation
Here is a list of things you need to prepare:
1. NIFTI files (.nii.gz) of CT volumes + ground-truth segmentation (for training purpose). Both volumes and segmentation should be pre-down-sampled to 1.5mm isotropically.
2. NIFTI files of ground truth cardiac imaging planes (for training purpose)
3. run tool_extract_affine.py to get ground truth vectors for plane re-slicing
4. run tool_partition.py to split the data for n-fold cross-validation

### Steps:
Follow the steps indicated by file-names to run the code

step 1: define default parameters by running ./main_step1_defaults.sh<br />
step 2A：train the segmentation model by running ./main_step2A_train_segmentation.py --batch N<br />
step 2B: pick the pre-trained segmentation model for weight initialzation for the vector training by running ./main_step2B_trian_vectors.py --batch N<br />
step 3: predict by running python main_step3_predict.py<br />
step 4: make cine image of predicted cardiac imaging planes by running python main_step4_generate_predicted_plane_image.py<br />

### Additional guidelines
see comments in the script

Please contact zhc043@eng.ucsd.edu or chenzhennong@gmail.com for any further questions.




