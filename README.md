# A DL Pipieline to Simultaneously Predict Multi-Chamber Segmentation and Cardiac Imaging Planes
**Author: Zhennong Chen, PhD**<br />

This is the GitHub repo for the published paper: <br />
*Automated Cardiac Volume Assessment and Cardiac Long- and Short-Axis Imaging Plane Prediction from ECG-gated CT Volumes Enabled By Deep Learning.*<br />
Authors: Zhennong Chen, Davis Vigneault, Marzia Rogolli, Francisco Contijoch<br />
Citation: Zhennong Chen, Marzia Rigolli, Davis Marc Vigneault, Seth Kligerman, Lewis Hahn, Anna Narezkina, Amanda Craine, Katherine Lowe, Francisco Contijoch, Automated cardiac volume assessment and cardiac long- and short-axis imaging plane prediction from electrocardiogram-gated computed tomography volumes enabled by deep learning, European Heart Journal - Digital Health, Volume 2, Issue 2, June 2021, Pages 311–322, https://doi.org/10.1093/ehjdh/ztab033

## Description
We developed a convolutional neural network to provide automatic, accurate and fast chamber segmentation (Left ventricle and Left atrium) as well as cardiac imaging planes re-slicing (two-chamber, three-chamber, four-chamber planes + a short-axis stack) from cardiac CT images. <br />
This convolutional neural network is a variant of conventional U-Net. We modified the U-Net architecuture so that (1) it can take the down-sampled 3D CT image directly as the input and (2) it can predict vectors that can be used to re-slice different cardiac imaging planes.

## User Guideline
### Environment Setup
The entire code is [containerized](https://www.docker.com/resources/what-container). This makes setting up environment swift and easy. Make sure you have nvidia-docker and Docker CE [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) on your machine before going further. 
    - You can build your own docker from provided dockerfile ```Dockerfile_cuda100_tensorflow```. ` - - This repo relies on a python package called dvpy. Make sure you have the latest version. If not, in terminal type: pip uninstall dvpy; pip install git+https://github.com/zhennongchen/dvpy.git#egg=dvpy

### Data preparation
Here is a list of things you need to prepare:
1. CT volumes, file format: NIfTI image
2. Ground truth chamber segmentation (at least LV and LA, better to have LVOT as well), file format: NIfTI image
3. Ground truth manual cardiac imaging planes (3 LAX and one SAX), file format: NIfTI image

### Experiment preparation
Here is a list of things you need to do before training the model.
1. re-sample all your data to a uniform pixel dimension (dafault = 1.5mm)
    - for CT volumes and segmentations, use ```./tool_resample_by_c3d.sh```
    - for planes, use ```./tool_resample_planes.sh```
2. extract ground truth plane vectors used to re-slice imaging planes by ```tool_extract_plane_vectors.py```
3. partition the data if you want to do n-fold cross-validation by ```tool_partition.py``` 
4. set default parameters for DL experiments by ```. ./set_defaults.sh```

### Train the Model
we first turn off the penalty of vector prediction and only teach the model to learn segmentation.<br />
we then turn on the penalty of vector prediction, initialize the new train by pre-trained segmentation model, and teach the model to learn vector prediction.<br />
Read the paper "Methods" section for more details about the training strategy.

```main_train_1_segmentation.py```: teach model to learn segmentation<br />
```main_train_2_vectors.py```: teach model to learn plane vector predictions<br />

### Predict for new cases
```main_prediction.py```: predict segmentation and plane vectors for new cases<br />
```main_generate_predicted_plane_movie```: use plane vectors to generate a cine movie of imaging planes showing the cardiac function across cardiac cycle. see Example_plane_cine.mp4 for how the movie looks like. <br />

We highly recommend to use [another GitHub Repo](https://github.com/ucsd-fcrl/DL_CT_Seg-Plane_Prediction_Final_v_ZC) designed specific to predict segmentation&planes on new cases using trained DL model. It can generate better (higher resolution) results and is useful when you have more than one trained models for the same task.


### Additional guidelines
see comments in the script

Please contact zhc043@eng.ucsd.edu or chenzhennong@gmail.com for any further questions.




