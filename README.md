# AI_chamber_segmentation_plane_re-slicing

Authors: Zhennong Chen, Davis Vigneault, Marzia Rogolli and Francisco Contijoch
Citation: Paper under review by European Heart Journal - Digital Health.

## Description

This convolutional neural network is a variant of conventional U-Net. We modified the U-Net architecuture so that (1) it can take the down-sampled 3D CT image directly as the input and (2) it can predict vectors that can be used to re-slice different cardiac imaging planes. 

## User Guideline
### Environment Setup
The entire code is [containerized](https://www.docker.com/resources/what-container). This makes setting up NMF swift and easy. Make sure you have nvidia-docker and Docker CE [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) on your machine before going further. 

You can build your own docker from provided dockerfile (Dockerfile_AI_plane)



