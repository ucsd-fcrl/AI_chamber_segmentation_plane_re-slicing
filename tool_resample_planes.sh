#!/usr/bin/env bash

# this script resample planes nii file into uniform pixel dimension (default pixel_dim = 1.5mm)
# run it by ./tool_resample_planes_by_c3d.sh

####### define patient list
patients=( /Data/McVeighLabSuper/projects/Zhennong/AI/AI_datasets/ucsd_ccta/*/ ) 

####### define folder
IMAGE_DIR='/Data/McVeighLabSuper/projects/Zhennong/AI/CNN/all-classes-all-phases-1.5/'
SAVE_DIR='/Data/McVeighLabSuper/projects/Zhennong/AI/CNN/all-classes-all-phases-1.5/'

# Folder where the origninal plane nii sits 
mpr_fld=mpr_new_nii
# Folder where the re-sampled volumes to be resliced sit
input_fld=img-nii-sm
# Folder where you want to save the resampled planes
output_fld=mpr-new-nii-sm-1.5




# main script (usually no need to change)
set -o nounset
set -o errexit
set -o pipefail

out_size=480;
out_spac=0.625;
out_value=-2047;

dv_utils_fld="/Experiment/Documents/Repos/dv-commandline-utils/bin/"

SLICE[0]=2C
SLICE[1]=3C
SLICE[2]=4C
# SLICE[3]=BASAL

for p in ${patients[*]};
do
    patient_id=$(basename ${p})
    patient_class=$(basename $(dirname ${p}))

    if  [ ! -d ${p}${mpr_fld} ];then
        echo "no plane image"
        continue
    fi

    save_folder=${SAVE_DIR}${patient_class}/${patient_id}/${output_fld}
    mkdir -p ${SAVE_DIR}${patient_class}
    mkdir -p ${SAVE_DIR}${patient_class}/${patient_id}
    mkdir -p ${save_folder}

    IMGS=(${IMAGE_DIR}${patient_class}/${patient_id}/${input_fld}/*.nii.gz) 

    for i in $(seq 0 $(( ${#IMGS[*]} - 1 )));
    do
        for s in ${SLICE[*]};
        do
            REF=( ${p}${mpr_fld}/${s}/$(basename ${IMGS[${i}]}))

            slc_folder=${save_folder}/${s}/
            mkdir -p ${slc_folder}

            output_file=${slc_folder}$(basename ${REF[0]})
        
            echo ${IMGS[${i}]}
            echo ${REF[0]}
            echo $output_file

            if [ -f ${output_file} ];then
                echo "already done"
                continue
            else
                ${dv_utils_fld}dv-resample-from-reference --input-image ${IMGS[${i}]} --reference-image ${REF[0]} --output-image $output_file --output-size $out_size --output-spacing $out_spac --outside-value $out_value
            fi
        done
    done
done