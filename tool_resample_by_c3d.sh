#!/usr/bin/env bash

# this script resample volumes (CT volume and CT segmentation) into uniform pixel dimensions. (default = 1.5mm)
# run it by ./tool_resample_by_c3d.sh

######## Get a list of patients.
patients=( /Data/McVeighLabSuper/projects/Zhennong/AI/AI_datasets/ucsd_ccta/*/ ) 
img_or_seg=0 # 1 is image, 0 is seg

######## Define folders
save_folder='/Data/McVeighLabSuper/projects/Zhennong/AI/CNN/all-classes-all-phases-1.5/'
if ((${img_or_seg} == 1))
then
img_folder="img-nii" #input folder
out_folder="img-nii-sm" #output folder
else
img_folder="seg-nii"
out_folder="seg-nii-sm" #output folder
fi


# main script (usually no need to change):
for p in ${patients[*]};
do
    if ! [ -d ${p}${img_folder} ] ;then
        echo "no image/seg"
        continue
    fi

    # find out the patient id
    patient_id=$(basename ${p})
    patient_class=$(basename $(dirname ${p}))
    
    # set output folder
    o_dir=${save_folder}${patient_class}/${patient_id}/${out_folder}
    mkdir -p ${save_folder}${patient_class}
    mkdir -p ${save_folder}${patient_class}/${patient_id}
    mkdir -p ${o_dir}
    

    # find all the data 
    IMGS=(${p}${img_folder}/*.nii.gz)

    for i in $(seq 0 $(( ${#IMGS[*]} - 1 )));
    do
        i_file=${IMGS[${i}]}
        o_file=${o_dir}/$(basename ${i_file})
        echo ${i_file}
            
        if [ -f ${o_file} ];then
            echo "already done this file"
            continue
        else
            if ((${img_or_seg} == 1))
            then
                c3d ${i_file} -interpolation Cubic -resample-mm 1.5x1.5x1.5mm -o ${o_file}
            else
                c3d ${i_file} -interpolation NearestNeighbor -resample-mm 1.5mmx1.5mmx1.5mm -o ${o_file}
            fi
        fi        
    done
done


