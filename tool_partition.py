#!/usr/bin/env python

## this script is used for partition data into several groups for cross-validation

# System
import os
import glob as gb
import pathlib as plib
import numpy as np
import dvpy as dv
import supplements
import function_list as ff
cg = supplements.Experiment()


#make the directories
os.makedirs(cg.model_dir, exist_ok = True)
# Create a list of all patients.
patient_list = ff.find_all_target_files(['ucsd_*/*'],cg.data_dir)
print(patient_list)
# Shuffle the patients.
np.random.shuffle(patient_list)
# Split the list into `cg.num_partitions` (approximately) equal sublists.
partitions = np.array_split(patient_list, cg.num_partitions)
# Save the partitions.
np.save(os.path.join(cg.model_dir,'partitions.npy'), partitions)

def create_img_lists():
    partitions = np.load(os.path.join(cg.model_dir,'partitions.npy'))
    for i, partition in enumerate(partitions):
        es = [os.path.join(c, 'es.txt') for c in partition] # it's the txt file savin ED and ES time frames
        es = [int(open(s, 'r').read()) for s in es]
        segs = [[os.path.join(c, 'seg-nii-sm/0.nii.gz'), os.path.join(c, 'seg-nii-sm/'+str(f)+'.nii.gz')] for c, f in zip(partition, es)]
        segs = dv.collapse_iterable(segs)
        imgs = [os.path.join(os.path.dirname(os.path.dirname(s)), 'img-nii-sm', os.path.basename(s)) for s in segs]
        assert(len(imgs) == len(segs))
        os.makedirs(os.path.join(cg.model_dir, 'partitions_dir'), exist_ok = True)
        np.save(os.path.join(cg.model_dir,'partitions_dir/img_list_'+str(i)+'.npy'),imgs)
        np.save(os.path.join(cg.model_dir,'partitions_dir/seg_list_'+str(i)+'.npy'),segs)
    
if __name__ == '__main__':

  # Set a seed, so that np.random.shuffle() is reproducible.
  np.random.seed(cg.seed)

  # Create the partition lists.
  create_img_lists()
   
