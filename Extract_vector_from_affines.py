import os
import numpy as np
import glob
import nibabel as nib
from nibabel.affines import apply_affine
import supplements
import supplements.utils as ut
import math
import function_list as ff
cg = supplements.Experiment()

o = [0,0,0]
x1 = [1,0,0]
y1 = [0,1,0]
z1 = [0,0,1]

class Vector():
    def __init__(self,main_folder,patient_class,patient_id,plane_list,plane_choice):
        self.main_folder = main_folder
        self.patient_class = patient_class
        self.patient_id = patient_id
        self.plane_list = plane_list
        self.plane_choice = plane_choice

    def get_vector_from_affines(self, i,j,i_affine,j_affine): 
        #i = CT image,j = CT planes
        x=nib.load(i)
        y=nib.load(j)
        s1=x.shape
        s2=y.shape
        A = np.linalg.inv(i_affine).dot(j_affine)

        # translation of center
        mpr_center=np.array([(s2[0]-1)/2,(s2[1]-1)/2,0])
        img_center=np.array([(s1[0]-1)/2,(s1[1]-1)/2,(s1[-1]-1)/2])
        mpr_center_img = ff.convert_coordinates(i_affine,j_affine,mpr_center)
        translation_c = mpr_center_img - img_center
            
        # normalize translation into padding coordinate system
        x_pad = ut.in_adapt(i)
        p_size = x_pad.shape
        translation_c_n = np.array([translation_c[0]/p_size[0]*2,translation_c[1]/p_size[1]*2,translation_c[2]/p_size[2]*2])

        
        # translation of origin
        translation_o = ff.convert_coordinates(i_affine,j_affine,[0,0,0]) - [0,0,0]
        translation_o_n = np.array([translation_o[0]/p_size[0]*2,translation_o[1]/p_size[1]*2,translation_o[2]/p_size[2]*2])


        # x_direction 
        x_d = ff.convert_coordinates(i_affine,j_affine,x1) - ff.convert_coordinates(i_affine,j_affine,o)
        x_scale = np.linalg.norm(x_d)
        x_n = np.asarray([i/x_scale for i in x_d])
            
            
        # y_direction
        y_d = ff.convert_coordinates(i_affine,j_affine,y1) - ff.convert_coordinates(i_affine,j_affine,o)
        y_scale = np.linalg.norm(y_d)
        y_n = np.asarray([i/y_scale for i in y_d])

        # z_direction
        z_d = ff.convert_coordinates(i_affine,j_affine,z1) - ff.convert_coordinates(i_affine,j_affine,o)
        z_scale = np.linalg.norm(z_d)
        z_n = np.asarray([i/z_scale for i in z_d])

        # scale
        x_s = ff.length(x_d)/1
        y_s = ff.length(y_d)/1
        z_s = ff.length(z_d)/1
        scale = np.array([x_s,y_s,z_s])

    
        # put all into a numpy array
        # no need to care about why there are three 0 here, it's just the default format of plane vector numpy array used in my paper
        vectors=np.array([0,0,0, translation_o, translation_o_n, x_d, x_n, y_d, y_n, z_d, z_n, scale,translation_c,translation_c_n,img_center]) 
        
        return vectors

    def save_vectors(self,img_fld,mpr_fld,save_fld):

        for i in self.plane_choice:
            plane = self.plane_list[i]

            # define save folder
            save_folder=os.path.join(self.main_folder,self.patient_class,self.patient_id,save_fld)
            save_file = os.path.join(save_folder, plane+'.npy')
            ff.make_folder([save_folder])
            
            if os.path.isfile(save_file):
                print('already get the matrix for ', plane)
                continue

            #image 1.5mm
            i = os.path.join(self.main_folder,self.patient_class,self.patient_id,img_fld,'0.nii.gz')
            i_affine = ff.check_affine(i)
        
            #mpr 1.5mm
            j = os.path.join(self.main_folder,self.patient_class,self.patient_id,mpr_fld,plane,'0.nii.gz')
            j_affine = nib.load(j).affine

            # get vectors
            matrix = self.get_vector_from_affines(i,j,i_affine,j_affine)

            # save
            np.save(save_file, matrix)


    def calculate_conversion(self,shape):
        volume_center=np.array([int((shape[0]-1)/2),int((shape[1]-1)/2),int((shape[-1]-1)/2)])
        delta=np.array([int((cg.dim[0]-1)/2),int((cg.dim[1]-1)/2),int((cg.dim[-1]-1)/2)])-volume_center
        return delta


    def save_padding_coordinate_conversion(self,save_fld):
        image_file = os.path.join(self.main_folder,self.patient_class, self.patient_id,'img-nii-sm','0.nii.gz')
        
        save_folder=os.path.join(self.main_folder,self.patient_class,self.patient_id,save_fld)
        save_file = os.path.join(save_folder, 'padding_coordinate_conversion.npy')
        ff.make_folder([save_folder])
            
        if os.path.isfile(save_file):
            print('already get the padding coordinate conversion')
        else:
            volume_data = nib.load(image_file).get_fdata()
            shape = volume_data.shape
            delta = self.calculate_conversion(shape)
            np.save(save_file,delta)



    
