import function_list as ff
import os
import math
import numpy as np
import nibabel as nib 
import supplements
from PIL import Image
cg = supplements.Experiment()


class Prepare():
    def __init__(self,main_folder,patient_class,patient_id,pixel_spacing,plane_image_size,image_folder):
        self.main_folder = main_folder
        self.patient_class = patient_class
        self.patient_id = patient_id
        self.pixel_spacing = pixel_spacing
        self.plane_image_size = plane_image_size
        self.image_folder = image_folder

    def load_plane_vectors(self, vector_folder, normal_vector_flip = False):
        patient_path = os.path.join(self.main_folder,self.patient_class,self.patient_id)

        volume_dim = nib.load(os.path.join(self.image_folder,'0.nii.gz')).shape
        image_center = np.array([(volume_dim[0]-1)/2,(volume_dim[1]-1)/2,(volume_dim[-1]-1)/2]) 
        scale = [1,1,0.67] # default

        # load vectors
        vector_2C = ff.get_predicted_vectors(os.path.join(patient_path,vector_folder,'pred_2C_t.npy'),os.path.join(patient_path,vector_folder,'pred_2C_r.npy'),scale, image_center)
        vector_3C = ff.get_predicted_vectors(os.path.join(patient_path,vector_folder,'pred_3C_t.npy'),os.path.join(patient_path,vector_folder,'pred_3C_r.npy'),scale, image_center)
        vector_4C = ff.get_predicted_vectors(os.path.join(patient_path,vector_folder,'pred_4C_t.npy'),os.path.join(patient_path,vector_folder,'pred_4C_r.npy'),scale, image_center)
        vector_SA = ff.get_predicted_vectors(os.path.join(patient_path,vector_folder,'pred_BASAL_t.npy'),os.path.join(patient_path,vector_folder,'pred_BASAL_r.npy'),scale, image_center)
       
        if normal_vector_flip == False:
            normal_vector = ff.normalize(np.cross(vector_SA['x'],vector_SA['y']))
        else:  # this is specific for SAX planes made by Horos/OsiriX
            normal_vector = -ff.normalize(np.cross(vector_SA['x'],vector_SA['y']))

        return image_center, vector_2C, vector_3C, vector_4C, vector_SA, normal_vector


    def obtain_affine_matrix(self, vector_2C, vector_3C, vector_4C):
        volume_affine = ff.check_affine(os.path.join(self.image_folder,'0.nii.gz'))
        A_2C = ff.get_affine_from_vectors(np.zeros(self.plane_image_size),volume_affine,vector_2C,1.0)
        A_3C = ff.get_affine_from_vectors(np.zeros(self.plane_image_size),volume_affine,vector_3C,1.0)
        A_4C = ff.get_affine_from_vectors(np.zeros(self.plane_image_size),volume_affine,vector_4C,1.0)
        return volume_affine, A_2C, A_3C, A_4C


    def define_SAX_range(self,vector_SA,image_center, seg_LV, txt_write = False):
        pixel_dim = math.sqrt((self.pixel_spacing**2)*3) # default = 2.59 = SQRT((1.5^2)*3)

        a,b = ff.find_num_of_slices_in_SAX(np.zeros(self.plane_image_size),image_center,vector_SA['t'],vector_SA['x'],vector_SA['y'],seg_LV,0,pixel_dim)
        print('a is ',a,'b is ',b)
        if txt_write == True:
            t_file = open(os.path.join(self.main_folder,self.patient_class,self.patient_id,"slice_num_info.txt"),"w+")
            t_file.write("num of slices before basal = %d\nnum of slices after basal = %d" % (a, b))
            t_file.close()
        return a,b

    def define_SAX_planes_center_list(self,vector_SA, image_center,a,b,normal_vector):
        # define the plane center coordinates for 9 SAX planes in the SAX stack
        pixel_dim = math.sqrt((self.pixel_spacing ** 2)*3) # default = 2.59 = SQRT((1.5^2)*3)
        # center list of the whole SAX stack
        center_list = ff.find_center_list_whole_stack(image_center + vector_SA['t'],normal_vector,a,b,8,pixel_dim)
        # center list of the 9 SAX planes
        index_list,center_list9,gap = ff.resample_SAX_stack_into_particular_num_of_planes(range(2,center_list.shape[0]),9,center_list)
        return center_list, center_list9, gap


class Make_Planes():
    def __init__(self,main_folder, patient_class, patient_id, volume_data,window_set, image_center,vectors, volume_affine, center_list9 ,pixel_spacing, plane_image_size, zoom_factor_LAX = 1.25, zoom_factor_SAX = 1.4):
        self.main_folder = main_folder
        self.patient_class = patient_class
        self.patient_id = patient_id

        self.volume_data = volume_data
        self.WL = window_set[0] # window_level
        self.WW = window_set[1] # window_width 
        self.image_center = image_center
        self.vector_2C = vectors[0]
        self.vector_3C = vectors[1]
        self.vector_4C = vectors[2]
        self.vector_SA = vectors[3]
        self.volume_affine = volume_affine
        self.center_list9 = center_list9
        self.zoom_factor_LAX = zoom_factor_LAX
        self.zoom_factor_SAX = zoom_factor_SAX
        self.pixel_spacing = pixel_spacing
        self.plane_image_size = plane_image_size



    def draw_lines(self,twoc,threec,fourc,sax_collection, normalize = True):
        # prepare new affines for zoomed images
        A_2C_zoomed = ff.get_affine_from_vectors(np.zeros(self.plane_image_size),self.volume_affine,self.vector_2C,self.zoom_factor_LAX)
        A_3C_zoomed = ff.get_affine_from_vectors(np.zeros(self.plane_image_size),self.volume_affine,self.vector_3C,self.zoom_factor_LAX)
        A_4C_zoomed = ff.get_affine_from_vectors(np.zeros(self.plane_image_size),self.volume_affine,self.vector_4C,self.zoom_factor_LAX)

        # draw intersection lines: LAX lines on SAX
        sax_w_2c = []; sax_w_3c = []; sax_w_4c = []
        for ii in range(0,9):
            line2c,_,_ = ff.draw_plane_intersection(sax_collection[ii],self.vector_2C['x'],self.vector_2C['y'],A_2C_zoomed,ff.get_affine_from_vectors(np.zeros(self.plane_image_size),self.volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(self.center_list9[ii],self.image_center,self.vector_SA),self.zoom_factor_SAX),self.volume_affine)
            line3c,_,_ = ff.draw_plane_intersection(sax_collection[ii],self.vector_3C['x'],self.vector_3C['y'],A_3C_zoomed,ff.get_affine_from_vectors(np.zeros(self.plane_image_size),self.volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(self.center_list9[ii],self.image_center,self.vector_SA),self.zoom_factor_SAX),self.volume_affine)
            line4c,_,_ = ff.draw_plane_intersection(sax_collection[ii],self.vector_4C['x'],self.vector_4C['y'],A_4C_zoomed,ff.get_affine_from_vectors(np.zeros(self.plane_image_size),self.volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(self.center_list9[ii],self.image_center,self.vector_SA),self.zoom_factor_SAX),self.volume_affine)
            sax_w_2c.append(line2c)
            sax_w_3c.append(line3c)
            sax_w_4c.append(line4c)
            
        # draw intersection lines: SAX lines on LAX
        twoc_line = np.copy(twoc); threec_line = np.copy(threec); fourc_line = np.copy(fourc)
        twoc_line,_,_ = ff.draw_plane_intersection(twoc_line,self.vector_SA['x'],self.vector_SA['y'],ff.get_affine_from_vectors(np.zeros(self.plane_image_size),self.volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(self.center_list9[2],self.image_center,self.vector_SA),self.zoom_factor_SAX),A_2C_zoomed,self.volume_affine)
        twoc_line,_,_ = ff.draw_plane_intersection(twoc_line,self.vector_SA['x'],self.vector_SA['y'],ff.get_affine_from_vectors(np.zeros(self.plane_image_size),self.volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(self.center_list9[4],self.image_center,self.vector_SA),self.zoom_factor_SAX),A_2C_zoomed,self.volume_affine)
        twoc_line,_,_ = ff.draw_plane_intersection(twoc_line,self.vector_SA['x'],self.vector_SA['y'],ff.get_affine_from_vectors(np.zeros(self.plane_image_size),self.volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(self.center_list9[6],self.image_center,self.vector_SA),self.zoom_factor_SAX),A_2C_zoomed,self.volume_affine)
        threec_line,_,_ = ff.draw_plane_intersection(threec_line,self.vector_SA['x'],self.vector_SA['y'],ff.get_affine_from_vectors(np.zeros(self.plane_image_size),self.volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(self.center_list9[2],self.image_center,self.vector_SA),self.zoom_factor_SAX),A_3C_zoomed,self.volume_affine)
        threec_line,_,_ = ff.draw_plane_intersection(threec_line,self.vector_SA['x'],self.vector_SA['y'],ff.get_affine_from_vectors(np.zeros(self.plane_image_size),self.volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(self.center_list9[4],self.image_center,self.vector_SA),self.zoom_factor_SAX),A_3C_zoomed,self.volume_affine)
        threec_line,_,_ = ff.draw_plane_intersection(threec_line,self.vector_SA['x'],self.vector_SA['y'],ff.get_affine_from_vectors(np.zeros(self.plane_image_size),self.volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(self.center_list9[6],self.image_center,self.vector_SA),self.zoom_factor_SAX),A_3C_zoomed,self.volume_affine)
        fourc_line,_,_ = ff.draw_plane_intersection(fourc_line,self.vector_SA['x'],self.vector_SA['y'],ff.get_affine_from_vectors(np.zeros(self.plane_image_size),self.volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(self.center_list9[2],self.image_center,self.vector_SA),self.zoom_factor_SAX),A_4C_zoomed,self.volume_affine)
        fourc_line,_,_ = ff.draw_plane_intersection(fourc_line,self.vector_SA['x'],self.vector_SA['y'],ff.get_affine_from_vectors(np.zeros(self.plane_image_size),self.volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(self.center_list9[4],self.image_center,self.vector_SA),self.zoom_factor_SAX),A_4C_zoomed,self.volume_affine)
        fourc_line,_,_ = ff.draw_plane_intersection(fourc_line,self.vector_SA['x'],self.vector_SA['y'],ff.get_affine_from_vectors(np.zeros(self.plane_image_size),self.volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(self.center_list9[6],self.image_center,self.vector_SA),self.zoom_factor_SAX),A_4C_zoomed,self.volume_affine)

        # normalize by WL and WW:
        if normalize == True:
            twoc_line = ff.set_window(twoc_line,self.WL,self.WW); twoc_line = np.flip(twoc_line.T,0)
            threec_line = ff.set_window(threec_line,self.WL,self.WW); threec_line = np.flip(threec_line.T,0)
            fourc_line = ff.set_window(fourc_line,self.WL,self.WW); fourc_line = np.flip(fourc_line.T,0)

            for j in range(0,9):
                sax_w_2c[j] = ff.set_window(sax_w_2c[j],self.WL,self.WW); sax_w_2c[j] = sax_w_2c[j].T
                sax_w_3c[j] = ff.set_window(sax_w_3c[j],self.WL,self.WW); sax_w_3c[j] = sax_w_3c[j].T
                sax_w_4c[j] = ff.set_window(sax_w_4c[j],self.WL,self.WW); sax_w_4c[j] = sax_w_4c[j].T

        return twoc_line, threec_line, fourc_line, sax_w_2c, sax_w_3c, sax_w_4c

        
    # main function 
    def plane_image(self, save_path, draw_lines = True, color_box_size = [10,20]):
        # define interpolation matrix
        interpolate = ff.define_interpolation(self.volume_data,Fill_value=self.volume_data.min(),Method='linear')

        # reslice long axis
        twoc = ff.reslice_mpr(np.zeros(self.plane_image_size),self.image_center + self.vector_2C['t'],self.vector_2C['x'],self.vector_2C['y'],self.vector_2C['s'][0]/self.zoom_factor_LAX,self.vector_2C['s'][1]/self.zoom_factor_LAX,interpolate)
        threec = ff.reslice_mpr(np.zeros(self.plane_image_size),self.image_center + self.vector_3C['t'],self.vector_3C['x'],self.vector_3C['y'],self.vector_3C['s'][0]/self.zoom_factor_LAX,self.vector_3C['s'][1]/self.zoom_factor_LAX,interpolate)
        fourc = ff.reslice_mpr(np.zeros(self.plane_image_size),self.image_center + self.vector_4C['t'],self.vector_4C['x'],self.vector_4C['y'],self.vector_4C['s'][0]/self.zoom_factor_LAX,self.vector_4C['s'][1]/self.zoom_factor_LAX,interpolate)

        # reslice short axis
        sax_collection = []
        for i in range(0,9):
            sax_collection.append(ff.reslice_mpr(np.zeros(self.plane_image_size),self.center_list9[i],self.vector_SA['x'],self.vector_SA['y'],self.vector_SA['s'][0]/self.zoom_factor_SAX,self.vector_SA['s'][1]/self.zoom_factor_SAX,interpolate))

         # if need to draw lines (use lines to represent SAX/LAX planes on LAX/SAX planes)
        if draw_lines == True:
            twoc_line, threec_line, fourc_line, sax_w_2c, sax_w_3c, sax_w_4c = self.draw_lines(twoc,threec,fourc,sax_collection,True)

        # normalize by WL and WW and rotate to the correct orientation
        twoc = ff.set_window(twoc,self.WL,self.WW); twoc = np.flip(twoc.T,0)
        threec = ff.set_window(threec,self.WL,self.WW); threec = np.flip(threec.T,0)
        fourc = ff.set_window(fourc,self.WL,self.WW); fourc = np.flip(fourc.T,0)
    
        for i in range(0,9):
            sax_collection[i] = ff.set_window(sax_collection[i],self.WL,self.WW)
            sax_collection[i] = sax_collection[i].T

        
        # make plane images (3x4 images, the first column contains three LAX, the second to fourth column contains the SAX stack)
        [h,w,d] = self.plane_image_size
        I = np.zeros((h*3,w*4,3))
        if draw_lines == True:
            I[0:h,0:w,0] = ff.color_box(twoc,color_box_size[0],color_box_size[1]); I[0:h,0:w,1] = twoc_line; I[0:h,0:w,2] = twoc
            I[h:h*2,0:w,0] = threec; I[h:h*2,0:w,1] = ff.color_box(threec_line,color_box_size[0],color_box_size[1]); I[h:h*2,0:w,2] = threec
            I[h*2:h*3,0:w,0] = fourc; I[h*2:h*3,0:w,1] = fourc_line; I[h*2:h*3,0:w,2] = ff.color_box(fourc,color_box_size[0],color_box_size[1])
            I[0:h,w:w*2,0] = sax_w_2c[0]; I[0:h,w:w*2,1] = sax_w_3c[0]; I[0:h,w:w*2,2] = sax_w_4c[0]
            I[0:h,w*2:w*3,0] = sax_w_2c[1]; I[0:h,w*2:w*3,1] = sax_w_3c[1]; I[0:h,w*2:w*3,2] = sax_w_4c[1]
            I[0:h,w*3:w*4,0] = sax_w_2c[2]; I[0:h,w*3:w*4,1] = sax_w_3c[2]; I[0:h,w*3:w*4,2] = sax_w_4c[2]
            I[h:h*2,w:w*2,0] = sax_w_2c[3]; I[h:h*2,w:w*2,1] = sax_w_3c[3]; I[h:h*2,w:w*2,2] = sax_w_4c[3]
            I[h:h*2,w*2:w*3,0] = sax_w_2c[4]; I[h:h*2,w*2:w*3,1] = sax_w_3c[4]; I[h:h*2,w*2:w*3,2] = sax_w_4c[4]
            I[h:h*2,w*3:w*4,0] = sax_w_2c[5]; I[h:h*2,w*3:w*4,1] = sax_w_3c[5]; I[h:h*2,w*3:w*4,2] = sax_w_4c[5]
            I[h*2:h*3,w:w*2,0] = sax_w_2c[6]; I[h*2:h*3,w:w*2,1] = sax_w_3c[6]; I[h*2:h*3,w:w*2,2] = sax_w_4c[6]
            I[h*2:h*3,w*2:w*3,0] = sax_w_2c[7]; I[h*2:h*3,w*2:w*3,1] = sax_w_3c[7]; I[h*2:h*3,w*2:w*3,2] = sax_w_4c[7]
            I[h*2:h*3,w*3:w*4,0] = sax_w_2c[8]; I[h*2:h*3,w*3:w*4,1] = sax_w_3c[8]; I[h*2:h*3,w*3:w*4,2] = sax_w_4c[8]
        else:
            I[0:h,0:w,0] = twoc; I[0:h,0:w,1] = twoc; I[0:h,0:w,2] = twoc
            I[h:h*2,0:w,0] = threec; I[h:h*2,0:w,1] = threec; I[h:h*2,0:w,2] = threec
            I[h*2:h*3,0:w,0] = fourc; I[h*2:h*3,0:w,1] = fourc; I[h*2:h*3,0:w,2] = fourc
            I[0:h,w:w*2,0] = sax_collection[0]; I[0:h,w:w*2,1] = sax_collection[0]; I[0:h,w:w*2,2] = sax_collection[0]
            I[0:h,w*2:w*3,0] = sax_collection[1]; I[0:h,w*2:w*3,1] = sax_collection[1]; I[0:h,w*2:w*3,2] = sax_collection[1]
            I[0:h,w*3:w*4,0] = sax_collection[2]; I[0:h,w*3:w*4,1] = sax_collection[2]; I[0:h,w*3:w*4,2] = sax_collection[2]
            I[h:h*2,w:w*2,0] = sax_collection[3]; I[h:h*2,w:w*2,1] = sax_collection[3]; I[h:h*2,w:w*2,2] = sax_collection[3]
            I[h:h*2,w*2:w*3,0] = sax_collection[4]; I[h:h*2,w*2:w*3,1] = sax_collection[4]; I[h:h*2,w*2:w*3,2] = sax_collection[4]
            I[h:h*2,w*3:w*4,0] = sax_collection[5]; I[h:h*2,w*3:w*4,1] = sax_collection[5]; I[h:h*2,w*3:w*4,2] = sax_collection[5]
            I[h*2:h*3,w:w*2,0] = sax_collection[6]; I[h*2:h*3,w:w*2,1] = sax_collection[6]; I[h*2:h*3,w:w*2,2] = sax_collection[6]
            I[h*2:h*3,w*2:w*3,0] = sax_collection[7]; I[h*2:h*3,w*2:w*3,1] = sax_collection[7]; I[h*2:h*3,w*2:w*3,2] = sax_collection[7]
            I[h*2:h*3,w*3:w*4,0] = sax_collection[8]; I[h*2:h*3,w*3:w*4,1] = sax_collection[8]; I[h*2:h*3,w*3:w*4,2] = sax_collection[8]

        # save
        Image.fromarray((I * 255).astype('uint8')).save(save_path)
