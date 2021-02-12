#!/usr/bin/env python

# this script saved most of functions widely used in other scripts
import numpy as np
import math
import glob as gb
import glob
import os
from scipy.interpolate import RegularGridInterpolator
import nibabel as nib
from nibabel.affines import apply_affine
import math
import xlsxwriter as xl
import string
import matplotlib.pyplot as plt
import cv2
import pandas as pd


# function: make folders
def make_folder(folder_list):
    for i in folder_list:
        os.makedirs(i,exist_ok = True)

# function: find all files under the name * in the main folder, put them into a file list
def find_all_target_files(target_file_name,main_folder):
    F = np.array([])
    for i in target_file_name:
        f = np.array(sorted(gb.glob(os.path.join(main_folder, os.path.normpath(i)))))
        F = np.concatenate((F,f))
    return F

# function: convert coordinates to different coordinate system by affine matrix                             
def convert_coordinates(target_affine, initial_affine, r):
    affine_multiply = np.linalg.inv(target_affine).dot(initial_affine)
    return apply_affine(affine_multiply,r)

# function: normalize one vector
def normalize(x):
    x_scale = np.linalg.norm(x)
    if x_scale == 0:
        return np.array([0,0,0])
    else:
        return np.asarray([i/x_scale for i in x])

# function: get length of one vector and angle between two vectors
def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    rad=math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
    result = rad / math.pi * 180
    return result

# function: calculate orientation error:
def orientation_error(x_truth,y_truth,x_pred,y_pred):
    n_truth = normalize(np.cross(x_truth,y_truth))
    n_pred = normalize(np.cross(x_pred,y_pred))

    error =  angle(n_truth,n_pred)
    if error > 180:
        error = error - 180
    return error

# function: project one vector onto a plane with known normal vectors
def project_onto_plane(u,n):
    '''n is the normal vector of the plane'''
    n = normalize(n)
    return (u - dotproduct(u,n) * n)

# function: only pick one time frame from each patient
def one_time_frame_per_patient(x):
  '''only pick one time frame for each patient'''
  for i in range(len(x)):
    if i%2 == 1:
      x[i]='0'
  return x[x!='0']

# function: turn normalized vector into pixel unit
def turn_to_pixel(vec,size=[160,160,96]):
    t=vec.reshape(3,).tolist()
    result = [t[i]*size[i]/2 for i in range(0,3)]
    return np.array(result)

# function: define the interpolation
def define_interpolation(data,Fill_value=0,Method='linear'):
    shape = data.shape
    [x,y,z] = [np.linspace(0,shape[0]-1,shape[0]),np.linspace(0,shape[1]-1,shape[1]),np.linspace(0,shape[-1]-1,shape[-1])]
    interpolation = RegularGridInterpolator((x,y,z),data,method=Method,bounds_error=False,fill_value=Fill_value)
    return interpolation

# function: reslice a mpr
def reslice_mpr(mpr_data,plane_center,x,y,x_s,y_s,interpolation):
    # plane_center is the center of a plane in the coordinate of the whole volume
    mpr_shape = mpr_data.shape
    new_mpr=[]
    centerpoint = np.array([(mpr_shape[0]-1)/2,(mpr_shape[1]-1)/2,0])
    for i in range(0,mpr_shape[0]):
        for j in range(0,mpr_shape[1]):
            delta = np.array([i,j,0])-centerpoint
            v = plane_center + (x*x_s)*delta[0]+(y*y_s)*delta[1]
            new_mpr.append(v)
    new_mpr=interpolation(new_mpr).reshape(mpr_shape)
    return new_mpr

# function: extract vectors from numpy file
def get_ground_truth_vectors(filename):
    a = np.load(os.path.join(filename),allow_pickle=True)
    [t,x,y,s,img_center] = [a[0],a[2],a[3],a[6],a[5]]
    result = {'t':t,'x':x,'y':y,'s':s,'img_center':img_center}
    return result

def get_predicted_vectors(file_t,file_r,scale,image_center):
    f1 = np.load(os.path.join(file_t),allow_pickle=True)
    f2 = np.load(os.path.join(file_r),allow_pickle=True)
    t = turn_to_pixel(f1[0])
    [x,y] = [f2[1],f2[-1]]
    result = {'t':t,'x':x,'y':y,'s':scale,'img_center':image_center}
    return result

# function: find the matrix of vectors for any plane in SA stack based on the basal vectors
def make_matrix_for_any_plane_in_SAX_stack(plane_center,image_center,basal_vectors):
    result = {'t':plane_center - image_center,'x':basal_vectors['x'],'y':basal_vectors['y'],'s':basal_vectors['s'],'img_center':basal_vectors['img_center']}
    return result

# function: get pixel dimensions
def get_voxel_size(nii_file_name):
    ii = nib.load(nii_file_name)
    h = ii.header
    return h.get_zooms()

# function: find the number of slices above the basal plane and the number of slices lower the basal plane for SAX stack. the stack starts from 2 planes before where
# LV segmentation starts and ends two planes after where LV segmentaion ends. 
def find_num_of_slices_in_SAX(mpr_data,image_center,t_m,x_m,y_m,seg_m_data,normal_vector_flip = 0, pixel_dimension = 2.59):
    ''' returned a is the number of planes upon the basal plane, returned b is the number of planes below the basal plane'''
    '''2.59 = SQRT((1.5^2)*3)'''
    if normal_vector_flip == 0:
        n_m =  normalize(np.cross(x_m,y_m))
    else:
        n_m = -normalize(np.cross(x_m,y_m))
        
    test_a = 1
    a_manual = 0
    while test_a == True:
        a_manual += 1
        plane = reslice_mpr(mpr_data,(image_center + t_m + (-n_m) * 8 * a_manual / pixel_dimension),x_m,y_m,1,1,define_interpolation(seg_m_data,Fill_value=0,Method='nearest'))
        test_a = (1.0 in plane)
    a_manual += 1

    test_b = 1
    b_manual = 2
    while test_b == True:
        b_manual += 1
        plane = reslice_mpr(mpr_data,(image_center + t_m + (n_m) * 8 * b_manual / pixel_dimension),x_m,y_m,1,1,define_interpolation(seg_m_data,Fill_value=0,Method='nearest'))
        test_b = (1.0 in plane)
        if test_b == False:
            # in case there is a gap that is not the end of LV, we keep searching with two slices away
            c = b_manual + 1
            plane1 = reslice_mpr(mpr_data,(image_center + t_m + (n_m) * 8 * c / pixel_dimension),x_m,y_m,1,1,define_interpolation(seg_m_data,Fill_value=0,Method='nearest'))
            
            cc = b_manual + 2
            plane2 = reslice_mpr(mpr_data,(image_center + t_m + (n_m) * 8 * cc / pixel_dimension),x_m,y_m,1,1,define_interpolation(seg_m_data,Fill_value=0,Method='nearest'))
            
            t1 = (1.0 in plane1);t2 = (1.0 in plane2)
            if t2 == True:
                b_manual = cc; test_b = 1
            if (t2 == False) and (t1 == True):
                b_manual = c; test_b = 1
            if (t1 == False) and (t2 == False):
                test_b = 0  
    b_manual += 1
    return a_manual,b_manual

# function: find a list of all center coordinate given start point and number of planes for SA stack
def find_center_list(start_center,n,num_of_plane,slice_thickness,pixel_dimension = 2.59):
    '''2.59 = SQRT((1.5^2)*3)'''
    # n is normal vector
    center_list = start_center.reshape(1,3)
    for i in range(1,num_of_plane):
        c = start_center + n * slice_thickness * (i) / pixel_dimension
        center_list = np.concatenate((center_list,c.reshape(1,3)))
    return center_list

# find center list for the whole stack (refer to function find_num_of_slices_in_SAX)
def find_center_list_whole_stack(start_center,n,num_a,num_b,slice_thickness,pixel_dimension = 2.59):
    # n is normal vector
    center_list = start_center.reshape(1,3)
    for i in range(1,num_a + 1):
        c = start_center + -n * slice_thickness * (i) / pixel_dimension
        center_list = np.concatenate((c.reshape(1,3),center_list))
    for i in range(1,num_b + 1):
        c = start_center + n * slice_thickness * (i) / pixel_dimension
        center_list = np.concatenate((center_list,c.reshape(1,3)))
    return center_list


# funciton: resample a SAX stack with n planes into a particular num of planes
# will return the indexes as well as the centerpoint_list
def resample_SAX_stack_into_particular_num_of_planes(range_of_index,num_of_planes,center_list):
    # range of index is a list in the form of range(start,end+1)
    gap = (range_of_index[-1] - range_of_index[0] ) / (num_of_planes - 1)
    if gap >= 1:
        index_list = []
        center_list_resample = []
        for i in range(0,num_of_planes):
            index = math.floor(range_of_index[0] + gap * i)
            index_list.append(index)
            center_list_resample.append(center_list[index,:])
        return index_list,center_list_resample,gap
    else: 
        return 0,0,gap

    
# function: check affine from all time frames (affine may have errors in some tf, that's why we need to find the mode )
def check_affine(one_time_frame_file_name):
    """this function uses the affine with each element as the mode in all time frames"""
    joinpath = os.path.join(os.path.dirname(one_time_frame_file_name),'*.nii.gz')
    f = np.array(sorted(glob.glob(joinpath)))
    a = np.zeros((4,4,len(f)))
    count = 0
    for i in f:
        i = nib.load(i)
        a[:,:,count] = i.affine
        count += 1
    mm =nib.load(f[0])
    result = np.zeros((4,4))
    for ii in range(0,mm.affine.shape[0]):
        for jj in range(0,mm.affine.shape[1]):
            l = []
            for c in range(0,len(f)):
                l.append(a[ii,jj,c])
            result[ii,jj] = max(set(l),key=l.count)
    return result


# function: get affine matrix from translation,x,y and scale
def get_affine_from_vectors(mpr_data,volume_affine,vector):
    # it answers one important question: what's [1 1 1] in the coordinate system of predicted plane in that
    # of the whole CT volume
    [t,x,y,s,i_center] = [vector['t'],vector['x'],vector['y'],[1,1,0.67],vector['img_center']]
    shape = mpr_data.shape
    mpr_center=np.array([(shape[0]-1)/2,(shape[1]-1)/2,0])
    Transform = np.ones((4,4))
    xx = normalize(x)*s[0]
    yy = normalize(y)*s[1]
    zz = normalize(np.cross(x,y))*s[-1]
    Transform[0:3,0] = xx
    Transform[0:3,1] = yy
    Transform[0:3,2] = zz
    t_o = (i_center + t) - (mpr_center[0]*xx + mpr_center[1]*yy + mpr_center[2]*zz)
    Transform[0:3,3] = t_o
    Transform[3,:] = np.array([0,0,0,1])
    mpr_A = np.dot(volume_affine,Transform)
    return mpr_A


# function: color box addition
def color_box(image,y_range = 10, x_range = 20):
    [sx,sy] = [image.shape[0],image.shape[1]]
    new_image = np.ones((sx,sy))
    for i in range(sx):
        for j in range(sy):
            new_image[i,j] = image[i,j]
    for j in range(sy-y_range,sy):
        for i in range(sx-x_range,sx):
            new_image[i,j] = new_image.max()
    return new_image

# function: draw an arbitrary axis on one image 
def draw_arbitrary_axis(image,axis,start_point,length = 500):
    '''length defines how long the axis we want to draw'''
    assert abs(axis[-1]) == 0.0 
    assert abs(start_point[-1]) == 0.0
    
    result = np.zeros((image.shape[0],image.shape[1],1))
    for ii in range(0,image.shape[0]):
        for jj in range(0,image.shape[1]):
            
            result[ii,jj,0] = image[ii,jj,0]

    axis = normalize(axis)
  
    for i in range(-length,length):
        [x,y] = [start_point[0] + axis[0]*i , start_point[1] + axis[1]*i]
        if x>=0 and y>=0 and x<result.shape[0] and y<result.shape[1]:
            
            result[int(x),int(y),0] = result.max()
    return result

# function: draw the intersection of two mpr planes on one plane (draw the intersection of plane 1 and 2 on plane 2)
def draw_plane_intersection(plane2_image,plane1_x,plane1_y,plane1_affine,plane2_affine,volume_affine):
    '''plane 2 is the plane in which we want to draw axis'''
    real_x = convert_coordinates(plane2_affine,volume_affine,np.array([1,1,1]+plane1_x)) - convert_coordinates(plane2_affine,volume_affine,np.array([1,1,1]))
    real_y = convert_coordinates(plane2_affine,volume_affine,np.array([1,1,1]+plane1_y)) - convert_coordinates(plane2_affine,volume_affine,np.array([1,1,1]))
    
    n1 = np.cross(real_x,real_y)
    n2 = np.array([0,0,1])
    intersect_direct = (0.5/(np.cross(n1,n2)[0])) * np.cross(n1,n2)
    
    # find one point in the intersection line
    # a plane is defined as <a,b,c> . <x-x0,y-y0,z-z0> = 0
    # ax+by+cz+(-ax0-by0-cz0) = ax+by+cz+d = 0
    # let's find one (x0,y0,z0) on the 2C plane
    
    plane1_p = convert_coordinates(plane2_affine,plane1_affine,np.array([150,100,0]))
    p = convert_coordinates(plane2_affine,plane1_affine,np.array([40,40,0]))
    d1 = -n1[0]*plane1_p[0] - n1[1]*plane1_p[1] - n1[2]*plane1_p[2]   
    d2 = 0
    u = np.cross(n1,n2)
    u_length = math.sqrt(u[0]**2+u[1]**2+u[2]**2)
    intersect_point = np.cross((d2*n1-d1*n2),u)/(u_length**2)
    result_line = draw_arbitrary_axis(plane2_image,intersect_direct,intersect_point)
    return result_line,intersect_direct,intersect_point

# function: count pixel in the image/segmentatio that belongs to one label
def count_pixel(seg,target_val):
    index_list = np.where(seg == target_val)
    count = index_list[0].shape[0]
    pixels = []
    for i in range(0,count):
        p = []
        for j in range(0,len(index_list)):
            p.append(index_list[j][i])
        pixels.append(p)
    return count,pixels

# function: DICE calculation
def DICE(seg1,seg2,target_val):
    p1_n,p1 = count_pixel(seg1,target_val)
    p2_n,p2 = count_pixel(seg2,target_val)
    p1_set = set([tuple(x) for x in p1])
    p2_set = set([tuple(x) for x in p2])
    I_set = np.array([x for x in p1_set & p2_set])
    I = I_set.shape[0] 
    DSC = (2 * I)/ (p1_n+p2_n)
    return DSC


# function: find time frame of a file
# function: find time frame of a file
def find_timeframe(file,num_of_dots,signal = '/'):
    k = list(file)
    if num_of_dots == 1: #.png
        num1 = [i for i, e in enumerate(k) if e == '.'][-1]
    else:
        num1 = [i for i, e in enumerate(k) if e == '.'][-2]
    num2 = [i for i,e in enumerate(k) if e==signal][-1]
    kk=k[num2+1:num1]
    if len(kk)>1:
        return int(kk[0])*10+int(kk[1])
    else: 
        return int(kk[0])

# function: sort files based on their time frames
def sort_timeframe(files,num_of_dots,signal = '/'):
    time=[]
    time_s=[]
    for i in files:
        a = find_timeframe(i,num_of_dots,signal)
        time.append(a)
        time_s.append(a)
    time_s.sort()
    new_files=[]
    for i in range(0,len(time_s)):
        j = time.index(time_s[i])
        new_files.append(files[j])
    new_files = np.asarray(new_files)
    return new_files

# function: set window level and width
def set_window(image,level,width):
    if len(image.shape) == 3:
        image = image.reshape(image.shape[0],image.shape[1])
    new = np.copy(image)
    high = level + width
    low = level - width
    # normalize
    unit = (1-0) / (width*2)
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            if image[i,j] > high:
                image[i,j] = high
            if image[i,j] < low:
                image[i,j] = low
            norm = (image[i,j] - (low)) * unit
            new[i,j] = norm
    return new

# function: adapt the re-slicing vector from low resolution for native resolution:
def adapt_reslice_vector_for_native_resolution(vector,volume_file_low,volume_file_native):
    A_low = check_affine(volume_file_low)
    A_native = check_affine(volume_file_native)
    dim_low = get_voxel_size(volume_file_low)
    dim_native = get_voxel_size(volume_file_native) 
    # adapt direction vectors:
    vector['x'] = convert_coordinates(A_native,A_low,vector['x']) - convert_coordinates(A_native,A_low,[0,0,0])
    vector['y'] = convert_coordinates(A_native,A_low,vector['y']) - convert_coordinates(A_native,A_low,[0,0,0])

    vector['s'] = np.array([length(vector['x']),length(vector['y'])])
    # also find the scale of vectors that can be used to reslice on orignial CT volume
    vector['final_s'] = np.array([vector['s'][0]/dim_low[0]*dim_native[0], vector['s'][1]/dim_low[1]*dim_native[1]])

    vector['x'] = normalize(vector['x'])
    vector['y'] = normalize(vector['y'])
    # adapt translation vector
    t_low = vector['t']
    vector['t'] = np.asarray([t_low[i] * dim_low[i] / dim_native[i] for i in range(0,t_low.shape[0])])
    return vector


# function: set scale for plane re-slicing in the case in which x and y scale are not the same (=1 for both)
# the value of scale_x and scale_y doesn't matter since it just influences the zoom of the image
# what matters is the ratio of scale_x and y
def set_scale_for_unequal_x_and_y(vector,zoom_factor = 1):
    if vector['s'][0] >= vector['s'][1]:
        return np.array([1.0/zoom_factor,1.0/vector['s'][0]*vector['s'][1]/zoom_factor])
    else:
        return np.array([1.0/vector['s'][1]*vector['s'][0]/zoom_factor,1.0/zoom_factor])

# function: make movies of several .png files
def make_movies(save_path,pngs,fps):
    mpr_array=[]
    i = cv2.imread(pngs[0])
    h,w,l = i.shape

    for j in pngs:
        img = cv2.imread(j)
        mpr_array.append(img)

    # save movies
    out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'mp4v'),10,(w,h))
    for j in range(len(mpr_array)):
        out.write(mpr_array[j])
    out.release()




    
    


    


    