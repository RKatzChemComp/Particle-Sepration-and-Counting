#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 14:03:35 2021

@author: rebeccakatz
"""


import cv2
import numpy as np
import os
import fnmatch
import sys
import re
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from skimage.measure import label
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from more_itertools import consecutive_groups
import argparse
import imutils

#initialize the dict of file to be segmented
dict_fil={}
iden=0
#np.set_printoptions(threshold=sys.maxsize)
#np.set_printoptions(threshold=sys.maxsize,linewidth=200)

#(HSV)
#H goes from 0 to 255 or 0 to 1 or 0 to 260 depending the scaling
#S goes from 0:white to 255:full color saturation
#V goes from 0:black to 255:full color value

def hsv_hue256_to_huedeg(hsv):
    hsv=hsv.astype(float)
    hsv[:,0]=hsv[:,0]*float(360/255)
    return hsv

def hsv_huedeg2hue255(huedeg):
    return huedeg*(255/360)

def hsv_hue255_2_huedeg(hue255):
    return hue255*(360/255)

def color_decomp_byhsv(filename):
    
    outdata = []
    
    print(f'color_decomp_byhsv Analysis\n')
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    outdata.append(filename)
    outdata.append(img)
    process(filename)
    
    print(f'Shape of image: {np.shape(img)}\n')
    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    img1=np.reshape(img1.flatten(),(-1,3))
    
    #scale and save 2D color arrays
    img1_hsvscaled=img1/255.0
    np.savetxt(f'img_hsv_scalled_array.txt', img1_hsvscaled)
    hsv_degscalled=hsv_hue256_to_huedeg(img1)
    np.savetxt(f'img_hsv_huedeg_array.txt', hsv_degscalled)
    np.savetxt(f'img_rgb_array.txt', matplotlib.colors.hsv_to_rgb(img1_hsvscaled))
    
    #Make scatter plot
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.scatter(img1[:,0], img1[:,1], img1[:,2], label='hsv', s=5, c=matplotlib.colors.hsv_to_rgb(img1_hsvscaled))
    #ax.legend()
    #ax.set_xlabel('H')
    #ax.set_ylabel('S')
    #ax.set_zlabel('V')
    #plt.show()
    
    uimg1, uimg1_count=np.unique(img1,axis=0,return_counts=True)
    
    #remove black HSV (V=0)
    #uimg1_count=uimg1_count[np.where(uimg1[:,2] != 0)]  
    #uimg1=uimg1[np.where(uimg1[:,2] != 0)]
    img1 = img1[np.where(img1[:,2] != 0)]
    print(f'Total number of nonblack pixels: {np.sum(uimg1_count)}')

    #calculate percent decomposition of each color region and create 
    # a labels array 
    # it is possible the sum of all decompsoition percentages != 1 
    # if not all pixels fall in the specified regions
    img1_copy = img1
    print(img1_copy.shape, img1_copy[:10])
    hsvr_labels_array = np.zeros(len(img1_copy))
    for hsvr_idx,hsvr in enumerate(hsvcolor_ranges_array):
        hsvr_idx += 1
        print(f'\nhsvr_idx: {hsvr_idx}')
        print(f'H: {hsv_hue255_2_huedeg(hsvr[0])}')
        print(f'S: {hsvr[1]}') 
        print(f'V: {hsvr[2]}')
        
        if hsvr[0,1] > hsvr[0,0]:   
            
            hsvr_labels_array = np.where(\
                                         ((img1_copy[:,0] >= hsvr[0,0]) & (img1_copy[:,0] <= hsvr[0,1]))\
                                             & ((img1_copy[:,1] >= hsvr[1,0]) & (img1_copy[:,1] <= hsvr[1,1]))\
                                                 & ((img1_copy[:,2] >= hsvr[2,0]) & (img1_copy[:,2] <= hsvr[2,1])), hsvr_idx, hsvr_labels_array)
            
        elif  hsvr[0,1] < hsvr[0,0]:
           
            hsvr_labels_array = np.where(\
                                         ((img1_copy[:,0] >= hsvr[0,0]) | (img1_copy[:,0] <= hsvr[0,1]))\
                                             & ((img1_copy[:,1] >= hsvr[1,0]) & (img1_copy[:,1] <= hsvr[1,1]))\
                                                 & ((img1_copy[:,2] >= hsvr[2,0]) & (img1_copy[:,2] <= hsvr[2,1])), hsvr_idx, hsvr_labels_array)
           
        
        print(f'Percent composition: {len(hsvr_labels_array[np.where(hsvr_labels_array == hsvr_idx)])/len(img1_copy)}')
    outdata.append(hsvr_labels_array)
    print(hsvr_labels_array.shape)
    
    return outdata

def process(filename: str=None) -> None:
    """
    View multiple images stored in files, stacking vertically

    Arguments:
        filename: str - path to filename containing image
    """
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # <something gets done here>
    plt.figure()
    plt.imshow(image)


def gen_part_ID(list_of_part_tags):
    
    leader_dict = {}
    leader_count = 0
    for idx,in_list in enumerate(list_of_part_tags):
        
        
        array = in_list[-1]
        img = in_list[1]
        filename = in_list[0]
        #process(filename)
        
        outdata = []

def nearest_square(limit):
    answer = 0
    while (answer+1)**2 < limit:
        answer += 1
    answer += 1
    return answer**2

def sort_contours(cnt_dict, method="left-to-right"):
    #print(cnt_dict)
    cnts = [c[2] for c in cnt_dict.values()]
    hues = [c[1][0] for c in cnt_dict.values()]
    #print(cnts)
    #print(hues)

    
	# initialize the reverse flag and sort index
    reverse = False
    i = 0
	# handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom 
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, hue, boundingBoxes) = zip(*sorted(zip(cnts, hues, boundingBoxes), key=lambda b:b[2][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
    return (cnts, hue, boundingBoxes) 

def color_decomp_bycountour(filename):
    print(f'color_decomp_bykmeans Analysis {filename}\n')
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #process(filename)
    
    print(f'Shape of image: {np.shape(img)}\n')
    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    print(img1.shape)
    
    plt.imshow(img)
    plt.show()
    
    img1_height=np.shape(img1)[0]
    img1_width=np.shape(img1)[1]
    

    
    black_img = img1.copy()
    black_img[:] = (0,0,0)
    black_img = black_img.reshape(img1_height,img1_width,3).astype(np.uint8)
    plt.imshow(black_img)
    
    #find rectangles for each color
    area_list = []
    cnt_list = []
    cnt_area_dict = {}
    dict_idx = 0
    for hsvr_idx,hsvr in enumerate(hsvcolor_ranges_array):
        hsvr_idx += 1
        
      
        #hsvr[0] = hsv_hue255_2_huedeg(hsvr[0])
        #print(f'\nhsvr_idx: {hsvr_idx}')
        #print(f'H: {hsvr[0]}')
        #print(f'S: {hsvr[1]}') 
        #print(f'V: {hsvr[2]}')
    
        area = 0.0
        if hsvr[0][0] > hsvr[0][1]:
            #make two masks and combine
            lower_bound_1 = np.asarray([0,50,50]).astype(np.uint8)
            upper_bound_1 = hsvr[:,1].astype(np.uint8)
            #print(lower_bound_1,upper_bound_1)
            mask1 = cv2.inRange(img1, lower_bound_1, upper_bound_1)
            
            lower_bound_2 = hsvr[:,0].astype(np.uint8)
            upper_bound_2 = np.asarray([179,255,255]).astype(np.uint8)
            #print(lower_bound_2,upper_bound_2)
            mask2 = cv2.inRange(img1, lower_bound_2, upper_bound_2)
            
            mask = mask1 + mask2
            
            box_color = np.asarray([0,255,255]).astype(np.uint8)
            
        else:
            lower_bound = hsvr[:,0].astype(np.uint8)
            upper_bound = hsvr[:,1].astype(np.uint8)

            #print(f'lower_bound: {lower_bound}, upper_bound: {upper_bound}')
        
            mask = cv2.inRange(img1, lower_bound, upper_bound)
            
            box_color = (upper_bound - lower_bound)/2 + lower_bound
        
        
        cnt = cv2.findContours(mask.copy(),
                              cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        cnt_list.append(cnt)
        
        if len(cnt) > 0:
            for c in cnt:
                #print(c)
                area += cv2.contourArea(c)
                cnt_area_dict[dict_idx] = [c,cv2.contourArea(c),hsvr]
                dict_idx += 1
            
        #print(f'ContourArea: {area}')
        area_list.append(area)
        
        
        #print(len(cnt))
        
        box_color = box_color.astype(np.uint8)
        #print(box_color, box_color.shape)
        box_color[1:3] = 255
        #print(box_color, box_color.shape)
        box_color = cv2.cvtColor(box_color.reshape((1,1,3)), cv2.COLOR_HSV2RGB)
        #print(box_color, box_color.shape)
        
        if len(cnt) > 0:
            cv2.drawContours(black_img,cnt,-1,(int(box_color[0,0,0]),int(box_color[0,0,1]),int(box_color[0,0,2])),1)
            
            max_cnt = max(cnt, key=cv2.contourArea)
            #print(max_cnt)
    
    
    #calculate total area
    total_area = np.sum(area_list)
    print(f'Total_area: {total_area}')
    
    for hsvr,area,cnt in zip(hsvcolor_ranges_array,area_list,cnt_list):
        print(f'H: {hsvr[0]}')
        print(f'S: {hsvr[1]}') 
        print(f'V: {hsvr[2]}')
        print(f'number of contours: {len(cnt)}')
        percent_area = area/total_area
        print(f'percent_area: {percent_area}')
    


    #find set of contours that meet area cutoff
    #print(len(cnt_area_dict))
    cutoff = 0.1
    cnt_area_dict_meets_cutoff = {}
    for k,v in cnt_area_dict.items():
        #print(k,v[1]/total_area,v[2][0])
        
        if v[1]/total_area > cutoff:
            cnt_area_dict_meets_cutoff[k] = [v[1]/total_area,v[2],v[0]]
        
            print(k,v[1]/total_area, v[2][0])
    
    
    plt.imshow(black_img)
    plt.show()
    
    sorted_cnts = sort_contours(cnt_area_dict_meets_cutoff, method="top-to-bottom")
    print(sorted_cnts[1])
    return(filename,sorted_cnts[1])
    
def sort_results(results):
    #print(results)
    leader_dict = {}
    leader_id = 0
    for i,r in enumerate(results):
       # print()
        #print(i,r)
        if i == 0:
            leader_dict[leader_id] = [r]
        #print(leader_dict)
        
        if i > 0:
            sub_array = np.vstack(r[1])
            leader_keys = list(leader_dict.keys())
            #print(leader_keys)
            for key_id,leader_key in enumerate(leader_keys):
                leader_array  = np.vstack(leader_dict[leader_key][0][1])
                #print(leader_dict)
                #print(sub_array)
                #print(leader_key,leader_array)
                
                if np.array_equal(sub_array,leader_array):
                    leader_dict[leader_key] += [r]
                    break
                elif np.array_equal(np.flip(sub_array, axis=0), leader_array):
                    leader_dict[leader_key] += [r]
                    break
                elif key_id == len(leader_keys)-1:
                    leader_id += 1
                    leader_dict[leader_id] = [r]
                    break
                

    return leader_dict
### MAIN ###
hsvcolor_ranges=[x.strip('\n') for x in open(sys.argv[1], 'r').readlines()]
print(hsvcolor_ranges)
hsvcolor_ranges_array=np.asarray([[x.split('-') for x in hsvr] for hsvr in [x.split(',') for x in hsvcolor_ranges]]).astype(float)

#hsvcolor_ranges_array[:,0,:]=hsv_huedeg2hue255(hsvcolor_ranges_array[:,0,:])
print(hsvcolor_ranges_array[:,0,:])
        
#explore the current directory 
for root, dirs, files in os.walk(".", topdown=False):
   for name in files:
       
       if re.match('.+png$', name):
           print(name)
           iden +=1
           dict_fil[iden]={}
           dict_fil[iden]['parent_dir']=root.split(sep="\\")[-1]
           dict_fil[iden]['path']=os.path.join(root,name)
           for file in os.listdir(root):
               if fnmatch.fnmatch(file, '*cat*'):                            
                   dict_fil[iden]['path_color']=os.path.join(root,file)

#do color decomposition for all files and generate label arrays
#label_arrays_for_imgs = [color_decomp_byhsv(dict_fil[key]['path']) for key in dict_fil.keys()]

#print(label_arrays_for_imgs)
#gen_part_ID(label_arrays_for_imgs)
#results = [color_decomp_bykmeans(dict_fil[key]['path']) for key in dict_fil.keys()]
results = [color_decomp_bycountour(dict_fil[key]['path']) for key in dict_fil.keys()]
#for r in results:
    #print(r)
sorted_results = sort_results(results)
for k,v in sorted_results.items():
    print()
    print(k,v, len(v))
#results = [color_decomp_byDBSCAN(dict_fil[key]['path']) for key in dict_fil.keys()]
#results = [color_decomp2(dict_fil[key]['path']) for key in dict_fil.keys()]

