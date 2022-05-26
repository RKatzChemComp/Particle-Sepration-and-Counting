# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:25:32 2020

@author: rkatz
"""

from matplotlib import pyplot as plt
from skimage.segmentation import clear_border
import cv2
import numpy as np
import os
import fnmatch
import sys



def seg_water(filename):
    
    img1 = cv2.imread(filename)
    #plt.imshow(img1)
    crop_img = img1[0:400, :]    
    img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)    
    blur = cv2.bilateralFilter(img,1,10,10)
    #plt.imshow(blur)
    #plt.show()
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    
    
    #Watershed Segmentation     
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations= 3)
    #plt.imshow(thresh)
    #plt.show()
    
    #throwout particles on edges 
    opening = clear_border(opening)
    
    #find background and particles
    sure_bg = cv2.dilate(opening, kernel, iterations= 1)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    #plt.imshow(sure_bg)
    #plt.show()
    
  
    
    ret2, sure_part = cv2.threshold(dist_transform, 0.1*dist_transform.max(), 255, 0)
    sure_part = np.uint8(sure_part)
    #plt.imshow(sure_part)
    #plt.show()
    unk = cv2.subtract(sure_bg, sure_part)
    #plt.imshow(unk)
    #plt.show()
    ret3, markers = cv2.connectedComponents(sure_part)    
    markers = markers+10  
    markers[unk==255] = 0
    plt.imshow(markers)
    plt.show()
    
    #watershead segmentation 
    markers = cv2.watershed(crop_img, markers)
    return markers


def rotate_bound(image, center, angle,contour):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (center[0],center[1])   
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # compute the coordinates of the center
    center=np.hstack([center,np.ones(1)])
    center_rotated=M.dot(center)
    #rotate contour points and extract the bounding boxes
    cont_new=[]
    bb={}
    

    for cnt in contour[:,0]:
        
        #sys.exit()
        b=np.hstack([cnt[0], cnt[1],np.ones(1)])
        
        cont_new.append(M.dot(b))
    cont_new=np.array(cont_new)
    bb['ymin'],bb['ymax']=int(cont_new[:,0].min())-1,int(cont_new[:,0].max())+1
    bb['xmin'],bb['xmax']=int(cont_new[:,1].min())-1,int(cont_new[:,1].max())+1
            
    # perform the actual rotation and return the i mage
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR),center_rotated, bb

list_imt=[]
list_or=[]
list_ms=[]
def save_ind(markers,im_cation,folder):
    img1 = cv2.imread(im_cation)
    plt.imshow(img1)
    
#    img2=np.swapaxes(img1,0,1)
    img2 = img1[0:400,:,:]
    plt.imshow(img2)
    
    count=0
    print(f'Number of unique markers {len(np.unique(markers))}')
    
    for i in np.unique(markers):
        
        
        if i <= 10 :
            pass
        else :
            print(f'Saving particle: {i}')
            markers1=np.where(markers == i ,0,1)
            markers1 = markers1.astype(np.uint8)
            
            markers2=np.where(markers == i ,1,0)
            markers2 = markers2.astype(np.uint8)    
            
#            print('image3{}'.format(img3.shape))
            mask=np.empty_like(img2)
            for j in range(3):
                mask[:,:,j]=markers2[:,:]
#            img1=img2[~mask.astype(bool)]
#            img1 = img1.astype(np.uint8)
#            list_or.append(markers1)
#            mask=np.ones_like(img2)
#            img1=mask(markers1)
#            for j in range(3):
#                mask[:,:,j]=np.where(markers1 == 0 ,1,0)
 #               mask[:,:,j]=markers1
 #           list_ms.append(mask)
            img3=np.multiply(img2,mask) 
            plt.imshow(img3)
            #plt.imshow(markers1)
            list_imt.append(img1)
            
            
            cont_ret, h= cv2.findContours(markers1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            #print (cont_ret[1])
            
            #print(cont_ret[1].shape)
            
            cont_ret = cont_ret[1]
            
            if len(cont_ret) <= 5:
                
                pass
            else:
                
                e=cv2.fitEllipse(cont_ret)
                particle, center_rotated,bb=rotate_bound(markers1,e[0],e[2],cont_ret)
                
                img4,_,_=rotate_bound(img3,e[0],e[2],cont_ret)

                count += 1
                particle2=particle[bb['xmin']:bb['xmax'],bb['ymin']:bb['ymax']]
                
                img4=img4[bb['xmin']:bb['xmax'],bb['ymin']:bb['ymax']]
 
                img = cv2.convertScaleAbs(particle2, alpha=(255.0))
    #            img3 = cv2.convertScaleAbs(img3, alpha=(255.0))
                #cv2.imwrite(os.path.join('Data','{}_{}.png'.format(folder,count)),img)
                #cv2.imwrite(os.path.join('Data','im{}_{}.png'.format(folder,count)),img4)
                print(f'{img.shape}')
                #img = cv2.resize(img, (1000,1000))
                #img4 = cv2.resize(img4, (200,200))
                print(f'{img.shape}')
                cv2.imwrite(f'./{count}_{filename}.png',img4)
                #cv2.imwrite(f'./{count}_cutout_img.png',img)
    



#initialize the dict of file to be segmented
dict_fil={}
iden=0



filename = 'Demo.jpg'
print(f'filename: {filename}')
markers = seg_water(filename)  
print(markers, markers.shape)         
save_ind(markers,filename,'.')
    


#fig, axes = plt.subplots(nrows=1,ncols=2,sharex=False,sharey=False,figsize=(10,10))
#axes[0].imshow(img1
##                ,cmap='gray',origin='lower')
#axes[1].imshow(markers
#                ,cmap='gray',origin='lower')


