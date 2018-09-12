# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from skimage import io, exposure, feature
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage import io#, exposure
import matplotlib.pyplot as plt
#rom train import train
import os
import scipy
import math
import operator
import random
import scipy.ndimage as ndi
from scipy.ndimage.filters import gaussian_filter

def my_gaussian_filter(sz):
    fltr = np.zeros((sz,sz))
    delta = sz/2
    sigma=0.2
    for ii in range(0,sz):
        for jj in range(0,sz):
            x = ii-delta
            y = jj-delta
            fltr[ii][jj] = (1/(2*3.14*sigma))*np.exp((-x*x-y*y)/2*sigma*sigma)
    return fltr

def get_random_window_from_texture(img, ws):
    return (np.random.randint(img.shape[0] - ws),
            np.random.randint(img.shape[1] - ws))
    
def start_the_img(input_img, ws):
#    rndm1 = np.random.randint(img.shape[0] - ws)
#    rndm2 = np.random.randint(img.shape[1] - ws)
    #w = np.zeros((ws,ws))
    m = np.zeros((200,200))
    for ii in range(0,input_img.shape[0]):
        for jj in range(0,input_img.shape[1]):
            #w[ii,jj] = input_img[rndm1+ii,rndm2+jj]
            m[ii,jj] = input_img[ii,jj]
#    return m
    
#    rndm1 = np.random.randint(img.shape[0] - ws)
#    rndm2 = np.random.randint(img.shape[1] - ws)
#    random_part = input_img[rndm1:rndm1+ws,rndm2:rndm2+ws]
##    pad0 = (200-seed)/2 + padding
##    pad1 = (200-seed)/2 + padding
#    npad = ((pad0,pad0), (pad1,pad1))
    return np.pad(m,pad_width=20, mode='constant', constant_values=0)


def find_pixel_value(x, y, i_img, o_img, ws):
    a = np.ones((ws,ws))
    leap = ws/2
    pxl_area = o_img[x-leap:x+leap+1, y-leap:y+leap+1]
    all_neighbor_distances = {}
    all_pixels = {}
    #print ws
    for ii in range(0,i_img.shape[0]-ws):
        for jj in range(0,i_img.shape[1]-ws):
            #print ii,jj
            patch_area = i_img[ii:ii+ws, jj:jj + ws]
            if (pxl_area.shape[0] == patch_area.shape[0]) and ( pxl_area.shape[1] == patch_area.shape[1]):
                mask = np.copy(pxl_area)
                mask[mask>0]=1
                patch_area = patch_area*mask
                diff_area = pxl_area*a-patch_area
                sq_diff_area = diff_area*a*diff_area#np.multiply(diff_area,diff_area)#[x**2 for x in diff_area]
                filtered = sq_diff_area*a*the_filter*mask
                
                value = np.sum(filtered)
#                if value > 0:
                all_neighbor_distances[(ii,jj)] = np.sum(filtered)
                
                all_pixels[(ii,jj)] = i_img[ii + leap + 1,jj + leap + 1]
#                else:
#                    print 'here'
            else:
                return 0
    sorted_dict = sorted(all_neighbor_distances.items(), key=operator.itemgetter(1))
    min_val = sorted_dict[0][1]
    
    if min_val == 0:
        d = dict((k, v) for k, v in all_neighbor_distances.items() if v == 0)
    else:
        threshold = 1.1
        d = dict((k, v) for k, v in all_neighbor_distances.items() if v < threshold*min_val)
    a_key =  random.choice(d.keys())
    #bl = i_img[46:46+ws, 72:72 + ws]
    
#    if x==170 and y==256:
#        im = np.copy(o_img)
#    ## Create figure and axes
#        fig,ax = plt.subplots(1)
#    ## Display the image
#        ax.imshow(im)
#    ## Create a Rectangle patch
#        rect = patches.Rectangle((50,100),40,30,linewidth=1,edgecolor='r',facecolor='none')
#    ## Add the patch to the Axes
#        ax.add_patch(rect)
#        plt.show()
    
   
    #return all_pixels[a_key[1],a_key[0]]
    return all_pixels[a_key]



filename = 'T5.gif'
img=io.imread(filename);
print img.shape

window_size = 11#5
window_size = 19#9
window_size = 23#11

padding = 20

# Get a random window from texture and start the image with it
the_img = start_the_img(img, window_size/2)
the_filter = my_gaussian_filter(window_size)

for ii in range(0+padding,img.shape[0]+padding):
    for jj in range(img.shape[1]+padding,the_img.shape[1]-padding):
        the_img[ii][jj] = find_pixel_value(ii,jj,img, the_img, window_size)
        #print ii, jj
#        
for ii in range(img.shape[0]+padding,200+padding):
    for jj in range(padding,padding+200):
        the_img[ii][jj] = find_pixel_value(ii,jj,img, the_img, window_size)
        

bl = the_img[padding:the_img.shape[0]-padding,padding:the_img.shape[1]-padding]
#io.imshow(bl)
#io.show()
scipy.misc.imsave('T5_11.jpg', bl) 
