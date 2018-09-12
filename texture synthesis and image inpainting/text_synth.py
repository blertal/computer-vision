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


def get_random_window_from_texture(img, ws):
    return (np.random.randint(img.shape[0] - ws),
            np.random.randint(img.shape[1] - ws))
    
def start_the_img(input_img, seed, padding):
    print padding
    my_img = np.ones((200 + 2*padding,200 + 2*padding))*(-1)
    w0 = np.random.randint(img.shape[0] - seed)
    h0 = np.random.randint(img.shape[1] - seed)
    
    # Copy the seed to my img
    init = (my_img.shape[0] - seed)/2
    print 'init',init
    for ii in range(0,seed):
        for jj in range(0,seed):
            print init+ii,init+jj
            my_img[init+ii][init+jj] = input_img[w0+ii][h0+jj]
    return my_img#, input_img[w0:w0+ws,h0:h0+ws]

def find_pixel_value(x, y, i_img, o_img, seed, padding, ws):
    #ws = patch.shape[0]


    #pxl_area  = o_img[ii:ii + ws, jj:jj + ws]
    #pxl_area  = np.ones((ws,ws))*(-1)
    pxl_area = o_img[x-ws/2-1+padding:x-ws/2-1+padding + ws, y-ws/2-1+padding:y-ws/2-1+padding+ws]
    print x-ws/2-1+padding,x-ws/2-1+padding+ws, y-ws/2-1+padding,y-ws/2-1+padding+ws

    diff_area = np.zeros((ws,ws))

    all_neighbor_distances = {}
    all_pixels = {}
    for ii in range(0,ws):
        for jj in range(0,ws):
#            pxl_area  = o_img[x-ws/2-1:x-ws/2-1 , y-ws/2-1:y-ws/2-1 ]
#            patch_area = patch[ii:ii, jj:jj]

            patch_area = i_img[ii:ii+ws, jj:jj + ws]
            
            #print pxl_area.shape
            #print patch_area.shape
            
            # Find SSD
            for ll in range(0,pxl_area.shape[0]):
                for mm in range(0,pxl_area.shape[1]):
                    if pxl_area[ll,mm] <> -1:
                        
                        a_diff = pxl_area[ll,mm] - patch_area[ll,mm]
                        diff_area[ll,mm] = math.pow(a_diff, 2)
                        #dist = dist + math.pow(a_diff, 2) * ndi.gaussian_filter(face, sigma=5)
            
            all_neighbor_distances[(ii,jj)] = np.sum(ndi.gaussian_filter(diff_area, sigma=ws/2))
            all_pixels[(ii,jj)] = patch_area[ws/2,ws/2]
    sorted_dict = sorted(all_neighbor_distances.items(), key=operator.itemgetter(1))
    min_val = sorted_dict[0][1]
    threshold = 1.1
    d = dict((k, v) for k, v in all_neighbor_distances.items() if v < threshold*min_val)
    while len(d.keys())==0:
        threshold = threshold + 0.1
        d = dict((k, v) for k, v in all_neighbor_distances.items() if v < 1.1*min_val)
    #print 'keys',len(d.keys()), x, y
#    if len(d.keys())==0:
#        print 'letssee', len(all_neighbor_distances.keys())
#        io.imshow(the_img)
#        plt.title('Binary Image')
#        io.show()
#        return all_pixels[sorted_dict[0][1][0],sorted_dict[0][1][1]]
#    else:
    a_key =  random.choice(d.keys())
    return all_pixels[a_key[0],a_key[1]]

filename = 'T1.gif'
img=io.imread(filename);
print img.shape

window_size = 11
padding = window_size/2
seed = 3

# Get a random window from texture
#(w,h) = get_random_window_from_texture(img, window_size)

#the_img, the_patch = start_the_img(img,11)
the_img = start_the_img(img,seed, padding)


#find_pixel_value(100,106,the_img, the_patch)

for ii in range(106,110):#105:194
    depth = 99+padding - (ii-99) #+ padding
    print depth
#    depth = (the_img.shape[0] + 2*padding - seed)/2 - 2
#    print 'depth',depth,depth+1,199-depth
    for jj in range(depth+padding,199-depth+padding):
        #print 9000
        the_img[ii][jj] = find_pixel_value(ii,jj,img, the_img, seed, padding, window_size)
        the_img[jj][ii] = find_pixel_value(jj,ii,img, the_img, seed, padding, window_size)
        
#for ii in reversed(range(4,100)):#107-109
for ii in reversed(range(90,103)):
    depth = ii
    for jj in range(depth,199-depth+padding+seed+2):
        the_img[ii][jj] = find_pixel_value(ii,jj,img, the_img, seed, padding, window_size)
        the_img[jj][ii] = find_pixel_value(jj,ii,img, the_img, seed, padding, window_size)
    

#the_img[ii][jj] = find_pixel_value(69,139,img, the_img, seed, padding, window_size)
#        
        
        

io.imshow(the_img)
plt.title('Binary Image')
io.show()
  
