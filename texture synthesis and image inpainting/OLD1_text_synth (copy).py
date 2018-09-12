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
    sigma=1
    for ii in range(0,sz):
        for jj in range(0,sz):
            x = ii-delta
            y = jj-delta
            fltr[ii][jj] = (1/(2*3.14*sigma))*np.exp((-x*x-y*y)/2*sigma*sigma)
            #fltr[ii][jj] = float(1)*np.exp((-x*x-y*y)/2*sigma*sigma)

    return fltr

def get_random_window_from_texture(img, ws):
    return (np.random.randint(img.shape[0] - ws),
            np.random.randint(img.shape[1] - ws))
    
def start_the_img(input_img, seed, padding):

    pad0 = (200-seed)/2 + padding
    pad1 = (200-seed)/2 + padding
    npad = ((pad0,pad0), (pad1,pad1))
    return np.pad(input_img,pad_width=npad, mode='constant', constant_values=0)

def find_pixel_value(x, y, i_img, o_img, padding, ws):
    #print x, y
#    print x-ws/2-1+padding,x-ws/2-1+padding + ws
#    print y-ws/2-1+padding,y-ws/2-1+padding+ws
    #pxl_area = o_img[x-ws/2-1+padding:x-ws/2-1+padding + ws, y-ws/2-1+padding:y-ws/2-1+padding+ws]
    leap = ws/2
    #print '111',x-leap, x+leap
    #print '222',y-leap,y+leap
    pxl_area = o_img[x-leap:x+leap+1, y-leap:y+leap+1]
    
    #print x-ws/2-1+padding,x-ws/2-1+padding+ws, y-ws/2-1+padding,y-ws/2-1+padding+ws

    #diff_area = np.zeros((ws,ws))

    all_neighbor_distances = {}
    all_pixels = {}
    for ii in range(0,ws):
        for jj in range(0,ws):
            patch_area = i_img[ii:ii+ws, jj:jj + ws]
            
            # Find SSD
#            for ll in range(0,pxl_area.shape[0]):
#                for mm in range(0,pxl_area.shape[1]):
#                    if pxl_area[ll,mm] <> -1:
#                        
#                        a_diff = pxl_area[ll,mm] - patch_area[ll,mm]
#                        diff_area[ll,mm] = math.pow(a_diff, 2)
            #diff_area = scipy.stats.stats.ss(np.subtract(pxl_area,patch_area))
            #print 'shape1', pxl_area.shape
            #print 'shape2', patch_area.shape
            if (pxl_area.shape[0] == patch_area.shape[0]) and ( pxl_area.shape[1] == patch_area.shape[1]):
                diff_area = np.square(np.subtract(pxl_area,patch_area))
    #            all_neighbor_distances[(ii,jj)] = np.sum(ndi.gaussian_filter(diff_area, sigma=ws/2))
    #            mean = (0,0)
    #            cov = [[1, 0], [0, 1]]
    #            gaussian_krnl = np.random.multivariate_normal(mean, cov, ws)
                #filtered = gaussian_filter(diff_area,0.5)
                filtered = np.multiply(diff_area,the_filter)
                all_neighbor_distances[(ii,jj)] = np.sum(filtered)
                #np.random.multivariate_normal((00), cov, 10000)
                
                all_pixels[(ii,jj)] = patch_area[ws/2,ws/2]
            else:
                return 0
    sorted_dict = sorted(all_neighbor_distances.items(), key=operator.itemgetter(1))
    min_val = sorted_dict[0][1]
    # Adjusting the threshold
    threshold = 1.1
    d = dict((k, v) for k, v in all_neighbor_distances.items() if v < threshold*min_val)
    while len(d.keys())==0:
        threshold = threshold + 0.1
        d = dict((k, v) for k, v in all_neighbor_distances.items() if v < threshold*min_val)
    
    a_key =  random.choice(d.keys())
    return all_pixels[a_key[0],a_key[1]]

filename = 'T4.gif'
img=io.imread(filename);
print img.shape

window_size = 9
padding = window_size/2
seed = np.min(img.shape)
#seed0 = img.shape[0]
#seed1 = img.shape[1]
#print seed

# Get a random window from texture
the_img = start_the_img(img, seed, padding)

the_filter = my_gaussian_filter(window_size)

#io.imshow(the_img)
#plt.title('Binary Image')
#io.show()









val = 72
#103-105
#102-106


## 33333333
#limit = 107 + seed/2#11
#limit = 106 + seed/2#9
#for ii in range(limit,limit+val+1):
#    depth = 99+padding - (ii-99)
#    for jj in range(depth+padding+2,199-depth+padding-2):
#        the_img[ii][jj] = find_pixel_value(ii,jj,img, the_img, padding, window_size)
#
#
#
## 22222222      
limit = 103 + seed/2
limit = 104 + seed/2
for ii in range(limit,limit+val+6):
    depth = 99+padding - (ii-99)
    #print ii
    for jj in range(depth+padding+1,199-depth+padding+1):
        the_img[jj][ii] = find_pixel_value(jj,ii,img, the_img, padding, window_size)
        the_img[ii][jj] = find_pixel_value(ii,jj,img, the_img, padding, window_size)

#
#
##### 1111111 and 4444444
limit = 102 - seed/2
limit = 100 - seed/2
for ii in reversed(range(limit -val,limit+3)):#padding,103
    depth = ii
    for jj in range(depth,208-depth-1):
        the_img[ii][jj] = find_pixel_value(ii,jj+1,img, the_img, padding, window_size)
        the_img[jj][ii] = find_pixel_value(jj,ii,img, the_img, padding, window_size)

#
### 4444444
#limit = 102 - seed/2
#limit = 101 - seed/2
##for ii in reversed(range(limit -val+2,limit+4)):#padding,103
#for ii in reversed(range(limit -val+2,limit+4)):
#    depth = ii
#    #for jj in range(depth-1,210-depth):
#    print depth-1,210-depth, limit -val+3,limit+4
#    for jj in range(depth-1,210-depth):
#        #print jj
#    #for jj in range(50,51):
#        the_img[jj][ii] = find_pixel_value(jj,ii,img, the_img, padding, window_size)

        
a_img = the_img[padding:210-padding,padding:210-padding]
plt.title('T4 image, window size is '+ str(window_size))
##plt.imshow(a_img)
#plt.savefig('T4_5.jpg')
#
io.imshow(a_img)
io.show()
#  
