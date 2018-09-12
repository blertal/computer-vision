# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 05:44:10 2017

@author: blerta
"""

from skimage import io
import numpy as np
import scipy
import operator
import random
import scipy.misc
from skimage import color


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

def find_pixel_value(x, y, i_img, o_img, ws):
    a = np.ones((ws,ws))
    leap = ws/2
    pxl_area = o_img[x-leap:x+leap+1, y-leap:y+leap+1]
    all_neighbor_distances = {}
    all_pixels = {}
    #print ws
    for ii in range(0,i_img.shape[0]-ws):
        for jj in range(0,i_img.shape[1]-ws):
            patch_area = i_img[ii:ii+ws, jj:jj + ws]
            if (pxl_area.shape[0] == patch_area.shape[0]) and ( pxl_area.shape[1] == patch_area.shape[1]):
                mask = np.copy(pxl_area)
                mask[mask>0]=1
                patch_area = patch_area*mask
                diff_area = pxl_area*a-patch_area
                sq_diff_area = diff_area*a*diff_area#np.multiply(diff_area,diff_area)#[x**2 for x in diff_area]
                filtered = sq_diff_area*a*the_filter*mask
                all_neighbor_distances[(ii,jj)] = np.sum(filtered)
            
                all_pixels[(ii,jj)] = i_img[ii + leap + 1,jj + leap + 1]
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

    return all_pixels[a_key]



window_size = 11#5
window_size = 19#9
window_size = 23#11
padding=20
the_filter = my_gaussian_filter(window_size)


img = color.rgb2gray(io.imread('test_im3.jpg'))
#img1 = np.copy(img[0:149,0:img.shape[1]])
#img2 = np.copy(img[190:img.shape[0],0:img.shape[1]])
img = np.pad(img,pad_width=padding, mode='constant', constant_values=0)
#img_src = np.copy(img[padding:img.shape[0]-padding,padding:img.shape[1]-padding])
#img_src = np.copy(img[370:505,100:390])# Man
img_src = np.copy(img[370:555,100:200])# Man
#img_src = np.copy(img[500:707,640:750])# Floor and Sign

##### Man src
#for ii in range(370,555):
#    for jj in range(100,200):
#        img[ii,jj] = 0

# Make part of image black
## Man
for ii in range(370,505):
    for jj in range(245,281):
        img[ii,jj] = 0
### Sign
#for ii in range(585,686):
#    for jj in range(812,825):
#        img[ii,jj] = 0
#for ii in range(534,585):
#    for jj in range(780,860):
#        img[ii,jj] = 0
## Floor
#for ii in range(458,687):
#    lmt = 1350-int(1.4*ii)
#    for jj in range(lmt -int(1.5*ii-458)+100,lmt):
#        if jj >=0 and jj < img.shape[1]:
#            img[ii,jj] = 0
            
            
            


## Man
for ii in range(370,505):
    for jj in range(245,281):
        img[ii,jj] = find_pixel_value(ii,jj,img_src, img, window_size)
        print ii, jj

### Sign
#for ii in range(585,686):
#    for jj in range(812,825):
#        img[ii,jj] = find_pixel_value(ii,jj,img_src, img, window_size)
#        print ii, jj
#for ii in range(534,585):
#    for jj in range(780,860):
#        img[ii,jj] = find_pixel_value(ii,jj,img_src, img, window_size)
#        print ii, jj

## Floor
#for ii in range(458,687):
#    lmt = 1350-int(1.4*ii)
#    for jj in range(lmt -int(1.5*ii-458)+100,lmt):
#        if jj >=0 and jj < img.shape[1]:
#            img[ii,jj] = find_pixel_value(ii,jj,img_src, img, window_size)
#            print ii, jj
            
bl = img[padding:img.shape[0]-padding,padding:img.shape[1]-padding]
io.imshow(bl)
io.show()

scipy.misc.imsave('3_museum_morning_man_11_fillingaps.jpg', bl)
