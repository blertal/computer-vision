# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 04:13:31 2017

@author: blerta
"""

from skimage import io
import numpy as np
import scipy
import math
import operator
from sets import Set
from skimage import color
import scipy.misc

def my_gaussian_filter(sz):
    fltr = np.ones((sz,sz))
#    fltr = np.zeros((sz,sz))
#    delta = sz/2
#    sigma=0.2
#    for ii in range(0,sz):
#        for jj in range(0,sz):
#            x = ii-delta
#            y = jj-delta
#            fltr[ii][jj] = (1/(2*3.14*sigma))*np.exp((-x*x-y*y)/2*sigma*sigma)
    return fltr

def get_random_window_from_texture(img, ws):
    return (np.random.randint(img.shape[0] - ws),
            np.random.randint(img.shape[1] - ws))

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
                
#                if value > 0:
                all_neighbor_distances[(ii,jj)] = np.sum(filtered)
                
                all_pixels[(ii,jj)] = i_img[ii,jj]
#                else:
#                    print 'here'
            else:
                return 0
#    sorted_dict = sorted(all_neighbor_distances.items(), key=operator.itemgetter(1))
#    min_val = sorted_dict[0][1]
    
#    #if min_val == 0:
#    d = dict((k, v) for k, v in all_neighbor_distances.items() if v == min_val)
##    else:
##        
##        # Adjusting the threshold
##        threshold = 1.1
##        #T5
##        
##        #T2
##        d = dict((k, v) for k, v in all_neighbor_distances.items() if v < threshold*min_val)
###    while len(d.keys())==0:
###        threshold = threshold + 0.1
###        d = dict((k, v) for k, v in all_neighbor_distances.items() if v < threshold*min_val)
##    #threshold = 1.1
#    a_key =  random.choice(d.keys())
    
    sorted_dict = sorted(all_neighbor_distances.items(), key=operator.itemgetter(1))
    a_key = sorted_dict[1][0]
    
#    sortedD = OrderedDict(sorted(all_neighbor_distances.items(), key=itemgetter(1)))
#    a_key = sortedD.keys()[0]
    
    
    
    
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
    #return all_pixels[a_key]
    #return np.copy(i_img[a_key[0]-leap:a_key[0]+leap+1, a_key[1]-leap:a_key[1]+leap+1])
    return np.copy(i_img[a_key[0]:a_key[0]+ws, a_key[1]:a_key[1] + ws]), a_key[0] + leap+1, a_key[1] + leap+1

def init_all_region(img_shape):
    rgn = Set()
    for ii in range(0,img_shape[0]):
        for jj in range(0,img_shape[1]):
            rgn.add((ii,jj))
    return rgn

def get_patch(p, ws):
    patch = Set()
    for ii in range(p[0]-ws/2,p[0]+ws/2+1):
        for jj in range(p[1]-ws/2,p[1]+ws/2+1):
            patch.add((ii,jj))
    return patch
    
def get_confidence(patch, src_rgn, confidence):
    intersect = patch.intersection(src_rgn)
    the_sum_conf = 0
    for pxl in intersect:
        the_sum_conf = the_sum_conf + confidence[pxl]

    return 1.0*the_sum_conf/len(patch)

def get_data_term(p_k, d_patch, img, o_rgn):
    alpha = 255
    gradient = np.gradient(img)
    gr_x = gradient[0][p_k[0],p_k[1]]
    gr_y = gradient[0][p_k[0],p_k[1]]
    sum_x = 0
    sum_y = 0
    for item in o_rgn:
        sum_x = sum_x + item[0]
        sum_y = sum_y + item[1]
        
    mean_x = sum_x/len(o_rgn)
    mean_y = sum_y/len(o_rgn)
    
    n_p_x = mean_x - gr_x
    n_p_y = mean_y - gr_y
    
    data_term = (gr_x*n_p_x + gr_y*n_p_y)/(alpha*math.sqrt(n_p_x*n_p_x + n_p_y*n_p_y))
    return data_term
    #np.gradient(np.array([[3,4,5], [7,8,9]], dtype=np.float))
    
def make_parts_of_image_black(img):
    omg_rgn = Set()
    conf = {}
    print img.shape[0]
    for ii in range(0,img.shape[0]):
        for jj in range(0,img.shape[1]):
            conf[ii,jj] = 1

#    # Test
#    for ii in range(370,380):
#        for jj in range(245,255):
#            img[ii,jj] = 0
#            omg_rgn.add((ii,jj))
#            conf[ii,jj] = 0
#    for ii in range(370,505):
#        for jj in range(245,281):
#            img[ii,jj] = 0
#            omg_rgn.add((ii,jj))
#            conf[ii,jj] = 0
    
##    # Man
#    for ii in range(370,505):
#        for jj in range(245,281):
#            img[ii,jj] = 0
#            omg_rgn.add((ii,jj))
#            conf[ii,jj] = 0
            
##     Sign
#    for ii in range(585,687):
#        for jj in range(812,825):
#            img[ii,jj] = 0
#            omg_rgn.add((ii,jj))
#            conf[ii,jj] = 0
#    for ii in range(534,585):
#        for jj in range(780,860):
#            img[ii,jj] = 0
#            omg_rgn.add((ii,jj))
#            conf[ii,jj] = 0
    
    # Floor
    for ii in range(458,687):
    #for ii in range(650,687):
        lmt = 1350-int(1.4*ii)
        for jj in range(lmt -int(1.5*ii-458)+100,lmt):
            if jj >=0 and jj < img.shape[1]:
                img[ii,jj] = 0
                omg_rgn.add((ii,jj))
                conf[ii,jj] = 0

    for ii in range(686,687):
        lmt = 1350-int(1.4*ii)
        for jj in range(lmt -int(1.5*ii-458)+100,lmt):
            if jj >=20 and jj < img.shape[1]:
                img[ii,jj] = 0.1
                omg_rgn.discard((ii,jj))
                conf[ii,jj] = 1
    for ii in range(651,687):
        for jj in range(20,21):
            #if jj >=20 and jj < img.shape[1]:
                img[ii,jj] = 0.1
                omg_rgn.discard((ii,jj))
                conf[ii,jj] = 1
    
    return img, omg_rgn, conf

def get_neighborhood(pxl):
    neighb = Set()
    for ii in [pxl[0]-1,pxl[0],pxl[0]+1]:
        for jj in [pxl[1]-1,pxl[1],pxl[1]+1]:
            neighb.add((ii,jj))
    neighb.discard(pxl)
    return neighb    
    
def find_fill_front(o_region, src_rgn):
    fill_front = Set()
    for pxl in o_region:
#        if pxl[0]==504 and pxl[1] == 280:
#            print 'here'
        if len(src_rgn.intersection(get_neighborhood(pxl))) > 0:
            fill_front.add(pxl)
    return fill_front
    
def contrast_rgn(img, rgn):
    cp_img = np.copy(img)
    for pxl in rgn:
        cp_img[pxl[0],pxl[1]] = 240
    return cp_img
    
padding=20
img = color.rgb2gray(io.imread('test_im3.jpg'))
img = np.pad(img,pad_width=padding, mode='constant', constant_values=0)
#print img.shape
#io.imshow(img)
#io.show()

window_size = 23
the_filter = my_gaussian_filter(window_size)
P   = {}
C = {}
D = {}

img, omega_region, C = make_parts_of_image_black(img)
#img_src = np.copy(img[padding:img.shape[0]-padding,padding:img.shape[1]-padding])
#img_src = np.copy(img[370:505,100:390])# Man
#img_src = np.copy(img[370:530,80:200])# Man
img_src = np.copy(img[440:687,620:760])# Floor and Sign
#io.imshow(img_src)
#io.show()


##define region to be removed
all_region    = init_all_region(img.shape)
source_region = all_region.difference(omega_region)

d_trgt_rgn = find_fill_front(omega_region, source_region)
#img = contrast_rgn(img, d_trgt_rgn)
io.imshow(img)
io.show()



while len(d_trgt_rgn) > 0:
    P={}

    for p in d_trgt_rgn:
    #    p_k = p[0]
    #    p_v = p[1]
        #print 1
        p_patch = get_patch(p, window_size)
        #print 2
        # confidence
        C[p] = get_confidence(p_patch, source_region, C)
        #print 'confidence', C[p]
        # data
        D[p] = get_data_term(p, p_patch, img, omega_region)#(p_k, d_patch, source_region, C)
        #print 4
    
        P[p] = C[p]*D[p]
        #print 'priority ',P[p], C[p],D[p]
        
    #sorted_P = OrderedDict(P.items(), key=itemgetter(1))
    #a_key = sorted_P.keys()[1]
        
    sorted_P = sorted(P.items(), key=operator.itemgetter(1), reverse=True)
    #max_val = sorted_P[0][1]
    #d = dict((k, v) for k, v in P.items() if v == max_val)
    #a_key =  random.choice(d.keys())
    a_key = sorted_P[0][0]

    new_patch, a, b = find_pixel_value(a_key[0],a_key[1],img_src, img, window_size)
    print 'HERE', a_key[0], a_key[1], a, b
    leap = window_size/2
    #print 'BEFORE' , len(omega_region)
    for ii in range(0,window_size):
        for jj in range(0,window_size):
            coord_x = a_key[0]-leap+ii
            coord_y = a_key[1]-leap+jj
            img[coord_x,coord_y] = new_patch[ii,jj]
            #print coord_x,coord_y,new_patch[ii,jj]
            C[coord_x,coord_y] = 1
            #print 'DISCARD',coord_x,coord_y
            omega_region.discard((coord_x,coord_y))
            source_region.add((coord_x,coord_y))
    print 'AFTER' , len(omega_region)
#    io.imshow(img)
#    io.show()

    d_trgt_rgn = find_fill_front(omega_region, source_region)
    #bl = contrast_rgn(img, d_trgt_rgn)
#    io.imshow(bl)
#    io.show()

## Get a random window from texture and start the image with it
#the_img = start_the_img(img, window_size/2)
#
#the_filter = my_gaussian_filter(window_size)
#
##print 'sdfxdf'
##the_img[20][20+window_size] = find_pixel_value(25,31,img, the_img, window_size)
##print the_img[25][31]
#
#
#for ii in range(20,window_size/2 + 20):
#    for jj in range(20+window_size/2,140):
#        the_img[ii][jj] = find_pixel_value(ii,jj,img, the_img, window_size)
#        #print ii, jj
#        
#for ii in range(20+window_size/2,100+20):
#    for jj in range(20,20+100):
#        the_img[ii][jj] = find_pixel_value(ii,jj,img, the_img, window_size)
#        #print the_img[ii][jj]
#        

#
#        
#a_img = the_img[padding:210-padding,padding:210-padding]
##plt.title('T3 image, window size is '+ str(window_size))
#plt.title('T5 image, window size is 5')
###plt.imshow(a_img)
##plt.savefig('T4_5.jpg')

bl = img[padding:img.shape[0]-padding,padding:img.shape[1]-padding]
io.imshow(bl)
io.show()
scipy.misc.imsave('4_museum_morning_filling_floor_11_2.jpg', bl)



cpImg = np.copy(img)
#cpImg[500][500] = 1

for ii in range(450,470):
    for jj in range(500,520):
        #if cpImg[ii][jj] == 0:
            cpImg[ii][jj] = 1#cpImg[ii][jj-50]

io.imshow(cpImg)
io.show()

