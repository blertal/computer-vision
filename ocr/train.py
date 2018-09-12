import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure, feature
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import scipy.signal as sgnl
from scipy import ndimage
from skimage.transform import resize


def train( filename,charts=False ):    
  #filename='a.bmp'
  #charts = True
  
  valleyThresholdMethod = True
  hysteresisThresholdMethod = False
  hysteresisThresholdMethodSimple = False
  
  hogFeatures = True
  openingAndClosing = False
  
  
  img=io.imread(filename);#'a.bmp'
  print img.shape

  if charts:
    io.imshow(img)
    plt.title('Original Image')
    io.show()
  

  hist = exposure.histogram(img)
  #We can visualize the histogram as following:
  if charts:
    plt.bar(hist[1], hist[0])
    plt.title('Histogram')
    plt.show()


  th = 200
  img_binary = (img < th).astype(np.double)

  ###############################################
  # Histogram valley threshold method
  # Define threshold######################
  if valleyThresholdMethod:
      #th=250#200
      #hist[1] = hist[1].append(hist[len(hist[1]-1) + 1])
      #hist[1] = np.append(hist[1],[hist[1][len(hist[1])-1] + 1])
      tuple_0 = np.append(hist[0],[0])
      tuple_1 = np.append(hist[1],[hist[1][len(hist[1])-1] + 1])
      hist = (tuple_0,tuple_1)
      
      # indices
      maxs = sgnl.argrelmax(hist[0])
      mins = sgnl.argrelmin(hist[0])
      
      # values
      max_vals = hist[0][maxs]
      min_vals = hist[0][mins]
      
      # indices of 2 largest peaks
      idx = (-max_vals).argsort()[:3]
      #idx[0]
      #idx[1]
      # indices of smallest valley between the peaks
  
      limit_low  = min(maxs[0][idx[0]], maxs[0][idx[1]])#min(max_vals[idx[0]], max_vals[idx[1]])
      limit_high = max(maxs[0][idx[0]], maxs[0][idx[1]])#max(max_vals[idx[0]], max_vals[idx[1]])
      #print 'THRESHOLD1 '+ str(limit_low)
      #print 'THRESHOLD2 '+ str(limit_high)
    
      min_list = []
      for x in mins[0]:
          if (x < limit_high and x > limit_low):
              min_list.append(x)
              #print x
      print 'THRESHOLD3 '+ str(min_list)
    
      valley_vals = hist[0][min_list]
      valley_indices = hist[1][min_list]
      # indices of 2 largest peaks
      valley_idx = (valley_vals).argsort()[:1]
      threshold_valley = valley_indices[valley_idx[0]]#hist[0][valley_idx[0][0]]
      print 'THRESHOLD4 '+ str(threshold_valley)
      
      count=0
      img_copy = img_binary
      while np.sum(img_copy) != 0:
          img_copy = ndimage.binary_erosion(img_copy).astype(img_copy.dtype)
          count = count + 1
      print 'COUNT ' + str(count)
      l = len(hist[0])
      
      # Real threshold vaule used
      th = (l - count)*0.9
      #th = (threshold_valley + maxs[0][idx[0]])*0.5
      img_binary = (img < th).astype(np.double)
  
  
  ####################################
  # BEGIN HYSTERESIS
  ####################################
  if hysteresisThresholdMethodSimple:
      count=0
      img_copy = img_binary
      while np.sum(img_copy) != 0:
          img_copy = ndimage.binary_erosion(img_copy).astype(img_copy.dtype)
          count = count + 1
      print 'COUNT ' + str(count)
      thresh_percentage = (76 + max(3*count,12))*0.01
    
      # hysteresis p-tile threshold
      #if hysteresisThresholdMethod:
      
      hist_cumsum = np.cumsum(hist,axis=1)
      thresh_high = thresh_percentage  * max(hist_cumsum[1])
      thresh_high = 0.9  * max(hist_cumsum[1])
      print 'thresh_high ' + str(thresh_high)
      thresh_low  = 0.8 * max(hist_cumsum[1])
      if charts:
        plt.bar(hist_cumsum[1], hist_cumsum[0])
        plt.title('Cumsum Histogram')
        plt.show()
    
      th1 = min(np.where( hist_cumsum > thresh_high )[1])
      th2 = min(np.where( hist_cumsum > thresh_low )[1])
      
      
      #maxs = sgnl.argrelmax(hist[0])
      #max_vals = hist[0][maxs]
      #idx = (-max_vals).argsort()[:1]
      #print 'IDX ' + str(maxs[0][idx[0]])
      th = th1#maxs[0][idx[0]]-count
      #th = 250
      print 'THRESHOLD ' + str(th)
      img_binary = (img < th).astype(np.double)
      
  if hysteresisThresholdMethod:
  
      #Pixels above the high threshold are classified as object and 
      #belowthe lowthresholdas background.
      #Pixels between the lowand high thresholds are classified as object only if theyareadjacent to other object pixels.
      strong_pxl_img_binary = (img < th2).astype(np.double)
      weak_pxl_img_binary_1 = (img < th1).astype(np.double)
      weak_pxl_img_binary_2 = (img > th2).astype(np.double)
      weak_pxl_img_binary   = np.logical_and(weak_pxl_img_binary_1, weak_pxl_img_binary_2)
      counter = 1
      while counter > 0:
        counter = 0
        for ii in range(0,img.shape[0]):
          for jj in range(0,img.shape[1]):
              if strong_pxl_img_binary[ii][jj] == 1 or weak_pxl_img_binary[ii][jj] == 0:
                  continue
              posx_1 = ii-1
              posx_2 = ii
              posx_3 = ii+1
              posy_1 = jj-1
              posy_2 = jj
              posy_3 = jj+1
              if posx_1 > 0 and posx_1 < img.shape[0] and posy_1 > 0 and posy_1 < img.shape[1] and strong_pxl_img_binary[posx_1][posy_1] == 1:
                     strong_pxl_img_binary[ii][jj] = 1
                     counter = counter + 1
                     continue
              if posx_1 > 0 and posx_1 < img.shape[0] and posy_2 > 0 and posy_2 < img.shape[1] and strong_pxl_img_binary[posx_1][posy_2] == 1:
                     strong_pxl_img_binary[ii][jj] = 1
                     counter = counter + 1
                     continue
              if posx_1 > 0 and posx_1 < img.shape[0] and posy_3 > 0 and posy_3 < img.shape[1] and strong_pxl_img_binary[posx_1][posy_3] == 1:
                     strong_pxl_img_binary[ii][jj] = 1
                     counter = counter + 1
                     continue
    
              if posx_2 > 0 and posx_2 < img.shape[0] and posy_1 > 0 and posy_1 < img.shape[1] and strong_pxl_img_binary[posx_2][posy_1] == 1:
                     strong_pxl_img_binary[ii][jj] = 1
                     counter = counter + 1
                     continue
              if posx_2 > 0 and posx_2 < img.shape[0] and posy_3 > 0 and posy_3 < img.shape[1] and strong_pxl_img_binary[posx_2][posy_3] == 1:
                     strong_pxl_img_binary[ii][jj] = 1
                     counter = counter + 1
                     continue
                 
              if posx_3 > 0 and posx_3 < img.shape[0] and posy_1 > 0 and posy_1 < img.shape[1] and strong_pxl_img_binary[posx_3][posy_1] == 1:
                     strong_pxl_img_binary[ii][jj] = 1
                     counter = counter + 1
                     continue
              if posx_3 > 0 and posx_3 < img.shape[0] and posy_2 > 0 and posy_2 < img.shape[1] and strong_pxl_img_binary[posx_3][posy_2] == 1:
                     strong_pxl_img_binary[ii][jj] = 1
                     counter = counter + 1
                     continue
              if posx_3 > 0 and posx_3 < img.shape[0] and posy_3 > 0 and posy_3 < img.shape[1] and strong_pxl_img_binary[posx_3][posy_3] == 1:
                     strong_pxl_img_binary[ii][jj] = 1
                     counter = counter + 1
                     continue
                 
      print str(counter)
    
      if charts:
        io.imshow(strong_pxl_img_binary)
        plt.title('Binary Image')
        io.show()
      if charts:
        io.imshow(weak_pxl_img_binary)
        plt.title('Binary Image')
        io.show()
        
      img_binary = strong_pxl_img_binary
  ####################################
  # END HYSTERESIS
  ####################################
  


  if openingAndClosing:
      img_binary = ndimage.binary_dilation(img_binary).astype(img_binary.dtype)
      img_binary = ndimage.binary_erosion(img_binary).astype(img_binary.dtype)
      img_binary = ndimage.binary_dilation(img_binary).astype(img_binary.dtype)
      #img_binary = ndimage.binary_dilation(img_binary).astype(img_binary.dtype)
  
  # Improvement 1 - find threshold
  # Standardize size of regions
  # Thickness of the lines, detect with erosion, and standardize
  
  

  
  
  
  
  
  
  # Visualize the binary image
  if charts:
    io.imshow(img_binary)
    plt.title('Binary Image')
    io.show()


  # Connected component analysis
  img_label = label(img_binary, background=0)
  # You can visualize the resulting component image:
  if charts:
    io.imshow(img_label)
    plt.title('Labeled Image')
    io.show()

  # Find number of connected components
  print np.amax(img_label)


  # Component bounding boxes
  regions = regionprops(img_label)
  #http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
  if charts:
    io.imshow(img_binary)
    ax = plt.gca()
    #ax.title('Bounding Boxes')

  Features = []
  Regions  = []
  for props in regions:
    minr, minc, maxr, maxc = props.bbox
    ################################################
    # Add threshold for height, width
    if minr + 13 <= maxr and minc + 13 <= maxc:
      if charts:
        ax.add_patch(Rectangle((minc,  minr),  maxc - minc,  maxr - minr, 
                     fill=False, edgecolor='red', linewidth=1))

      # Computing Hu Moments and Removing Small Components
      roi = img_binary[minr:maxr, minc:maxc]
      m = moments(roi)
      cr = m[0, 1] / m[0, 0]
      cc = m[1, 0] / m[0, 0]
      mu = moments_central(roi, cr, cc)
      nu = moments_normalized(mu)
      hu = moments_hu(nu)

      Regions.append(props)
      
      #hog features
      if hogFeatures:
          hogs = feature.hog(resize(roi,(25,25)), orientations=9, pixels_per_cell=(5, 5),
                             cells_per_block=(2, 2), normalise=False)
          Features.append(np.concatenate((hu,hogs),axis=1))
      else:
          Features.append(hu)
          #Features.append(np.concatenate((hu,[(maxc - minc),(maxr - minr)]),axis=1))
          
  if charts:
    io.show()


  return Features, Regions, img_binary

