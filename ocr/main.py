import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
#from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io#, exposure
import matplotlib.pyplot as plt
#from matplotlib.patches import Rectangle
#import pickle

from train import train
import os


all_features = []
all_labels   = []
all_regions  = []
all_files    = {}
all_imgs = {}
all_imgs_features = {}
img_counter = 0
feat_counter = 0
for file in os.listdir(os.getcwd()):
    if file.endswith('.bmp') and len(file) == 5 and file[0].isalpha() and file[0].islower():
      print file
      letter_features, letter_regions, img_binary = train(file,False)
      letter = file[0]
      all_files[img_counter] = letter
      #all_labels.append(file[0]*len(letter_features))
      #letter_features.append(counter)
      all_labels = all_labels + [letter for ii in range(0,len(letter_features))]
      #all_features.append(letter_features)
      all_features = all_features + letter_features
      all_regions  = all_regions + letter_regions
      all_imgs[img_counter] = img_binary
      
      # 
      feat_numbers = []
      for feat in letter_features:
          feat_numbers.append(feat_counter)
          feat_counter = feat_counter + 1
      all_imgs_features[img_counter] = feat_numbers
      
      img_counter = img_counter + 1


standardized_all_features = []
B=np.asmatrix(all_features)

# Finding the means
mean_0 = np.mean(B[:,0])
mean_1 = np.mean(B[:,1])
mean_2 = np.mean(B[:,2])
mean_3 = np.mean(B[:,3])
mean_4 = np.mean(B[:,4])
mean_5 = np.mean(B[:,5])
mean_6 = np.mean(B[:,6])

# Finding the standard deviations
std_0 = np.std(B[:,0])
std_1 = np.std(B[:,1])
std_2 = np.std(B[:,2])
std_3 = np.std(B[:,3])
std_4 = np.std(B[:,4])
std_5 = np.std(B[:,5])
std_6 = np.std(B[:,6])

for ii in range(0,len(all_features)):
    B[ii,0] = (B[ii,0] - mean_0)/std_0
    B[ii,1] = (B[ii,1] - mean_1)/std_1
    B[ii,2] = (B[ii,2] - mean_2)/std_2
    B[ii,3] = (B[ii,3] - mean_3)/std_3
    B[ii,4] = (B[ii,4] - mean_4)/std_4
    B[ii,5] = (B[ii,5] - mean_5)/std_5
    B[ii,6] = (B[ii,6] - mean_6)/std_6
    
# Normalized features
standardized_all_features = B.tolist()


# RECOGNITION ON TRAINING DATA
D=cdist(standardized_all_features, standardized_all_features)
#io.imshow(D)
#plt.title('Distance Matrix')
#io.show()

D_index = np.argsort(D, axis=1)

#find number of matches
matches = 0
result_labels = []
for ii in range(0,len(all_features)):
    result_labels = result_labels + [all_labels[D_index[ii,1]]]
#    if all_labels[ii] == all_labels[D_index[ii,1]]:
#        matches = matches + 1

#print matches/len(all_features)


# CONFUSION MATRIX
#results = np.asmatrix(D_index)
#results_list = results[:,1].tolist()
#confM = confusion_matrix(all_labels, result_labels)#which takes as input, the correct classes Ytrue (as a vector) and the output classes Ypred (as a vector)
#io.imshow(confM)
#plt.title('Confusion Matrix')
#io.show()



# Create the image

total_counter   = 0
sum_correct_counter = 0
#for img_no in all_imgs.iterkeys():
    correct_counter = 0
    img_no = 15
    #print img_no
    img_binary = all_imgs[img_no]
    io.imshow(img_binary)
    ax = plt.gca()
    match_regions = [all_regions[feat] for feat in all_imgs_features[img_no] ]
    # Labels
    match_lbls = [result_labels[feat] for feat in all_imgs_features[img_no]  ]
    for ii in range(0,len(match_regions)):
        props = match_regions[ii]
        minr, minc, maxr, maxc = props.bbox
        lbl = match_lbls[ii]
        total_counter = total_counter + 1
        if lbl == (all_files[img_no]):
            correct_counter = correct_counter + 1
            ax.add_patch(Rectangle((minc,  minr),  maxc - minc,  maxr - minr, 
                     fill=False, edgecolor='green', linewidth=1))
        else:
            ax.add_patch(Rectangle((minc,  minr),  maxc - minc,  maxr - minr, 
                     fill=False, edgecolor='red', linewidth=1))
            
    #match_lbls = result_labels[all_imgs_features[img_no]]
    plt.title('Recognition results for letter ' + all_files[img_no] + ' - ' + str(correct_counter*100/80) + '%')
    plt.savefig('result_'+all_files[img_no]+'.png')
    #io.show()
    #print correct_counter,'-', correct_counter*100/80,'%'
    #print correct_counter
    sum_correct_counter = sum_correct_counter + correct_counter
#print sum_correct_counter,'-', sum_correct_counter*100/(16*80),'%'
#print sum_correct_counter, total_counter