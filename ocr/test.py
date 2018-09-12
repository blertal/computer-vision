import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
import pickle
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from train import train
import os

def test( testFilename, testGTFilename, charts=False ): 
#testFilename   = 'test2.bmp'
#testGTFilename = 'test2_gt.pkl'

    knnClassifier = False
    otherClassifier = False
    	
  

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
          letter_features, letter_regions, img_binary = train(file,charts)
          letter = file[0]
          all_files[img_counter] = letter
          all_labels = all_labels + [letter for ii in range(0,len(letter_features))]
          all_features = all_features + letter_features
          all_regions  = all_regions + letter_regions
          all_imgs[img_counter] = img_binary
          
          feat_numbers = []
          for feat in letter_features:
              feat_numbers.append(feat_counter)
              feat_counter = feat_counter + 1
          all_imgs_features[img_counter] = feat_numbers
          
          img_counter = img_counter + 1


    # Get the test file features
    test_features, test_regions, test_img_binary = train(testFilename,False)



    stand_train_feats = []
    stand_test_feats  = []
    B = np.asmatrix(all_features)
    B_test = np.asmatrix(test_features)

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
        
        
    for ii in range(0,len(test_features)):
        B_test[ii,0] = (B_test[ii,0] - mean_0)/std_0
        B_test[ii,1] = (B_test[ii,1] - mean_1)/std_1
        B_test[ii,2] = (B_test[ii,2] - mean_2)/std_2
        B_test[ii,3] = (B_test[ii,3] - mean_3)/std_3
        B_test[ii,4] = (B_test[ii,4] - mean_4)/std_4
        B_test[ii,5] = (B_test[ii,5] - mean_5)/std_5
        B_test[ii,6] = (B_test[ii,6] - mean_6)/std_6
    
    # Normalized features
    stand_train_feats = B.tolist()
    stand_test_feats  = B_test.tolist()
    stand_all_feats   = stand_train_feats + stand_test_feats



    # RECOGNITION ON TRAINING DATA
    D=cdist(stand_test_feats, stand_train_feats)
    if charts:
        io.imshow(D)
        plt.title('Distance Matrix')
        plt.savefig('distmatrix_'+testFilename[0:len(testFilename)-4]+'.png')
        io.show()
    
    D_index = np.argsort(D, axis=1)
    
    # Find matches
    result_labels = []
    for ii in range(0,len(test_features)):
        result_labels = result_labels + [all_labels[D_index[ii,0]]]




    
    
    if knnClassifier:
        neigh = KNeighborsClassifier(n_neighbors=4)#, metric='euclidean')#euclidean
        neigh.fit(stand_train_feats, all_labels[0:len(stand_train_feats)])
        result_labels = []
        for ii in range(0,len(test_features)): #was test_features
            result_labels = result_labels + [neigh.predict(stand_test_feats[ii])]
            
    if otherClassifier:
        result_labels = []
        clf = GaussianNB()
        clf = DecisionTreeClassifier()
        #clf = RandomForestClassifier()
        clf.fit(stand_train_feats, all_labels[0:len(stand_train_feats)]) 
        result_labels = []
        for ii in range(0,len(test_features)): #was test_features
            result_labels = result_labels + [clf.predict(stand_test_feats[ii])]



    # Load the test file ground truth
    pkl_file  = open(testGTFilename,'rb')
    mydict    = pickle.load(pkl_file)
    classes   = mydict['classes']
    locations = mydict['locations']


    # Displaying result
    if charts:
        io.imshow(test_img_binary)
        ax = plt.gca()
    counter = 0
    for ii in range(0,len(locations)):
        locs = locations[ii]
        posx = locs[0]
        posy = locs[1]
        for jj in range(0,len(test_regions)):
            props = test_regions[jj]
            minr, minc, maxr, maxc = props.bbox
            if posx <= maxc and posx >= minc and posy <= maxr and posy >= minr and classes[ii] == result_labels[jj]:
                counter = counter + 1
                if charts:
                    ax.add_patch(Rectangle((minc,  minr),  maxc - minc,  maxr - minr, 
                                 fill=False, edgecolor='green', linewidth=1))
                break
            elif posx <= maxc and posx >= minc and posy <= maxr and posy >= minr:
                if charts:
                    ax.add_patch(Rectangle((minc,  minr),  maxc - minc,  maxr - minr, 
                                 fill=False, edgecolor='red', linewidth=1))
                break
            
   # print counter
    if charts:
        plt.title('Recognition results for test file ' + testFilename + ' - ' + str(counter*100/len(locations)) + '%')
        plt.savefig('hysteresisThresholdMethod_'+testFilename[0:len(testFilename)-4]+'.png')
        io.show()



