import cv2
import sys
from os import listdir
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import data
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn import svm
from skimage.feature import local_binary_pattern
import mr8
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, chain

#preparing the dataset
pathSand = os.path.join(os.path.dirname(__file__),'sandresize')
pathPebbles = os.path.join(os.path.dirname(__file__),'gravelresize')
imagesSand = [ os.path.join(pathSand,f) for f in listdir(pathSand) ]
imagesPebbles = [ os.path.join(pathPebbles,f) for f in listdir(pathPebbles) ]
X = np.empty([1,2])
for i in imagesSand[:-5]:
    tuple = np.array([i, 0])
    X = np.vstack([X,tuple])
for i in imagesPebbles[:-5]:
    tuple = np.array([i, 1])
    X = np.vstack([X,tuple])
#print X
   
#test data sepration
test =  np.empty([1,2])  
for i in imagesSand[-5:]:
    tuple = np.array([i, 0])
    test = np.vstack([test,tuple])
for i in imagesPebbles[-5:]:
    tuple = np.array([i, 1])
    test = np.vstack([test,tuple])
    
#MR8 features
sigmas = [1, 2, 4]
n_sigmas = len(sigmas)
n_orientations = 6
edge, bar, rot = mr8.makeRFSfilters(sigmas=sigmas,
            n_orientations=n_orientations)

n = n_sigmas * n_orientations

#EXTRACTING THE  TRAIN FEATURES
X_feat = []
Y = []
# settings for LBP
radius = 3
n_points = 8 * radius
for image in X[1:]:
    img = cv2.imread(image[0], cv2.IMREAD_GRAYSCALE)
    #st = 'Orginal' + str(image[1])
    #cv2.imshow(st,img)
    
    filterbank = chain(edge, bar, rot)
    n_filters = len(edge) + len(bar) + len(rot)
    response = mr8.apply_filterbank(img, filterbank)
    fHist = []
    for r in response:
        (hist,_) = np.histogram(r.ravel(), bins=np.arange(0,256), range=(0,256))
        # Normalize the histogram
        #hist /= (hist.sum())
        fHist.extend(list(hist))
    X_feat.append(fHist)
    Y.append(image[1])
    
#EXTRACTING THE  TEST FEATURES  
X_test = []
Y_out = []
for image in test[1:]:
    img = cv2.imread(image[0], cv2.IMREAD_GRAYSCALE)
    st = 'Orginal' + str(image[1])
    #cv2.imshow(st,img)
    #cv2.waitKey(0)
    filterbank = chain(edge, bar, rot)
    n_filters = len(edge) + len(bar) + len(rot)
    response = mr8.apply_filterbank(img, filterbank)
    fHist = []
    for r in response:
        cv2.imshow(st,r)
        cv2.waitKey(0)
        (hist,_) = np.histogram(r.ravel(), bins=np.arange(0,256), range=(0,256))
        # Normalize the histogram
        #hist /= (hist.sum())
        fHist.extend(list(hist))
    X_test.append(fHist)
    Y_out.append(image[1])
print X_test
print Y_out    

#USING THE CLASSFIER
clf = svm.SVC()
clf.fit(X_feat,Y)
print clf.score(X_test,Y_out)
for i in range(len(X_test)):
    res = (clf.predict(X_test[i])[0])==Y_out[i]
    if not res:
        print test[i+1][0]

