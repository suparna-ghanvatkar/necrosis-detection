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
from sklearn.neighbors import KNeighborsClassifier

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
print X
   
#test data sepration
test =  np.empty([1,2])  
for i in imagesSand[-5:]:
    tuple = np.array([i, 0])
    test = np.vstack([test,tuple])
for i in imagesPebbles[-5:]:
    tuple = np.array([i, 1])
    test = np.vstack([test,tuple])

#EXTRACTING THE  TRAIN FEATURES
X_feat = np.empty([1,26])
Y = []
# settings for LBP
radius = 3
n_points = 8 * radius
for image in X[1:]:
    img = cv2.imread(image[0], cv2.IMREAD_GRAYSCALE)
    #st = 'Orginal' + str(image[1])
    #cv2.imshow(st,img)
    
    lbp = local_binary_pattern(img, n_points, radius, method = 'default')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
 
		# normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    print hist.shape
    X_feat = np.vstack([X_feat,hist])
    Y.append(image[1])
    
#EXTRACTING THE  TEST FEATURES  
X_test = np.empty([1,26])
Y_out = []
for image in test[1:]:
    img = cv2.imread(image[0], cv2.IMREAD_GRAYSCALE)
    st = 'Orginal' + str(image[1])
    #cv2.imshow(st,img)
    #cv2.waitKey(0)
    lbp = local_binary_pattern(img, n_points, radius, method = 'default')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
 
		# normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    print hist.shape
    X_test = np.vstack([X_test,hist])
    Y_out.append(image[1])
print X_test
print Y_out    
print Y
#USING THE CLASSFIER
clf = svm.SVC(kernel = 'rbf')
clf.fit(X_feat[1:],Y)
print clf.predict(X_test[1:])
print clf.score(X_test[1:],Y_out)
neigh = KNeighborsClassifier(n_neighbors=20)
neigh.fit(X_feat[1:],Y) 
print neigh.predict(X_test[1:])
print neigh.score(X_test[1:],Y_out)


