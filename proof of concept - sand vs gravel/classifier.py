import cv2
import sys
from os import listdir
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import data
from sklearn.model_selection import train_test_split
from sklearn import svm

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

#EXTRACTING THE  TRAIN FEATURES
X_feat = np.empty([1,3])
Y = []
for image in X[1:]:
    img = cv2.imread(image[0], cv2.IMREAD_GRAYSCALE)
    glcm = greycomatrix(img, [5], [0], 256, symmetric=True, normed=True)
    tuple = np.array([  greycoprops(glcm, 'contrast')[0, 0], greycoprops(glcm, 'correlation')[0, 0],greycoprops(glcm, 'ASM')[0, 0] ] )
    X_feat = np.vstack([X_feat,tuple])
    Y.append(image[1])
    
#EXTRACTING THE  TEST FEATURES  
X_test = np.empty([1,3])
Y_out = []
for image in test[1:]:
    img = cv2.imread(image[0], cv2.IMREAD_GRAYSCALE)
    glcm = greycomatrix(img, [5], [0, np.pi/4, np.pi/2, 3 * np.pi/4], 256, symmetric=True, normed=True)
    tuple = np.array([  greycoprops(glcm, 'contrast')[0, 0], greycoprops(glcm, 'correlation')[0, 0],greycoprops(glcm, 'ASM')[0, 0] ] )
    X_test = np.vstack([X_test,tuple])
    Y_out.append(image[1])
print X_test
print Y_out    

#USING THE CLASSFIER
clf = svm.SVC()
clf.fit(X_feat[1:],Y)
print clf.predict(X_test[1:])
print clf.score(X_test[1:],Y_out)

