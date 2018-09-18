import cv2
import sys
from os import listdir
import os
import cvutils
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
X_test = []
X_name = []
y_test = []
# settings for LBP
radius = 3
n_points = 8 * radius
for image in X[1:]:
    img = cv2.imread(image[0], cv2.IMREAD_GRAYSCALE)
    st = 'Orginal' + str(image[1])
    #cv2.imshow(st,img)
    
    lbp = local_binary_pattern(img, n_points, radius, method = 'uniform')
    # Calculate the histogram
    x = itemfreq(lbp.ravel())
    # Normalize the histogram
    hist = x[:, 1]/sum(x[:, 1])
    # Append image path in X_name
    X_name.append(image[0])
    # Append histogram to X_name
    X_test.append(hist)
    # Append class label in y_test
    y_test.append(str(image[1]))
    
#EXTRACTING THE  TEST FEATURES  

for image in test[1:]:
    im_gray = cv2.imread(image[0], cv2.IMREAD_GRAYSCALE)
    st = 'Orginal' + str(image[1])
    # Uniform LBP is used
    radius = 3
    # Number of points to be considered as neighbourers 
    no_points = 8 * radius
    lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
    # Calculate the histogram
    x = itemfreq(lbp.ravel())
    # Normalize the histogram
    hist = x[:, 1]/sum(x[:, 1])
    # Display the query image
    cvutils.imshow("** Query Image -> {}**".format(image[0]), im_gray)
    results = []
    # For each image in the training dataset
    # Calculate the chi-squared distance and the sort the values
    for index, x in enumerate(X_test):
        score = cv2.compareHist(np.array(x, dtype=np.float32), np.array(hist, dtype=np.float32), cv2.cv.CV_COMP_CHISQR)
        results.append((X_name[index], round(score, 3)))
    results = sorted(results, key=lambda score: score[1])
    # Display the results
    nrows = 2
    ncols = 3
    fig, axes = plt.subplots(nrows,ncols)
    fig.suptitle("** Scores for -> {}**".format(image[0]))
    for row in range(nrows):
        for col in range(ncols):
            axes[row][col].imshow(cv2.cvtColor(cv2.imread(results[row*ncols+col][0]), cv2.COLOR_BGR2RGB))
            axes[row][col].axis('off')
            axes[row][col].set_title("Score {}".format(results[row*ncols+col][1]))

