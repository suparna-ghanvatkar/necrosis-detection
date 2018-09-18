# import the necessary packages
#from pyimagesearch.helpers import pyramid
#from pyimagesearch.helpers import sliding_window
import argparse
import time
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
import imutils
from scipy.stats import skew
import math

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

 
def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image
 
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
 
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
 
		# yield the next image in the pyramid
		yield image

#preparing the dataset
pathNecrosis = os.path.join(os.path.dirname(__file__),'necrosis_train')
pathNonNecrosis = os.path.join(os.path.dirname(__file__),'non-necrosis_test')
imagesNecrosis = [ os.path.join(pathNecrosis,f) for f in listdir(pathNecrosis) ]
imagesNonNecrosis = [ os.path.join(pathNonNecrosis,f) for f in listdir(pathNonNecrosis) ]
X = np.empty([1,2])
for i in imagesNecrosis:
    tuple = np.array([i, 0])
    X = np.vstack([X,tuple])
for i in imagesNonNecrosis:
    tuple = np.array([i, 1])
    X = np.vstack([X,tuple])
#print X

(winW, winH) = (500, 500)	#sliding window size

#training features
X_feat = np.empty([1,7])
Y = []
# settings for LBP
radius = 3
n_points = 8 * radius
for image in X[1:]:
    img = cv2.imread(image[0])
    hs = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h,l,s = cv2.split(hs)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #st = 'Orginal' + str(image[1])
    #cv2.imshow(st,img)
    #img = cv2.resize(img,(100,100))

    lbp = local_binary_pattern(img, n_points, radius, method = 'uniform')
    raveled = lbp.ravel()
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    threshs = cv2.adaptiveThreshold(s,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    threshv = cv2.adaptiveThreshold(l,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,2)
    #print thresh.shape
    num_total = len(s.ravel())
    num_white = np.count_nonzero(s==255)
    whiteprops = 1.0*num_white/num_total
    num_total = len(l.ravel())
    num_white = np.count_nonzero(l==255)
    whitepropv = 1.0*num_white/num_total
    #num_total = len(s.ravel())
    #num_nonz = np.count_nonzero(s)
    #num_zeros = num_total - num_nonz
    #whiteprop = 1.0*num_zeros/num_total
		# normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum())
    mean = raveled.mean()
    variance = raveled.var()
    energy = reduce(lambda x,y: x+y,[i**2 for i in hist])
    entropy = reduce(lambda x,y: x+y,[i*math.log(i) for i in hist if i!=0])
    skewness = skew(raveled)
    feat = [energy, entropy, mean, variance, skewness,whiteprops,whitepropv]
    print image[0],feat
    X_feat = np.vstack([X_feat,feat])
    Y.append(image[1])


#print X_feat
neigh = KNeighborsClassifier(n_neighbors=20)
#neigh = svm.SVC(kernel = 'rbf')
neigh.fit(X_feat[1:],Y)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# load the image and define the window width and height
image = cv2.imread(args["image"])

img_no = 0
# loop over the image pyramid
for resized in pyramid(image, scale=1.5):
	clone = resized.copy()
	mask = np.zeros(clone.shape)
	masked = np.zeros(clone.shape)
	# loop over the sliding window for each layer of the pyramid
	for (x, y, window) in sliding_window(resized, stepSize=300, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
 		
		image = resized[y:y+winH, x:x+winW]
		img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		hs = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
		h,l,s = cv2.split(hs)
		#cv2.imwrite("resized.png",img)
		#img = cv2.resize(img,(100,100))
		lbp = local_binary_pattern(img, n_points, radius, method = 'uniform')
		(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
	
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + 1e-7)
        #X_test = np.array(['img',hist])
		num_total = len(s.ravel())
		num_white = np.count_nonzero(s==255)
		whiteprops = 1.0*num_white/num_total
		num_total = len(l.ravel())
		num_white = np.count_nonzero(l==255)
		whitepropv = 1.0*num_white/num_total
		mean = raveled.mean()
		variance = raveled.var()
		energy = reduce(lambda x,y: x+y,[i**2 for i in hist])
		entropy = reduce(lambda x,y: x+y,[i*math.log(i) for i in hist if i!=0])
		skewness = skew(raveled)
		feat = np.array([energy, entropy, mean, variance, skewness, whiteprops, whitepropv])
		
		pred = neigh.predict([feat])
		#pred = True
		if pred=='0':		
		# since we do not have a classifier, we'll just draw the window
			cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 0, 0), 2)
			mask[y:y+winH, x:x+winW,1] = 255
		#else:
			#cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
	cv2.imwrite("mask"+str(img_no)+".png",mask)
	mask = cv2.imread("mask"+str(img_no)+".png")
	masked = cv2.addWeighted(resized,0.8,mask,0.2,0)
	cv2.imwrite("masked"+str(img_no)+".png",masked)
	cv2.imwrite("Window"+str(img_no)+".png", clone)
	img_no = img_no+1
	cv2.waitKey(1)
	time.sleep(0.025)

