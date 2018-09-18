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
from sklearn import svm, tree
from skimage.feature import local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier
import math
from scipy.stats import skew
from sklearn.model_selection import cross_validate
import sklearn.metrics

#preparing the dataset
pathNecrosis_train = os.path.join(os.path.dirname(__file__),'necrosis_train')
pathNonNecrosis_train = os.path.join(os.path.dirname(__file__),'non-necrosis_train')
imagesNecrosis_train = [ os.path.join(pathNecrosis_train,f) for f in listdir(pathNecrosis_train) ]
imagesNonNecrosis_train = [ os.path.join(pathNonNecrosis_train,f) for f in listdir(pathNonNecrosis_train) ]
pathNecrosis_test = os.path.join(os.path.dirname(__file__),'necrosis_test')
pathNonNecrosis_test = os.path.join(os.path.dirname(__file__),'non-necrosis_test')
imagesNecrosis_test = [ os.path.join(pathNecrosis_test,f) for f in listdir(pathNecrosis_test) ]
imagesNonNecrosis_test = [ os.path.join(pathNonNecrosis_test,f) for f in listdir(pathNonNecrosis_test) ]

X = np.empty([1,2])
for i in imagesNecrosis_train:
    print i
    tuple = np.array([i, 0])
    X = np.vstack([X,tuple])
for i in imagesNonNecrosis_train:
    tuple = np.array([i, 1])
    X = np.vstack([X,tuple])
print X

#test data sepration
test =  np.empty([1,2])
for i in imagesNecrosis_test:
    print i
    tuple = np.array([i, 0])
    test = np.vstack([test,tuple])
for i in imagesNonNecrosis_test:
    tuple = np.array([i, 1])
    test = np.vstack([test,tuple])
print test

#EXTRACTING THE  TRAIN FEATURES
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
    #num_white = (s>250).sum()
    whiteprops = 1.0*num_white/num_total
    num_total = len(l.ravel())
    num_white = np.count_nonzero(l==255)
    #num_white = (l>250).sum()
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
    #print image[0],feat
    X_feat = np.vstack([X_feat,feat])
    Y.append(image[1])

#EXTRACTING THE  TEST FEATURES
X_test = np.empty([1,7])
Y_out = []
img_no = 0
for image in test[1:]:
    img = cv2.imread(image[0])
    print "testing "+image[0]
    hs = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h,l,s = cv2.split(hs)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    st = 'Threshl' + str(img_no) + '.png'
    #cv2.waitKey(0)
    #img = cv2.resize(img,(100,100))
    lbp = local_binary_pattern(img, n_points, radius, method = 'uniform')
    raveled = lbp.ravel()
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    threshs = cv2.adaptiveThreshold(s,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    threshv = cv2.adaptiveThreshold(l,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,2)
    #print thresh.shape
    num_total = len(s.ravel())
    num_white = np.count_nonzero(s==255)
    #num_white = (s>250).sum()
    whiteprops = 1.0*num_white/num_total
    num_total = len(l.ravel())
    num_white = np.count_nonzero(l==255)
    #num_white = (l>250).sum()
    whitepropv = 1.0*num_white/num_total
    #num_nonz = np.count_nonzero(s)
    #num_zeros = num_total - num_nonz
    #whiteprop = 1.0*num_zeros/num_total
		# normalize the histogram
    cv2.imwrite(st,threshv)
    hist = hist.astype("float")
    hist /= (hist.sum())
    mean = raveled.mean()
    variance = raveled.var()
    energy = reduce(lambda x,y: x+y,[i**2 for i in hist])
    entropy = reduce(lambda x,y: x+y,[i*math.log(i) for i in hist if i!=0])
    skewness = skew(raveled)
    feat = [energy, entropy, mean, variance, skewness, whiteprops,whitepropv]
    X_test = np.vstack([X_test,feat])
    Y_out.append(image[1])
    img_no = img_no + 1
#print X_test
#print Y_out
#print Y
#USING THE CLASSFIER
#clf = svm.SVC(kernel = 'rbf')
#clf.fit(X_feat[1:],Y)
#pred = clf.predict(X_test[1:])
#neigh = KNeighborsClassifier(n_neighbors=11)
#neigh = svm.SVC(kernel = 'rbf')
neigh = tree.DecisionTreeClassifier()
neigh.fit(X_feat[1:],Y)
pred = neigh.predict(X_test[1:])
print neigh.score(X_test[1:],Y_out)
TP = 0.0
TN = 0.0
FP = 0.0
FN = 0.0
for i in range(len(pred)):
	img = cv2.imread(test[i+1][0])
	#print pred[i]
	if pred[i] == '0':
		cv2.putText(img, "necrosis", (10,30), cv2.FONT_ITALIC, 1.0, (0,0,255),3)
	else:
		cv2.putText(img, "not necrosis", (10,30), cv2.FONT_ITALIC, 1.0, (0,0,255),3)
	cv2.imwrite(str(i)+".png",img)
	if pred[i]=='0' and pred[i]==str(Y_out[i]):	#True positive
		TP = TP+1
	elif pred[i]=='0' and pred[i]!=str(Y_out[i]):	#Flase positive
		FP = FP+1
	elif pred[i]=='1' and pred[i]==str(Y_out[i]):
		TN = TN+1
	else:
		FN = FN+1
print "TP:",str(TP),"FP:",str(FP),"TN:",str(TN),"FN:",str(FN)
sens = (TP/(TP+FN))
print "sensitivity:", str(sens)
spec = TN/(TN+FP)
print "specificity:", str(spec)
print "precision:", str(TP/(TP+FP))
print "negative predictive rate:", str(TN/(TN+FN))
print "false negative rate:", str(FN/(FN+TP))
print "false positive rate:", str(FP/(FP+TN))
print "false discovery rate:", str(FP/(FP+TP))
print "false ommission rate:", str(FN/(FN+TN))
print "F1 score:", str(2*TP/((2*TP)+FP+FN))
print "Matthews correlation coefficient:", str(((TP*TN)-(FP*FN))/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
print "informedness:", str(sens+spec-1)
#print clf.score(X_test[1:],Y_out)

