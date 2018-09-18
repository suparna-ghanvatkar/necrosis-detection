import cv2
import sys
from os import listdir
import os
import numpy as np
from PIL import Image
#import pytesseract
#from pytesseract import image_to_string
from matplotlib import pyplot as plt


#path = 'C:\Users\NISHANT\Desktop\Sem 3\RE\Code\Images'
path = os.path.join(os.path.dirname(__file__),os.path.join('..','Images'))
images = [ f for f in listdir(path) ]
print images
img_no = 1

for image in images:
    img = cv2.imread(os.path.join(path,image))
    cv2.imshow('Image', img)
    cv2.waitKey(0)

    hs_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    #split the image into HSV planes
    h, l, s = cv2.split(hs_img)
    """
    h1 = np.copy(h)
    s1 = np.copy(s)
    v1 = np.copy(v)

    #fill the saturation and values copy to 255 for displaying hue in color format
    s1.fill(255)
    v1.fill(255)
    h1 = cv2.cvtColor(cv2.merge([h,s1,v1]), cv2.COLOR_HSV2RGB)
    cv2.imshow('Hue Channels', h1)
    cv2.waitKey(0)

    mser = cv2.MSER_create()
    regions = mser.detectRegions(hs_image)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
    cv2.polylines(im3, hulls, 3, (0,255,0))
    cv2.imshow('norm',im3)
    key = cv2.waitKey(0)
    """


    lower = np.array([0,0, 0])
    upper = np.array([225,225, 225])
    mask = cv2.inRange(hs_img,lower, upper)
    cv2.imwrite('output\Mask'+str(img_no)+'.png',mask)
    result = cv2.bitwise_and(img,img,mask = mask)
    cv2.imwrite('output\Result'+str(img_no)+'.png',result)
    img_no = img_no+1

