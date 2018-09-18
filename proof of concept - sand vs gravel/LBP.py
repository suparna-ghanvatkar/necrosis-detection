import cv2
from skimage.feature import local_binary_pattern


img = cv2.imread('C:\Users\NISHANT\Desktop\Sem 3\RE\Sand vs Pebble\G5tn-0.jpg', cv2.IMREAD_GRAYSCALE)

# settings for LBP
radius = 3
n_points = 8 * radius
lbp = local_binary_pattern(img, n_points, radius, method = 'default')
cv2.imwrite("FDSF2.jpg",lbp)
cv2.waitKey(0)