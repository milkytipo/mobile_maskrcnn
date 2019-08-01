import os
import cv2
import numpy as np
import skimage
import random
ROOT_DIR = os.path.abspath("")
IMAGE_DIR = os.path.join(ROOT_DIR, "images2")
file_names = next(os.walk(IMAGE_DIR))[2]
img = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

print(file_names)
cv2.imshow('raw ',img) 

kernel = np.ones((5,5),np.uint8)

erosion = cv2.erode(img,kernel, iterations=1)

cv2.imshow('erosion',erosion) 

dilation = cv2.dilate(erosion,kernel, iterations=1)

cv2.imshow('dilation',dilation) 

cv2.waitKey(0)
cv2.destroyAllWindows()
