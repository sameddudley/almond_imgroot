#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 17:33:15 2024

@author: sdudley
"""

#this code goes through a directory and adaptively thresholds the images in it

#this uses adaptive thresholding to make the initial binary image
#parameters to change of cv2.adaptiveThreshold are at the end of the arguments: the first must be multiples of 10+1, so 31, 51, 101,151,201 etc, and this controls how large of an area
#the second number is how much noise it takes out, too high a number and you lose fine roots but too small and it will not remove background

import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage
from skimage import measure
import os





#check for or create results folder
cwd = os.getcwd()
results = os.path.join(cwd, 'cropped_tifs')
if os.path.isdir(results):
    print("results folder already exists")
else:
    print("making results folder")
    os.mkdir(results)


dir_path = input('enter path to directory of images to convert to tif and crop')
print(dir_path)



for image_name in os.listdir(dir_path):
    
    print("processing " + image_name)
    #pathname to the image to process
    original_pathname = dir_path + '/' + image_name

    #open and threshold image
    img = cv2.imread(original_pathname, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"


    y1, y2, x1, x2 = 675, 3180, 1392, 3756
    crop = img[y1:y2, x1:x2]

    
    #save segmented image
    save_path = results + '/' + image_name.split('.')[0] + '.tif'
    cv2.imwrite(save_path, crop)
    
print("Finished")