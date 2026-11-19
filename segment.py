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

#takes out the small noise
def remove_outliers(image, area):
    output = cv2.connectedComponentsWithStats(image, 8, cv2.CV_32S)
    remove_labels = np.where(output[2][:, cv2.CC_STAT_AREA] < area)[0]
    matrix = output[1]
    matrix[np.isin(matrix, remove_labels)] = 0
    matrix[matrix > 0] = 255
    return np.uint8(matrix)


def threshold(thresh_img, pixel_neighborhood, C):
    thresh_img = cv2.medianBlur(thresh_img,5)

    #adaptive threshold and invert image
    threshed = remove_outliers(cv2.adaptiveThreshold(thresh_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY, pixel_neighborhood , C), 100)
    threshed = (255-threshed)


    #remove background noise by keeping the largest white portion
    input_mask = threshed
    labels_mask = measure.label(input_mask)                       
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1
    mask = labels_mask
    
    mask = mask*255
    mask = mask.astype('uint8')
    
    return mask
    


    
    
##############################
#this block excecutes the code
##############################



#check for or create results folder
cwd = os.getcwd()
results = os.path.join(cwd, 'segmented_images')
if os.path.isdir(results):
    print("results folder already exists")
else:
    print("making results folder")
    os.mkdir(results)


dir_path = input('enter path to directory of images')
print(dir_path)



for image_name in os.listdir(dir_path):
    
    print("processing " + image_name)
    #pathname to the image to process
    original_pathname = dir_path + '/' + image_name

    #open and threshold image
    img = cv2.imread(original_pathname, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    thick_thresh = threshold(img, 501, 8)

    
    #save segmented image
    save_path = results + '/' + image_name.split('.')[0] + '_' + 'binary.tif'
    cv2.imwrite(save_path, thick_thresh)
    
print("Finished")