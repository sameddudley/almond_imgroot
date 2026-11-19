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
    

def main(image_path, path_for_saving):
    #read in the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    assert img is not None, "file could not be read, check with os.path.exists()"
    
    thick_thresh = threshold(img, 201, 8)
    
    
    
    """
    # Step 3: Find contours in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Create a new blank image to fill the contours
    filled_image = np.zeros_like(mask)

    # Step 5: Fill each contour (white color)
    cv2.drawContours(filled_image, contours, -1, (255), thickness=cv2.FILLED)
    """

    
    
    
    #final_mask = cv2.bitwise_and(mask, filled_image)
    
    
    #saves_image
    cv2.imwrite(path_for_saving, thick_thresh)
    
    
##############################
#this block excecutes the code
##############################

dir_path = input('enter path to directory of images')
#dir_path = '/Users/sdudley/Library/CloudStorage/Box-Box/Bartlett Lab/Drought tolerant rootstock project/QTL Experiment/previous_testing_data/real_images_thresh'
input("Do you have the right directory??")
for image_name in os.listdir(dir_path):
    
    print("processing " + image_name)
    #pathname to the image to process
    original_pathname = dir_path + '/' + image_name
    
    #create path to save segmented image
    prefix = image_name.split('.')
    save_path = dir_path + '/' + prefix[0] + '_' + 'binary.tif'
    
    #segment current image in loop
    main(original_pathname, save_path)
    
print("Finished")