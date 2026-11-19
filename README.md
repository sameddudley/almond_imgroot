# Almond imgroot
The codebase for image processing pipeline for almond root photos.
Do not alter the source jpgs in any way before this pipeline, it must be run on raw images.

## Clone and install



## Pipeline
Step 1: Crop and convert raw files to tif files  
Step 2: Segment tifs into binary files  
Step 3: Extract data from binary files into networks and summary file  


### Convert and crop images
Copy path to your directory of raw images

    python convert_crop.py

Review output and adjust cropping if needed

### Segment cropped images

    python segment.py
    cropped_tifs/

Review the output to make sure that it is correct

### Extract data from binary images
To extract data from binary images run:

    python almond_imgroot.py
    segmented_images/


