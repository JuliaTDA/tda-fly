# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:02:22 2024

@author: SergioNote
"""

import cv2
from PIL import Image, ImageOps
import numpy as np
import os




import shutil


####################################
## empty the indicated folder
def clear_folder(folder):
# folder = '/path/to/folder'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

#######################################################################
### creates a new folder
### newpath = './newfoldername'
### if it already exists empties the folder
def mkfolder(newpath):
# check whether directory already exists BEFORE trying to create file
    if not os.path.exists(newpath):
        os.mkdir(newpath)
        print("Folder %s created!" % newpath)
    else:
        print("Folder %s already exists" % newpath)
        clear_folder(newpath)





# Trim Whitespace Section
def trim_whitespace_image(image):
    # Convert the image to grayscale
    gray_image = image.convert('L')

    # Convert the grayscale image to a NumPy array
    img_array = np.array(gray_image)
    # img_array = np.array(image)

    # Apply binary thresholding to create a binary image 
    # (change the value here default is 250)    ↓
    _, binary_array = cv2.threshold(img_array, 250, 255, cv2.THRESH_BINARY_INV)

    # Find connected components in the binary image
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_array)

    # Find the largest connected component (excluding the background)
    largest_component_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    largest_component_mask = (labels == largest_component_label).astype(np.uint8) * 255

    # Find the bounding box of the largest connected component
    x, y, w, h = cv2.boundingRect(largest_component_mask)

    # Crop the image to the bounding box
    cropped_image = image.crop((x, y, x + w, y + h))

    return cropped_image


######################################################
## list of images in the images file
## if the file isnot ./images change it here
######################################################
pathname = './images'
files = os.listdir(pathname)

listimg = []

for file in files:
    # make sure file is an image
    if file.endswith(('.jpg', '.png', 'jpeg')):
        img_path = pathname +'/' +file
        listimg.append(img_path)
        
        
        
         
######################################################################
# create new directory named re_im
# os.mkdir(path)        




newpath = './re_im'

mkfolder(newpath)
         

######################################################################
#######################################################################
#$ read image files - resize then save in re_im file
########################################################################
#######################################################################



### testing imageopen and img.size
# for file in listimg:
#     # image = pathname +'/' +file
#     # image = resize(image)
#     img=Image.open(file)
#     x, y =img.size
#     print(x,y)

# for name in listimg:
#     print(name) 


#######################################################################
## 1 - crop the images (remove white border)
## 2 - resize to the desired size
## 3 - save
######################################################################



############################################################
## todo: resize images if necessary
##
############################################################

size = (800,300)

## function that resize the image 
def resize(image):
    image =Image.open(image)
    cropped = trim_whitespace_image(image)
    # imageBox = image.getbbox()
    # cropped = image.crop(imageBox)
    x, y =cropped.size
    # out = im.resize(size)
    print(x,y)
    if (x,y) != size :
        reim = ImageOps.pad(cropped, size, color="#fff")#.save("imageops_pad.png")
        # reim = ImageOps.pad(cropped, size, color="#fff")#.save("imageops_pad.png")
        # out.save(image)
    else:
        reim = cropped
    return reim

for file in files:
    if file.endswith(('.jpg', '.png', 'jpeg')):
        image = pathname +'/' +file
        reimg = resize(image)
        savefile = newpath +'/'+file
        reimg.save(savefile)
       
        
        
        
        
        
        
        
        
        
        
