# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 10:16:59 2024

@author: SergioNote
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageOps

## reshape images to a single pattern



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

# check whether directory already exists BEFORE trying to create file
if not os.path.exists(newpath):
  os.mkdir(newpath)
  print("Folder %s created!" % newpath)
else:
  print("Folder %s already exists" % newpath)        

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
    imageBox = image.getbbox()
    cropped = image.crop(imageBox)
    x, y =cropped.size
    # out = im.resize(size)
    print(x,y)
    if (x,y) != size :
        reim = ImageOps.pad(cropped, size, color="#f00")#.save("imageops_pad.png")
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




