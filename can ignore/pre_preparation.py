# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 13:48:39 2024

@author: SergioNote
"""

###############################
### pre-pre-treating of image
###
################################

import os, cv2
import shutil
import sys, pathlib
import numpy as np


########################################################################
## empty the said folder
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
        
#######################################################################        

kernel = np.ones((3,5),np.uint8)

#########################################################################


prepathname = './images'

if not os.path.exists(prepathname):
  print("Folder %s does not exist" % './images')
  sys.exit()

prefiles = os.listdir(prepathname)

prelistimg = []

for file in prefiles:
    # make sure file is an image
    if file.endswith(('.jpg', '.png', 'jpeg')):
        img_path = prepathname +'/' +file
        prelistimg.append(img_path)




newprepath = './prepreimage'
mkfolder(newprepath)


for file in prefiles:
    if file.endswith(('.jpg', '.png', 'jpeg')):
    # Load an color image in grayscale
    ## dont change the zeros
        image = prepathname +'/' +file
        # open the image
        preimg = cv2.imread(image,0)
        # naming stuff
        messygray = 'pre_' + file
        file2 ='pre'+ file + '.png'
        p = pathlib.Path(messygray)
        file2= p.with_name(p.name.split('.')[0]).with_suffix('.png')
        # transformation here
        preimg = cv2.morphologyEx(preimg, cv2.MORPH_OPEN, kernel) # opening
        # preimg = cv2.morphologyEx(preimg, cv2.MORPH_OPEN, kernel) # opening
        ## you can change the numbers : needs to be small integer 
        # preimg = cv2.erode(preimg,kernel,iterations = 1) #erosion engorda
        # preimg = cv2.dilate(preimg,kernel,iterations = 1) # dilation
        # preimg = cv2.morphologyEx(preimg, cv2.MORPH_CLOSE, kernel) #closing
        cv2.imwrite(os.path.join(newprepath, file2), preimg)




