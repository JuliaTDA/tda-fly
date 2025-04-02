# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 02:33:35 2024

@author: SergioNote
"""

import glob
import os, sys
from PIL import Image, ImageOps
import cv2
import pathlib
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as matimg
import pandas as pd
import random
# from ripser import Rips
import ripser
import persim
from persim import plot_diagrams
import scipy



import shutil

####################################
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
        

# path_names = glob.glob('images/*.jpg')


# print(path_names)


# pathname = 'C:/Users/SergioNote/OneDrive/Desktop/Relearn Python/
# arquivos resgatados/Wings 2/preset dendogram'
# path = "C:/Users/SergioNote/OneDrive/Desktop/Relearn Python\
# arquivos resgatados/Wings 2/preset dendogram"

## the ./ stands for the current working directory. 

pathname = './re_im'

if not os.path.exists(pathname):
  print("Folder %s does not exist" % './re_im')
  sys.exit()

files = os.listdir(pathname)

listimg = []

for file in files:
    # make sure file is an image
    if file.endswith(('.jpg', '.png', 'jpeg')):
        img_path = pathname +'/' +file
        listimg.append(img_path)

# print('files path', listimg)


# os.listdir() returns everything inside a directory
 # -- including both files and directories.
 
 
 #  the ./ stands for the current working directory. 
newpath = './greyscale'

# create new directory named greyscale
# os.mkdir(path)

# check whether directory already exists BEFORE trying to create file


mkfolder(newpath)



############################################################
## todo: resize images if necessary
##
############################################################

# size = (800,300)

# def resize(image):
#     im =Image.open(image)
#     x, y =im.size
#     out = im.resize(size)
#     # print(x,y)
#     # if (x,y) isnot size
        #ImageOps.cover(im, size, color="#f00").save("imageops_pad.png")
        # 
#         # out.save(image)
#     return image


seclist = []

# transforms images into grayscale and saves it in new file
for file in files:
    if file.endswith(('.jpg', '.png', 'jpeg')):
    # Load an color image in grayscale
        image = pathname +'/' +file
        # image = resize(image)
        img = cv2.imread(image,0)
        messygray = 'gray_' + file
        seclist.append(messygray)
        file2 ='gray'+ file + '.png'
        p = pathlib.Path(messygray)
        file2= p.with_name(p.name.split('.')[0]).with_suffix('.png')
        cv2.imwrite(os.path.join(newpath, file2), img)

        # =====================================================================
        # Load an color image in grayscale and (optional) saves a copy
        # =====================================================================
        #img = cv2.imread('euphylidorea_meigenii_limoniidae.jpg',0)
        # cv2.imshow('image',img)
        # k = cv2.waitKey(0)  & 0xFF
        # if k == 27:         # wait for ESC key to exit
        #     cv2.destroyAllWindows()
        # elif k == ord('s'): # wait for 's' key to save and exit
        #     cv2.imwrite('messigray.png',img)
        #     cv2.destroyAllWindows()
        # else: cv2.destroyAllWindows()
    
    

# p = pathlib.Path('/path/to.my/file.foo.bar.baz.quz')
# print(p.with_name(p.name.split('.')[0]).with_suffix('.jpg'))



#################################################################

## binarization

graypath = newpath

grayfiles = os.listdir(graypath)

graylist = []

for file in grayfiles:
    # make sure file is an image
    if file.endswith(('.jpg', '.png', 'jpeg')):
        img_path = graypath +'/' +file
        graylist.append(img_path)

# print('files path', graylist)


binpath = './binaryimages'

mkfolder(binpath)
# if not os.path.exists(binpath):
#   os.mkdir(binpath)
#   print("Folder %s created!" % binpath)
# else:
#   print("Folder %s already exists" % binpath)
#   clear_folder(binpath)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))


for image in grayfiles:
    if image.endswith('.png'):
        img_path = graypath +'/' +image
        img = cv2.imread(img_path,0)
        # erosion = cv2.erode(img,kernel,iterations = 2)
        # blur = cv2.GaussianBlur(img,(5,5),0)
        blur = img
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        bimage = binpath+'/'+'bi_'+image
        cv2.imwrite(bimage,th3)



# =============================================================================
# morphological transformations
# =============================================================================
# opening = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)
# cv2.imwrite('2teste.png',opening)

# closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
# cv2.imwrite('2teste2.png',closing)

# erosion = cv2.erode(opening,kernel,iterations = 1)
# cv2.imwrite('2teste3.png',erosion)



## turn to csv file


csvpath = './csvfles'
mkfolder(csvpath)
# if not os.path.exists(csvpath):
#   os.mkdir(csvpath)
#   print("Folder %s created!" % csvpath)
# else:
#   print("Folder %s already exists" % csvpath)
#   clear_folder(csvpath)

binfiles = os.listdir(binpath)

for image in binfiles:
    bin_path = binpath + '/' + image
    bimg = Image.open(bin_path)
    imageToMatrice = np.asarray(bimg)
    imageMat = matimg.imread(bin_path)
    aaa = imageMat.shape
    if len(aaa)>2:
        if(imageMat.shape[2] == 3):
    # reshape it from 3D matrice to 2D matrice
            imageMat_reshape = imageMat.reshape(imageMat.shape[0],
     									-1)
            print("Reshaping to 2D array:", 
                  imageMat_reshape.shape)
        else:imageMat_reshape = imageMat
    # if image is grayscale
    #####################################
    else:
    # remain as it is
        imageMat_reshape = imageMat
    # ###############################################	
    # # converting it to dataframe.
    # ###############################################
    mat_df = pd.DataFrame(imageMat_reshape)

    ### obs: mat_df.iat[4,3] to acess a value from a cell of a dataframe mat_df

    ################################################
    # exporting dataframe to CSV file.
    #############################################3##
    p = pathlib.Path(image)    
    name2 = p.name.split('.')[0]
    # mat_df.to_csv(csvpath+'/'+name2 +'.csv',
    #  			header = None,
    #  			index = None)
   ##################################################  
   #  Storing (x,y) coordinate of black pixels
   ##############################################
    arr = np.asarray(mat_df)
    black_pixels = np.array(np.where(arr == 0))
    black_pixel_coordinates = list(zip(black_pixels[0],black_pixels[1]))
    blackdotsdf = pd.DataFrame(black_pixel_coordinates)
    blackdotsdf.to_csv(csvpath+'/'+name2 +'.csv',
     			header = None,
     			index = None)


### samples (lesser points)


### samplesize = 7000 is too big
### samplesize around 2000 seens ok

samplesize = 2000

samplepath = './samples'
mkfolder(samplepath)
# if not os.path.exists(samplepath):
#   os.mkdir(samplepath)
#   print("Folder %s created!" % samplepath)
# else:
#   print("Folder %s already exists" % samplepath)
#   clear_folder(samplepath)

csvfiles = os.listdir(csvpath)

for csvfile in csvfiles:
    csv_path = csvpath+'/' + csvfile
    mydata = pd.read_csv(csv_path)
    coords = np.array(mydata)
    list_coords = []
    for point in coords:
        list_coords.append([point[0], point[1]])
    if len(list_coords)<samplesize:
        print('small file ' + csvfile)
    else:
        sample = random.sample(list_coords, samplesize)
        sample_df = pd.DataFrame(sample)
        sample_df.to_csv(samplepath+'/'+csvfile,header = None,index = None)

####################################################
#prototipe to create image from data
# #####################################################
# data = np.zeros((200, 300, 1), dtype=np.int64)
# data[0:200, 0:300] = [256]
# print(data[5,6])
# for dot in sample:
#     # print(dot[0],dot[1])
#     # print(data[dot[0],dot[1]])
#     data[dot[0],dot[1]] = 0
# cv2.imwrite(samplepath+'/'+'.png', data)


##  diagrams

ripspath = './ripsdgm'
mkfolder(ripspath)
# if not os.path.exists(ripspath):
#   os.mkdir(ripspath)
#   print("Folder %s created!" % ripspath)
# else:
#   print("Folder %s already exists" % ripspath)
#   clear_folder(ripspath)
  
samplefiles = os.listdir(samplepath)
# rips = Rips()
  
idx = 0
idxpairs =[]
for sdata in samplefiles:
    datapath = samplepath+'/'+sdata
    data = pd.read_csv(datapath)
    datas = np.array(data)
    idx = idx+1
    # diagram = rips.fit_transform(data)
    diagram = ripser.ripser(datas)['dgms'][1] #create H1 diagram
    # idxpairs.append([idx, diagram])
    # plot_diagrams(diagram, show=True)
    print(idx)
    diagram_df = pd.DataFrame(diagram)
    diagram_df.to_csv(ripspath+'/'+sdata,header = None,index = None)


 
 

