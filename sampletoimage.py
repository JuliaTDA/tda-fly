# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 04:17:52 2024

@author: SergioNote
"""



# import glob
import os
# import sys
# from PIL import Image, ImageOps
import cv2
# import pathlib
import numpy as np
# from matplotlib import pyplot as plt
# import matplotlib.image as matimg
import pandas as pd
# import random
# from ripser import Rips
# import ripser
# import persim
# from persim import plot_diagrams
# import scipy
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





samplepath = './samples'

samplefiles = os.listdir(samplepath)

samplelist = []
for samplefile in samplefiles:
    if samplefile.endswith('.csv'):
        samplelist.append(samplefile)


anewpath = './image_sample'

# check whether directory already exists BEFORE trying to create file
if not os.path.exists(anewpath):
  os.mkdir(anewpath)
  print("Folder %s created!" % anewpath)
else:
  print("Folder %s already exists" % anewpath)  
  clear_folder(anewpath)


for samplefile in samplelist:
    data = np.zeros((300, 800, 1), dtype=np.int64)
    data[0:300, 0:800] = [256]
    # print(data[5,6])
    csv_path = samplepath +'/'+samplefile
    sampledata1 = pd.read_csv(csv_path)
    sample = np.array(sampledata1)
    for dot in sample:
        # print(dot[0],dot[1])
        # print(data[dot[0],dot[1]])
        data[dot[0],dot[1]] = 0
    cv2.imwrite(anewpath +'/'+samplefile+'.png', data)
