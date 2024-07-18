# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 21:40:58 2024

@author: SergioNote
"""



import glob
import os, sys
from PIL import Image, ImageOps
import cv2
# import pathlib
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as matimg
import pandas as pd
# import random
# from ripser import Rips
import ripser
import persim
from persim import plot_diagrams
# import scipy
from math import sqrt
# import matplotlib.pyplot as plt

import shutil


from scipy.spatial import distance_matrix


import networkx as nx
# G = nx.Graph()


###########################################################
### until find better choice
def pix_dist(X_1, X_2):
    distance = sqrt((X_1[0]-X_2[0])**2+(X_1[1]-X_2[1])**2)
    return distance

###############################
## auxiliary map (saves lists)
def saveList(myList,filename):
    # the filename should mention the extension 'npy'
    np.save(filename,myList)
    print("Saved successfully!")
    
def loadList(filename):
    # the filename should mention the extension 'npy'
    tempNumpyArray=np.load(filename)
    return tempNumpyArray.tolist()




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
        


samplepath = './samples'


ripspath = './ripsdgm'

graphpath = './graph'



samplefiles = os.listdir(samplepath)
# rips = Rips()
sdata=samplefiles[0]  
datapath = samplepath+'/'+sdata
data = pd.read_csv(datapath)
datas = np.array(data)

datapath = graphpath +'/'+'bi_gray_austrolimnophila_unica_limoniidae2.csv'
matrixpd = pd.read_csv(datapath)
matrixnp = np.array(matrixpd)

    # diagram = rips.fit_transform(data)
    # diagram = ripser.ripser(datas)['dgms'][1] #create H1 diagram
    # idxpairs.append([idx, diagram])
    # plot_diagrams(diagram, show=True)
    # print(idx)
    # diagram_df = pd.DataFrame(diagram)
    # diagram_df.to_csv(ripspath+'/'+sdata,header = None,index = None)

size = len(datas)

D= matrixnp
x=[]
y=[]
for n in range(len(datas)):
    x.append(datas[n][0])
    y.append(datas[n][1])
fig = plt.figure(figsize=(80, 25))    
plt.scatter(y,x, marker='o')

for n in range(size):
    # print(n)
    # idx1, name1 = idxlist[n]
    X_1 = datas[n]
    for m in range(n+1,size):
        X_2 = datas[m]
        x_values = [X_1[0], X_2[0]]
        y_values = [X_1[1], X_2[1]]
        if sqrt((X_1[0]-X_2[0])**2+(X_1[1]-X_2[1])**2)<=10:
            plt.plot(y_values, x_values, 'bo', linestyle="-")


plt.show()