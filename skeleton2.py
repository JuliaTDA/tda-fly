# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:22:14 2024

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


import shutil


from scipy.spatial import distance_matrix


import networkx as nx
G = nx.Graph()


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

mkfolder(graphpath)

 
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
    # diagram = ripser.ripser(datas)['dgms'][1] #create H1 diagram
    # idxpairs.append([idx, diagram])
    # plot_diagrams(diagram, show=True)
    print(idx)
    # diagram_df = pd.DataFrame(diagram)
    # diagram_df.to_csv(ripspath+'/'+sdata,header = None,index = None)



###################################################################
### reduced distance matrix
#################################################################

size = len(datas)
print(size)
red_mat = []
for n in range(size):
    # print(n)
    # idx1, name1 = idxlist[n]
    X_1 = datas[n]
    for m in range(n+1,size):
        X_2 = datas[m]
        d = pix_dist(X_1,X_2)
        red_mat.append(d)
    # print(datas[n])
#     for m in range(n+1,size):
#         idx2, name2 = idxlist[m]
#         dgmpath2 = ripspath+'/'+name2
#         dgm2 = pd.read_csv(dgmpath2)
#         dgms2 = np.array(dgm2)
#         d, matching = persim.bottleneck(
#             dgm1,
#             dgm2,
#             matching=True
#         )
#         red_mat.append(d)

reduced_array = np.array(red_mat)
saveList(red_mat, graphpath +'/'+'red_dist_mat.npy')    


#########################################################
### 

D= distance_matrix(datas, datas)
# D=np.zeros((size,size))
# for n in range(size):
#     # print(n)
#     # idx1, name1 = idxlist[n]
#     X_1 = datas[n]
#     for m in range(size):
#         X_2 = datas[m]
#         D[n,m]=sqrt((X_1[0]-X_2[0])**2+(X_1[1]-X_2[1])**2)
distmat = pd.DataFrame(D)
distmat.to_csv(graphpath+'/'+sdata,header = None,index = None)
print('matrix created')

for n in range(size):
    X_1 = datas[n]
    # nodex = [X_1[0], X_1[1]]
    G.add_node(n,object=X_1)

print('nodes created')

# print(G.number_of_nodes())

for n in range(size):
    # print(n)
    # idx1, name1 = idxlist[n]
    X_1 = datas[n]
    for m in range(n+1,size):
        X_2 = datas[m]
        if D[n,m]>5:
            G.add_edge(n,m)
            

print('edges created')