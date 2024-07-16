# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 19:29:56 2024

@author: SergioNote
"""





# import sys
# import cv2
import os 
# import pathlib
import pandas as pd
# import persim
# import ripser
import scipy

import numpy as np
# import tadasets
# import matplotlib.pyplot as plt


from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

from matplotlib import pyplot as plt

#########################################################
##objective: create dendogram from the matrix
########################################################

distancemtxpath = './distancemtx'

ripspath = './ripsdgm'  #this is the file with the H1 diagrams
ripsfiles = os.listdir(ripspath)

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

############################################################
############################################################

thelist = loadList('./distancemtx/namelist.npy')
# print(thelist)
###the list of names

# distancematrix = loadList(distancemtxpath+'/'+'distancematrix.npy') 
# print(distancematrix)

#############################################################
#############################################################


################################################################
##### now obtain the dendogram

datapath = distancemtxpath+'/'+'distmatrix_df.csv'
distmat =  pd.read_csv(datapath)

red_dm =  loadList(distancemtxpath+'/'+'red_dist_mat.npy')
y=red_dm
Z =linkage(y, method='single', metric='euclidean', optimal_ordering=False)

fig = plt.figure(figsize=(25, 10))

dn = dendrogram(Z)
plt.show()



