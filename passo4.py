# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:29:26 2024

@author: SergioNote
"""


################################################################
#### caution: slow, very time consuming
################################################################

import os
# import sys
# import cv2 
# import pathlib
import pandas as pd
import persim
# import ripser
# import scipy

import numpy as np
# import tadasets
# import matplotlib.pyplot as plt

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




##try to make the dendogram
##############################

##create distancematrix folder
################################

# ripspath = './ripsdgm'  #this is the file with the H1 diagrams
ripspath = './red_ripsdgm'  #this is the file with the H1 diagrams
# reddenpath = './red_ripsdgm'
distancemtxpath = './distancemtx'
if not os.path.exists(distancemtxpath):
    os.mkdir(distancemtxpath)
    print("Folder %s created!" % distancemtxpath)
else:
    print("Folder %s already exists" % distancemtxpath)
    clear_folder(distancemtxpath)
  
  
ripsfiles = os.listdir(ripspath)


############################################################
## segundo exemplo for bottleneck distances (for reference)


# dgm1 = np.array([
#     [0.6, 1.05],
#     [0.53, 1],
#     [0.5, 0.51]
# ])
# dgm2 = np.array([
#     [0.55, 1.1],
#     [0.8,0.9]
# ])

# d, matching = persim.bottleneck(
#     dgm1,
#     dgm2,
#     matching=True
# )

# d = persim.bottleneck(
#     dgm1,
#     dgm2,
#     matching=False
# )

# persim.bottleneck_matching(dgm1, dgm2, matching, 
                           # labels=['Clean $H_1$', 'Noisy $H_1$'])
# plt.title("Distance {:.3f}".format(d))
# plt.show()


#If we print the matching, we see that the diagram 1 point at index 0 gets 
#matched to diagram 2’s point at index 1, and vice versa. The final row with 2,
# -1 indicates that diagram 1 point at index 2 was matched to the diagonal

# print(matching)
# print(d)

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





###########################################################
#### listing diagrams
################################

idx = 0
idxlist = []
for ripsdgm in ripsfiles:
    # dgmpath = ripspath+'/'+ripsdgm
    # dgm = pd.read_csv(dgmpath)
    # dgms = np.array(dgm)
    idxlist.append([idx,ripsdgm])
    print(idx)
    idx = idx+1
ripdgm_df = pd.DataFrame(idxlist)
saveList(idxlist, distancemtxpath+'/'+'namelist.npy')
aaa = loadList(distancemtxpath+'/'+'namelist.npy')
print(aaa)
# ripdgm_df.to_csv(distancemtx+'/'+list,header = None,index = None)


##############################################################
#### distances matrix
########################################################

size = len(idxlist)
print(size)
matrix = np.zeros((size,size))
print(matrix)
# matrix[1,1] = 3
# a = matrix[0,0]
# b = matrix[1,1]
# print(a,b)
# print(matrix)
# for n in range(size):
#     idx1, name1 = idxlist[n]
#     dgmpath1 = ripspath+'/'+name1
#     dgm1 = pd.read_csv(dgmpath1)
#     dgms1 = np.array(dgm1)
#     for m in range(size):
#         idx2, name2 = idxlist[m]
#         dgmpath2 = ripspath+'/'+name2
#         dgm2 = pd.read_csv(dgmpath2)
#         dgms2 = np.array(dgm2)
#         d, matching = persim.bottleneck(
#             dgm1,
#             dgm2,
#             matching=True
#         )
#         matrix[m,n] = d
        
# saveList(matrix, distancemtxpath+'/'+'distancematrix.npy')    

###################################################################
### reduced distance matrix
#################################################################

red_mat = []
for n in range(size-1):
    idx1, name1 = idxlist[n]
    dgmpath1 = ripspath+'/'+name1
    dgm1 = pd.read_csv(dgmpath1)
    dgms1 = np.array(dgm1)
    for m in range(n+1,size):
        idx2, name2 = idxlist[m]
        dgmpath2 = ripspath+'/'+name2
        dgm2 = pd.read_csv(dgmpath2)
        dgms2 = np.array(dgm2)
        d, matching = persim.bottleneck(
            dgm1,
            dgm2,
            matching=True
        )
        red_mat.append(d)

reduced_array = np.array(red_mat)
saveList(red_mat, distancemtxpath+'/'+'red_dist_mat.npy')    


#####################################################
#### dataframe dos diagramas com os nomes como indices 
#### nao usei ainda

dataf = []
namelist = []
for ripsdgm in ripsfiles:
    dgmpath = ripspath+'/'+ripsdgm
    dgm = pd.read_csv(dgmpath)
    dgms = np.array(dgm)
    dataf.append([dgms])
    namelist.append(ripsdgm)

df = pd.DataFrame(dataf, index= namelist)

dataframe = pd.DataFrame(matrix,  index=df.index, columns=df.index)

dataframe.to_csv(distancemtxpath+'/'+'distmatrix_df.csv',header = None,
                 index = None)
## obs:nao salva os indices quando salva o dataframe em csv

    

