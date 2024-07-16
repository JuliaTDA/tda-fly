# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 02:13:54 2024

@author: SergioNote
"""

import numpy as np
import os
import pandas as pd

##################################################################
##################################################################
##   reduce the diagram, removing short intervals
##################################################################
##################################################################


ripspath = './ripsdgm'
reddenpath = './red_ripsdgm'
if not os.path.exists(reddenpath):
  os.mkdir(reddenpath)
  print("Folder %s created!" % reddenpath)
else:
  print("Folder %s already exists" % reddenpath)

dendfiles = os.listdir(ripspath)

#########################################
# 1 - read the diagram
# 2 - detect short intervals
# 3 - remove said intervals
###########################################



def thin_den(dendogram):
    rip_list = []
    for n in range(len(dendogram)):
        # print(n)
        # print(dendogram[n])
        comp = dendogram[n][1]-dendogram[n][0]
        # print(n, comp)
        ## comp >5 seens to be the ideal
        if comp> 5:          ##### parametro ajustavel aqui ######
            rip_list.append(dendogram[n])
    red_dendogram = np.array(rip_list)      
    print(len(dendogram), len(red_dendogram))      
    return red_dendogram

# 

for dend in dendfiles:
    data1=pd.read_csv(ripspath+'/'+dend)
    dendogram=np.array(data1)
    red_dendogram = thin_den(dendogram)
    diagram_df = pd.DataFrame(red_dendogram)
    diagram_df.to_csv(reddenpath+'/'+dend,header = None,index = None)
    




