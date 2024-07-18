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

ripspath = './ripsdgm'
reddenpath = './red_ripsdgm'
if not os.path.exists(reddenpath):
  os.mkdir(reddenpath)
  print("Folder %s created!" % reddenpath)
else:
  print("Folder %s already exists" % reddenpath)
  clear_folder(reddenpath)

dendfiles = os.listdir(ripspath)

#########################################
# 1 - read the diagram
# 2 - detect short intervals
# 3 - remove said intervals
###########################################

## cut parameter
cparm = 5

def thin_den(dendogram, cparm):
    rip_list = []
    for n in range(len(dendogram)):
        # print(n)
        # print(dendogram[n])
        comp = dendogram[n][1]-dendogram[n][0]
        # print(n, comp)
        ## comp >5 seens to be the ideal
        if comp> cparm:          ##### parametro ajustavel aqui ######
            rip_list.append(dendogram[n])
    red_dendogram = np.array(rip_list)      
    print(len(dendogram), len(red_dendogram))      
    return red_dendogram

# 

for dend in dendfiles:
    data1=pd.read_csv(ripspath+'/'+dend)
    dendogram=np.array(data1)
    red_dendogram = thin_den(dendogram,cparm)
    diagram_df = pd.DataFrame(red_dendogram)
    diagram_df.to_csv(reddenpath+'/'+dend,header = None,index = None)
    
