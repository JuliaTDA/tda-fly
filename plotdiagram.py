
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

samplepath = './samples'
samplefiles = ['./samples/bi_gray_Asilidae 2.csv']
for sdata in samplefiles:
    datapath = sdata
    data = pd.read_csv(datapath)
    datas = np.array(data)
    # diagram = rips.fit_transform(data)
    diagram = ripser.ripser(datas)['dgms'][1] #create H1 diagram
    red_dendogram = thin_den(diagram,cparm)
    plot_diagrams(diagram, show=True)
    plot_diagrams(red_dendogram, show=True)
