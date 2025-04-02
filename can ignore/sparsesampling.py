import pandas as pd
from scipy.spatial.distance import euclidean
import os
import numpy as np



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
 


newsplepath = "./newsamples"

mkfolder(newsplepath)

DISTANCE_MIN = 5


csvpath = "./csvfles"
csvfiles = os.listdir(csvpath)

for csvfile in csvfiles:
    csv_path = csvpath +'/'+ csvfile
    df = pd.read_csv(csv_path)
    # coords = np.array(df)
    first = True
    list_ok = []
    for index, row in df.iterrows():
        if first:
            list_ok.append([row[0], row[1]])
            first = False
            continue
    
        point = [row[0], row[1]]
        if any((euclidean(point, point_ok) < DISTANCE_MIN for point_ok in list_ok)):
            continue
    
        list_ok.append(point)

    df_out = pd.DataFrame(list_ok)
    df_out.to_csv(newsplepath+'/'+csvfile,header = None,index = None)
    

print("end")
