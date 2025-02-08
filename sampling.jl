
using PyCall
@pyimport PIL.Image as Image 
@pyimport PIL.ImageOps as ImageOps 
shutil = pyimport("shutil")
os = pyimport("os")
np = pyimport("numpy")
cv2 = pyimport("cv2")
using Printf
using NumPyArrays
using MatrixNetworks
using CSV

import Glob as glob
@pyimport sys
# from PIL import Image, ImageOps
# import cv2
@pyimport pathlib
# import numpy as np

@pyimport matplotlib.image as matimg
@pyimport pandas as pd
@pyimport random
# from ripser import Rips
using Ripserer
# @pyimport ripser
@pyimport persim
@pyimport scipy
using DataFrames
using Distances


DISTANCE_MIN = 5

####################################
## empty the indicated folder
function clear_folder(folder)
    # folder = '/path/to/folder'
    for filename in os.listdir(folder)
        file_path = os.path.join(folder, filename)
        try
            rm(file_path, force=true, recursive=true) #rm = remove
        catch e #Exception as e
            @printf("Failed to delete %s. Reason: %s", file_path, e)
        end
    end
end

    
#######################################################################
### creates a new folder
### newpath = './newfoldername'
### if it already exists, empties the folder
function mkfolder(newpath)
    if os.path.exists(newpath)
        println("Folder $newpath already exists")
        clear_folder(newpath)
    else
        os.mkdir(newpath)# make directory
        println("Folder $newpath created!")
    end
end






newsplepath = "./newsamples"

mkfolder(newsplepath)



csvpath = "./csvfles"
csvfiles = os.listdir(csvpath)



for csvfile in csvfiles
    csv_path = csvpath *"/"* csvfile
    # local_df = pd.read_csv(csv_path)
    # mydata = DataFrame(CSV.File(csv_path))
    local_df = DataFrame(CSV.File(csv_path))
    # println(local_df)
    # coords = np.array(df)
    # df_size = size(df)[1]
    first = true 
    list_ok = Array[]
    for row in eachrow(local_df) #  index, row in df.iterrows()
        # println(row[1])
        if first
            # list_ok.append([row[1], row[2]])
            push!(list_ok, [row[1], row[2]])
            first = false
            continue
        end    
        point = [row[1], row[2]]
        if any((euclidean(point, point_ok) < DISTANCE_MIN for point_ok in list_ok))
            continue
        end    
        # list_ok.append(point)
        push!(list_ok,point)
    end
    # df_out = pd.DataFrame(list_ok)
    # df_out.to_csv(newsplepath+'/'+csvfile,header = None,index = None)
    sample_df = DataFrame(NamedTuple{(:a, :b)}.(list_ok))
    CSV.write(newsplepath*"/"*csvfile, sample_df)
    println(size(sample_df))
end

print("end")
