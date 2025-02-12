"""read all files """


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

## the ./ stands for the current working directory. 

pathname = "./re_im"

if !os.path.exists(pathname)
    @printf("Folder %s does not exist", "./re_im")
    return
end


files = os.listdir(pathname)

listimg = []

for file in files
    # make sure file is an image
    if endswith(file, r".jpg|.png|jpeg")
        img_path = pathname *"/" *file
        push!(listimg, img_path) #this adds img_path to listimg #in place of list.append
    end
end

newpath = "./greyscale"
# create new directory named greyscale
# os.mkdir(path)
mkfolder(newpath)


seclist = []


# transforms images into grayscale and saves it in new file
for file in files
    if endswith(file, r".jpg|.png|jpeg")
    # Load an color image in grayscale
        image = pathname *"/" *file
        # image = resize(image)
        img = cv2.imread(image,0)
        messygray = "gray_" * file
        push!(seclist, messygray)
        # file2 ="gray"* file * ".png"
        # p = pathlib.Path(messygray)
        file2= rsplit(messygray,".";limit=2)[1]*".png"
        cv2.imwrite(os.path.join(newpath, file2), img)

        # =====================================================================
        # Load an color image in grayscale and (optional) saves a copy
        # =====================================================================
        #img = cv2.imread('euphylidorea_meigenii_limoniidae.jpg',0)
        # cv2.imshow('image',img)
        # k = cv2.waitKey(0)  & 0xFF
        # if k == 27:         # wait for ESC key to exit
        #     cv2.destroyAllWindows()
        # elif k == ord('s'): # wait for 's' key to save and exit
        #     cv2.imwrite('messigray.png',img)
        #     cv2.destroyAllWindows()
        # else: cv2.destroyAllWindows()
    end
end
println("saved images in grayscale")


#################################################################

## binarization

graypath = newpath

grayfiles = os.listdir(graypath)

graylist = []

for file in grayfiles
    # make sure file is an image
    if endswith(file, r".jpg|.png|.jpeg")
        img_path = graypath *"/" *file
        push!(graylist, img_path)
    end
end
# print('files path', graylist)


binpath = "./binaryimages"

mkfolder(binpath)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))



for image in grayfiles
    if endswith(image, ".png")
        img_path = graypath *"/" *image
        img = cv2.imread(img_path,0)
        # erosion = cv2.erode(img,kernel,iterations = 2)
        # blur = cv2.GaussianBlur(img,(5,5),0)
        blur = img
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        bimage = binpath*"/"*"bi_"*image
        cv2.imwrite(bimage,th3)
    end
end

println("saved binary images")


# =============================================================================
# morphological transformations
# =============================================================================
# opening = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)
# cv2.imwrite('2teste.png',opening)

# closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
# cv2.imwrite('2teste2.png',closing)

# erosion = cv2.erode(opening,kernel,iterations = 1)
# cv2.imwrite('2teste3.png',erosion)



## turn to csv file


csvpath = "./csvfles"
mkfolder(csvpath)
# if not os.path.exists(csvpath):
#   os.mkdir(csvpath)
#   print("Folder %s created!" % csvpath)
# else:
#   print("Folder %s already exists" % csvpath)
#   clear_folder(csvpath)

binfiles = os.listdir(binpath)



for image in binfiles
    bin_path = binpath *"/"* image
    bimg = Image.open(bin_path)
    # imageToMatrice = np.asarray(bimg)
    # println(size(imageToMatrice))
    imageMat = matimg.imread(bin_path)
    # println(typeof(imageMat))
    aaa = size(imageMat)
    if length(aaa)>2
        if(aaa[3] == 3)
    # reshape it from 3D matrice to 2D matrice
            imageMat_reshape = imageMat.reshape(imageMat.shape[0],
     									-1)
            print("Reshaping to 2D array:", 
                  imageMat_reshape.shape)
        else
            imageMat_reshape = imageMat
        end
    # if image is grayscale
    #####################################
    else
    # remain as it is
        imageMat_reshape = imageMat
    end
    # ###############################################	
    # # converting it to dataframe.
    # ###############################################
    mat_df = pd.DataFrame(imageMat_reshape)


    ### obs: mat_df.iat[4,3] to acess a value from a cell of a dataframe mat_df

    ################################################
    # exporting dataframe to CSV file.
    #############################################3##
    p = pathlib.Path(image)    
    # name2 = p.name.split(".")[1]
    name2= rsplit(image,".";limit=2)[1]
    # mat_df.to_csv(csvpath+'/'+name2 +'.csv',
    #  			header = None,
    #  			index = None)
   ##################################################  
   #  Storing (x,y) coordinate of black pixels
   ##############################################
    arr = np.asarray(mat_df)
    black_pixels = Tuple.(findall(iszero, arr))
    # println(size(arr))
    # println(typeof(black_pixels))
    # println(black_pixels[651])
    # println("ate aqui")
    # black_pixels = np.array(np.where(arr == 0))
    data = black_pixels
    # df = DataFrame(map(idx -> getindex.(data, idx), eachindex(first(data))), [:a, :b])
    # blackdotsdf = DataFrame(collect.(data), [:a, :b], copycols=false)
    blackdotsdf = DataFrame(NamedTuple{(:a, :b)}.(data))
    # black_pixel_coordinates = list(zip(black_pixels[0],black_pixels[1])) #dont working
    # blackdotsdf = pd.DataFrame(black_pixel_coordinates)
    # println("ate agora")
    # blackdotsdf.to_csv(csvpath*"/"*name2*".csv", )
    CSV.write(csvpath*"/"*name2*".csv", blackdotsdf)
end

println("created the dataframes and saved in csv files")

## obs to read a dataframe use 
# DataFrame(CSV.File(input))




#####################################
### samples (lesser points)
###################################

### samplesize = 7000 is too big
### samplesize around 2000 seens ok

# samplesize = 1000

# samplepath = "./samples"
# mkfolder(samplepath)

# csvfiles = os.listdir(csvpath)


# for csvfile in csvfiles
#     csv_path = csvpath *"/"* csvfile
#     mydata = DataFrame(CSV.File(csv_path))
#     # mydata = pd.read_csv(csv_path)
#     # coords = np.array(mydata)
#     coordsmat = Tuple.(eachrow(mydata))
#     list_coords = []
#     for point in coordsmat
#         # list_coords.append([point[0], point[1]])
#         push!(list_coords, point) #[point[1], point[2]])
#     end
#     # println(first(mydata,1))
#     if length(list_coords)<samplesize
#         println("file too small " * csvfile)
#     else
#         # println("ate aqui")
#         sample = random.sample(list_coords, samplesize)
#         # sample_df = pd.DataFrame(sample)
#         # sample_df.to_csv(samplepath*"/"*csvfile,header = None,index = None)#header {(:a, :b)}
#         sample_df = DataFrame(NamedTuple{(:a, :b)}.(sample))
#         CSV.write(samplepath*"/"*csvfile, sample_df)
#     end
# end



newsplepath = "./newsamples"
mkfolder(newsplepath)

csvpath = "./csvfles"
csvfiles = os.listdir(csvpath)

for csvfile in csvfiles
    csv_path = csvpath *"/"* csvfile
    local_df = DataFrame(CSV.File(csv_path))
    first = true 
    list_ok = Array[]
    for row in eachrow(local_df) 
        if first
            push!(list_ok, [row[1], row[2]])
            first = false
            continue
        end    
        point = [row[1], row[2]]
        if any((euclidean(point, point_ok) < DISTANCE_MIN for point_ok in list_ok))
            continue
        end    
        push!(list_ok,point)
    end
    sample_df = DataFrame(NamedTuple{(:a, :b)}.(list_ok))
    CSV.write(newsplepath*"/"*csvfile, sample_df)
    println(size(sample_df))
end



println("created and saved samples")


####################################################
#prototipe to create image from data
# #####################################################
# data = np.zeros((200, 300, 1), dtype=np.int64)
# data[0:200, 0:300] = [256]
# print(data[5,6])
# for dot in sample:
#     # print(dot[0],dot[1])
#     # print(data[dot[0],dot[1]])
#     data[dot[0],dot[1]] = 0
# cv2.imwrite(samplepath+'/'+'.png', data)


##  diagrams

ripspath = "./ripsdgm"
mkfolder(ripspath)
  
# samplefiles = os.listdir(samplepath)
samplefiles = os.listdir(newsplepath)
# rips = Rips()



# a = 0
idxpairs =[]
for sdata in samplefiles
    # datapath = samplepath*"/"*sdata
    datapath = newsplepath*"/"*sdata
    # data = pd.read_csv(datapath)
    data = DataFrame(CSV.File(datapath))
    # datas = np.array(data)
    datas = Tuple.(eachrow(data))
    datass = np.array(datas)
    # a = a+1
    # println(global a)
    # diagram = rips.fit_transform(data)
    diagram = ripserer(datas)[2]#;alg=:homology["dgms"][1] #create H1 diagram
    println(typeof(diagram))
    println(size(diagram,2))
    println(length(diagram))
    println(diagram[5])
    println(diagram[5][1])
    # diagram = ripser.ripser(datass)["dgms"][1] #create H1 diagram
    # idxpairs.append([idx, diagram])
    # plot_diagrams(diagram, show=True)
    # diagram_df = pd.DataFrame(diagram)
    # diagram_df.to_csv(ripspath*"/"*sdata)#, header = None,index = None)
    # diagram_df = DataFrame(NamedTuple{(:a, :b, :c)}.(diagram))
    diagram_df = DataFrame(diagram) #,:auto)
    CSV.write(ripspath*"/"*sdata, diagram_df)
end

println("fin")