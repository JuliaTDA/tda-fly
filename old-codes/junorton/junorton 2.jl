""" julia 1"""

using Printf
using Images

using CSV
using JLD2

using PyCall
os = pyimport("os")
cv2 = pyimport("cv2")
np = pyimport("numpy")

shutil = pyimport("shutil")
@pyimport PIL.Image as Image 
@pyimport PIL.ImageOps as ImageOps 
@pyimport sys
@pyimport pathlib
@pyimport matplotlib.image as matimg
pd = pyimport("pandas")
@pyimport random
@pyimport persim
@pyimport scipy

using NumPyArrays
using MatrixNetworks
import Glob as glob
using Ripserer
using DataFrames
using Distances

DISTANCE_MIN = 5

###############################
## auxiliary map (saves lists)
function  saveList(myList,filename)
    # the filename should mention the extension 'npy'
    np.save(filename,myList)
    print("Saved successfully!")
end
    
function loadList(filename)
    # the filename should mention the extension 'npy'
    tempNumpyArray=np.load(filename)
    println(typeof(tempNumpyArray))
    return tempNumpyArray #.tolist()
end


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

function mkfolder(newpath)
    if os.path.exists(newpath)
        println("Folder $newpath already exists")
        clear_folder(newpath)
    else
        os.mkdir(newpath)# make directory
        println("Folder $newpath created!")
    end
end

listpath = "./junlist"
lista = load_object(listpath*"/list.jld2") 

pathname = "./junimg"

if !os.path.exists(pathname)
    @printf("Folder %s does not exist", "./re_im")
    return
end


ncsvpath = "./juncsvfles"
mkfolder(ncsvpath)

binfiles = os.listdir(pathname)



for image in binfiles
    bin_path = pathname *"/"* image
    bimg = Image.open(bin_path)
    imageMat = matimg.imread(bin_path)
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
   ##################################################  
   #  Storing (x,y) coordinate of black pixels
   ##############################################
    arr = np.asarray(mat_df)
    black_pixels = Tuple.(findall(iszero, arr))
    data = black_pixels
    blackdotsdf = DataFrame(NamedTuple{(:a, :b)}.(data))
    CSV.write(ncsvpath*"/"*name2*".csv", blackdotsdf)
end

println("created the dataframes and saved in csv files")



newsplepath = "./junsamples"
mkfolder(newsplepath)

csvpath = "./juncsvfles"
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


##  diagrams

ripspath = "./junripsdgm"
mkfolder(ripspath)
  
samplefiles = os.listdir(newsplepath)


idxpairs =[]
for sdata in samplefiles
    datapath = newsplepath*"/"*sdata
    data = DataFrame(CSV.File(datapath))
    datas = Tuple.(eachrow(data))
    datass = np.array(datas)
    diagram = ripserer(datas)[2]#;alg=:homology["dgms"][1] #create H1 diagram
    # println(typeof(diagram))
    # println(size(diagram,2))
    # println(length(diagram))
    # println(diagram[5])
    # println(diagram[5][1])
    diagram_df = DataFrame(diagram) #,:auto)
    CSV.write(ripspath*"/"*sdata, diagram_df)
end

println("fin")
