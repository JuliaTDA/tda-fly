""" Distance Matrix """

using PyCall

os = pyimport("os")
np = pyimport("numpy")
@pyimport pandas as pd
shutil = pyimport("shutil")
@pyimport persim
R = pyimport("scipy.spatial.distance")

using Printf
using DataFrames
using CSV
using SciPy
using JLD2
import SciPy

##################################################################################### 
###                     #############################################################
DIRECTIONS = 8          #############################################################
###                     #############################################################
##################################################################################### 

###############################
## auxiliary map (saves lists)
function saveList(myList,filename)
    # the filename should mention the extension 'npy'
    np.save(filename,myList)
    println("Saved successfully!")
end

function loadList(filename)
    # the filename should mention the extension 'npy'
    tempNumpyArray=np.load(filename)
    return tempNumpyArray.tolist()
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

#########################################################################
#########################################################################

###############################
##create distancematrix folder
###############################
new_distancemtxpath = "./new_distancemtx"

mkfolder(new_distancemtxpath)






##############################################
### to create the condensed distance matrix
##############################################
ripspath = "./pht_df"  #this is the file with the H1 diagrams
# ripsfiles = os.listdir(ripspath)
listpath = "./dflist"
list_df = load_object(listpath*"/list.jld2")
msize = length(list_df)

#############################################
## first create a indexed list of diagrams

idxlist = []
len_list = msize
for idx in 1:len_list
    ripsdgm =list_df[idx]
    push!(idxlist, [idx,ripsdgm])
end
# rip_list = pd.DataFrame(idxlist)
saveList(idxlist, new_distancemtxpath*"/"*"namelist.npy")


##################################################################################

matrix = np.zeros((msize,msize)) ###### not sure if needed

###################################################################
### 1-D condensed distance matrix
#################################################################
cond_mat = []
# for i in 1:DIRECTIONS
#     string_i = string(i)
#     for name in list_df
#         name_i = string_i*name
#     end
# end
for n in 1:msize
    idx_x, name_x = idxlist[n]
    println(n, name_x)
    # println("heres")
    # dgmpath1 = ripspath*"/"*name1
    # dgm1 = pd.read_csv(dgmpath1)
    # dgms1 = np.array(dgm1)
    for m in n+1:msize
        idx_y, name_y = idxlist[m]
        dxy = 0
        for i in 1:DIRECTIONS
            st_i = string(i)
            dgm_x = pd.read_csv(ripspath*"/"*st_i*name_x)
            dgmsx = np.array(dgm_x)
            dgm_y = pd.read_csv(ripspath*"/"*st_i*name_y)
            dgmsy = np.array(dgm_y)
            d, matching = persim.bottleneck(dgmsx, dgmsy, matching=true)
            dxy = dxy+d
        end
        # dgmpath2 = ripspath*"/"*name2
        # dgm2 = pd.read_csv(dgmpath2)
        # dgms2 = np.array(dgm2)
        # d, matching = persim.bottleneck(dgm1, dgm2, matching=true)
        # println(d)
        push!(cond_mat, dxy)
    end
end

reduced_array = np.array(cond_mat)
saveList(cond_mat, new_distancemtxpath*"/red_dist_mat.npy")  
println(typeof(reduced_array))
