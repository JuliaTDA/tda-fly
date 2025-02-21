""" julia 3.1"""


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
import SciPy



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


ripspath = "./junred_ripsdgm"  #this is the file with the H1 diagrams

distancemtxpath = "./jundistancemtx"

mkfolder(distancemtxpath)

ripsfiles = os.listdir(ripspath)


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


###########################################################
#### indexed listing of diagrams
##########################################################

idxnb = 0
idxlist = []
len_list = length(ripsfiles)
# for ripsdgm in ripsfiles
for idx in 1:len_list
    ripsdgm =ripsfiles[idx]
    push!(idxlist, [idx,ripsdgm])
end
ripdgm_df = pd.DataFrame(idxlist)
saveList(idxlist, distancemtxpath*"/"*"namelist.npy")

mtx_size = length(idxlist)

###################################################################
### 1-D condensed distance matrix
#################################################################

red_mat = []
for n in 1:mtx_size
    idx1, name1 = idxlist[n]
    # println("aqui")
    dgmpath1 = ripspath*"/"*name1
    dgm1 = pd.read_csv(dgmpath1)
    # dgm1 = DataFrame(CSV.File(ripspath*"/"*name1))
    dgms1 = np.array(dgm1)
    for m in n+1:mtx_size
        idx2, name2 = idxlist[m]
        dgmpath2 = ripspath*"/"*name2
        dgm2 = pd.read_csv(dgmpath2)
        dgms2 = np.array(dgm2)
        d, matching = persim.bottleneck(dgm1, dgm2, matching=true)
        # println(d)
        push!(red_mat, d)
        # red_mat.append(d)
    end
end