""" julia 3.1"""


using PyCall

os = pyimport("os")
np = pyimport("numpy")
pd = pyimport("pandas")
# @pyimport pandas as pd
shutil = pyimport("shutil")
@pyimport persim
R = pyimport("scipy.spatial.distance")
using JLD2
using Printf
using DataFrames
using CSV
using SciPy
import SciPy

listpath = "./junlist"
lista = load_object(listpath*"/list.jld2") 

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

# mtx_size = length(idxlist)

###################################################################
### 1-D condensed distance matrix
#################################################################
mtx_size = length(lista) # list of figures
foo = length(idxlist) # list of cut figures
faa = Int64(foo*(foo+1)/2)

cond_mat = []
for n in 1:mtx_size
    name1 = lista[n]
    dgmp_1 = ripspath*"/"*name1*"-1.csv"
    dgmp_2 = ripspath*"/"*name1*"-2.csv"
    dgmp_3 = ripspath*"/"*name1*"-3.csv"
    dgmp_4 = ripspath*"/"*name1*"-4.csv"
    dgm1 = pd.read_csv(dgmp_1)
    dgm2 = pd.read_csv(dgmp_2)
    dgm3 = pd.read_csv(dgmp_3)
    dgm4 = pd.read_csv(dgmp_4)
    dgma1 = np.array(dgm1)
    dgma2 = np.array(dgm2)
    dgma3 = np.array(dgm3)
    dgma4 = np.array(dgm4)
    for m in n+1:mtx_size
        name2 = lista[m]
        dgmp_1 = ripspath*"/"*name2*"-1.csv"
        dgmp_2 = ripspath*"/"*name2*"-2.csv"
        dgmp_3 = ripspath*"/"*name2*"-3.csv"
        dgmp_4 = ripspath*"/"*name2*"-4.csv"
        dgm1 = pd.read_csv(dgmp_1)
        dgm2 = pd.read_csv(dgmp_2)
        dgm3 = pd.read_csv(dgmp_3)
        dgm4 = pd.read_csv(dgmp_4)
        dgmb1 = np.array(dgm1)
        dgmb2 = np.array(dgm2)
        dgmb3 = np.array(dgm3)
        dgmb4 = np.array(dgm4)
        d_1, matching = persim.bottleneck(dgma1, dgmb1, matching=true)
        d_2, matching = persim.bottleneck(dgma2, dgmb2, matching=true)
        d_3, matching = persim.bottleneck(dgma3, dgmb3, matching=true)
        d_4, matching = persim.bottleneck(dgma4, dgmb4, matching=true)
        d = d_1 + d_2 + d_3 +d_4
        # println(d)
        push!(cond_mat, d)
        # red_mat.append(d)
    end
    # for i in 1:4
    #     dgmp = ripspath*"/"*name1*"-$(i).csv"
    #     dgm = pd.read_csv(dgmp)
    #     @eval $(Symbol("dgma", i)) = np.array(dgm)
    # end
end
condensed_array = np.array(cond_mat)
saveList(cond_mat, distancemtxpath*"/"*"cond_dist_mat.npy")  


# for i in 1:4
#     iname ="mat"*"$(i)x"
#     red_mat = []
#     for n in 1:mtx_size
#         name1 = lista[n]
#         # println(name1)
#         dgmpath1 = ripspath*"/"*name1*"-$(i).csv"
#         # println(dgmpath1)
#         dgm1 = pd.read_csv(dgmpath1)
#         dgms1 = np.array(dgm1)
#         for m in n+1:mtx_size
#             name2 = lista[m]
#             dgmpath2 = ripspath*"/"*name2*"-$(i).csv"
#             dgm2 = pd.read_csv(dgmpath2)
#             dgms2 = np.array(dgm2)
#             d, matching = persim.bottleneck(dgm1, dgm2, matching=true)
#             # println(d)
#             push!(red_mat, d)
#             # red_mat.append(d)
#         end
#     end
#     # reduced_array = np.array(red_mat)
#     saveList(red_mat, distancemtxpath*"/"*iname*".npy")
#     # if i==1
#     #     println(i)
#     #     nred_mat = red_mat
#     # else 
#     #     println(i)
#     #     nred_mat = nred_mat + red_mat
#     # end
# end


# saveList(nred_mat, distancemtxpath*"/"*"nred_dist_mat.npy") 

println("end")