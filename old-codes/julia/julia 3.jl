"""
################################################################
#### caution: slow, very time consuming
#### calculate dendograms
############################################################
"""


using PyCall

os = pyimport("os")
np = pyimport("numpy")
@pyimport pandas as pd
shutil = pyimport("shutil")
@pyimport persim

using Printf
using DataFrames
using CSV



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



##try to make the dendogram
##############################

##create distancematrix folder
################################

# ripspath = './ripsdgm'  #this is the file with the H1 diagrams
ripspath = "./red_ripsdgm"  #this is the file with the H1 diagrams
# reddenpath = './red_ripsdgm'
distancemtxpath = "./distancemtx"

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
#### listing diagrams
################################

idxnb = 0
idxlist = []
len_list = length(ripsfiles)
# for ripsdgm in ripsfiles
for idx in 1:len_list
    # dgmpath = ripspath+'/'+ripsdgm
    # dgm = pd.read_csv(dgmpath)
    # dgms = np.array(dgm)
    ripsdgm =ripsfiles[idx]
    push!(idxlist, [idx,ripsdgm])
    # println(idx)
    # idxnb = idxnb+1
end
ripdgm_df = pd.DataFrame(idxlist)
saveList(idxlist, distancemtxpath*"/"*"namelist.npy")
aaa = loadList(distancemtxpath*"/"*"namelist.npy")
# println(aaa)
# ripdgm_df.to_csv(distancemtx+'/'+list,header = None,index = None)






##############################################################
#### distances matrix
########################################################

mtx_size = length(idxlist)
# println(mtx_size," size of matrix" )
# println(typeof(idxlist))
matrix = np.zeros((mtx_size,mtx_size))
# print(matrix)
# matrix[1,1] = 3
# a = matrix[0,0]
# b = matrix[1,1]
# print(a,b)
# print(matrix)
# for n in range(size):
#     idx1, name1 = idxlist[n]
#     dgmpath1 = ripspath+'/'+name1
#     dgm1 = pd.read_csv(dgmpath1)
#     dgms1 = np.array(dgm1)
#     for m in range(size):
#         idx2, name2 = idxlist[m]
#         dgmpath2 = ripspath+'/'+name2
#         dgm2 = pd.read_csv(dgmpath2)
#         dgms2 = np.array(dgm2)
#         d, matching = persim.bottleneck(
#             dgm1,
#             dgm2,
#             matching=True
#         )
#         matrix[m,n] = d
        
# saveList(matrix, distancemtxpath+'/'+'distancematrix.npy')    


###################################################################
### reduced distance matrix
#################################################################

red_mat = []
for n in 1:mtx_size
    idx1, name1 = idxlist[n]
    # println("aqui")
    dgmpath1 = ripspath*"/"*name1
    dgm1 = pd.read_csv(dgmpath1)
    dgms1 = np.array(dgm1)
    for m in n+1:mtx_size
        idx2, name2 = idxlist[m]
        dgmpath2 = ripspath*"/"*name2
        dgm2 = pd.read_csv(dgmpath2)
        dgms2 = np.array(dgm2)
        d, matching = persim.bottleneck(dgm1, dgm2, matching=true)
        println(d)
        push!(red_mat, d)
        # red_mat.append(d)
    end
end

reduced_array = np.array(red_mat)
saveList(red_mat, distancemtxpath*"/"*"red_dist_mat.npy")  
println(red_mat, "reduced matrix")



#####################################################
#### dataframe dos diagramas com os nomes como indices 
#### nao completei ainda

dataf = []
namelist = []
for ripsdgm in ripsfiles
    dgmpath = ripspath*"/"*ripsdgm
    # println("aqui")
    dgm = pd.read_csv(dgmpath)
    dgms = np.array(dgm)
    push!(dataf, [dgms])
    # dataf.append([dgms])
    push!(namelist, ripsdgm)
    # namelist.append(ripsdgm)
end

df = pd.DataFrame(dataf, index= namelist)

dataframe = DataFrame(matrix,:auto)

# dataframe.to_csv(distancemtxpath+'/'+'distmatrix_df.csv',header = None, #  index = None)

CSV.write(distancemtxpath*"/"*"distmatrix_df.csv", dataframe)                 
## obs:nao salva os indices quando salva o dataframe em csv

    




println("fin")
