"""julia 2.1"""

using PyCall

os = pyimport("os")
np = pyimport("numpy")
pd = pyimport("pandas")
shutil = pyimport("shutil")
using CSV
using DataFrames
using Printf
using SciPy


##################################################################
##################################################################
##   reduce the diagram, removing short intervals
##  this reduces computations
##################################################################
##################################################################


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



ripspath = "./junripsdgm"
reddenpath = "./junred_ripsdgm"

mkfolder(reddenpath)

dendfiles = os.listdir(ripspath)

#########################################
# 1 - read the diagram
# 2 - detect short intervals
# 3 - remove said intervals
###########################################


## cut parameter
cut_parm = 2


##########################
function thin_den(dendogram, cut_parm)
    del_list = []
    nn = size(dendogram)[1]
    for n in 1:1:size(dendogram)[1]
        comp = dendogram[n,2]-dendogram[n,1]
        if comp < cut_parm
            push!(del_list, n)
        end
    end
    # red_dendogram = dendogram
    red_dendogram = deleterows!(dendogram, del_list) 
    # # deleteat!for recent julia version as deleterows is deprecated
    println(nn,"/", size(red_dendogram)[1])      
    return red_dendogram
end

########################################


for dend in dendfiles
    data1 = DataFrame(CSV.File(ripspath*"/"*dend))
    # dendogram = Tuple.(eachrow(data1))
    dendogram = np.array(data1)
    red_dendogram = thin_den(data1,cut_parm)
    # red_dendogram = thin_den(dendogram,cut_parm)
    println(size(red_dendogram))
    println(typeof(red_dendogram))
    # diagram_df = DataFrame(red_dendogram, :auto)
    diagram_df = DataFrame(red_dendogram)
    CSV.write(reddenpath*"/"*dend, diagram_df)
end

println("fin")

