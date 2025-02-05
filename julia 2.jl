"""remove short intervals in diagrams"""

using PyCall

os = pyimport("os")
np = pyimport("numpy")
@pyimport pandas as pd
shutil = pyimport("shutil")
using CSV
using DataFrames
using Printf
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

ripspath = "./ripsdgm"
reddenpath = "./red_ripsdgm"

mkfolder(reddenpath)

dendfiles = os.listdir(ripspath)

#########################################
# 1 - read the diagram
# 2 - detect short intervals
# 3 - remove said intervals
###########################################


## cut parameter
cut_parm = 5


##########################
function thin_den(dendogram, cut_parm)
    rip_list = []
    # println(dendogram)
    for n in 1:size(dendogram)[1] #range(1,size(dendogram)[1])
        # print(n)
        # println(size(dendogram))
        # println(typeof(dendogram))
        #  println(dendogram[n])
        # println(dendogram[n][1])
        # println("here")
        comp = dendogram[n,2]-dendogram[n,1]
        # print(n, comp)
        ## comp >5 seens to be the ideal
        if comp> cut_parm         ##### parametro ajustavel aqui ######
            push!(rip_list, (dendogram[n,1], dendogram[n,2]))
            # push!(rip_list, dendogram[n, _])
        end
    end
    # println(typeof(rip_list))
    red_dendogram = np.array(rip_list)    
    # println(typeof(red_dendogram))
    # println(size(red_dendogram))
    # println(red_dendogram[1])  
    println(size(dendogram)[1],"/", size(red_dendogram)[1])      
    return red_dendogram
end

########################################



for dend in dendfiles
    # data1 = pd.read_csv(ripspath*"/"*dend)
    # temp_1 = CSV.File(ripspath*"/"*dend)
    # println("aqui")
    # println(typeof(temp_1))
    data1 = DataFrame(CSV.File(ripspath*"/"*dend))
    # dendogram = Tuple.(eachrow(data1))
    dendogram = np.array(data1)
    # println(typeof(data1))
    # println(size(data1))
    # println("aqui")
    # println(length(data1))
    # println(data1[5, 2])
    # # println(data1[1][2])
    # println(size(dendogram))
    # println(typeof(dendogram))
    # println(length(data1))
    # println(dendogram[1, 1])
    # println(dendogram[n][1])
    red_dendogram = thin_den(data1,cut_parm)
    println(size(red_dendogram))
    diagram_df = pd.DataFrame(red_dendogram)
    # println("aqui")
    # diagram_df.to_csv(reddenpath*"/"*dend,header = None,index = None)
    # diagram_df = DataFrame(NamedTuple{(:a, :b)}.(red_dendogram))
    println(typeof(red_dendogram))
    diagram_df = DataFrame(red_dendogram, :auto)
    # diagram_df = DataFrame(red_dendogram)
    # println("e aqui")
    CSV.write(reddenpath*"/"*dend, diagram_df)
end
    


