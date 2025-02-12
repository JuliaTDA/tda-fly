# module Tipulidae


# Diagramas Persistencia Direcoes


### JuliaVersion ###
# JuliaVersion
### Requirements ###

using CSV
using DataFrames
using DiscretePersistentHomologyTransform
using Eirene
using Hungarian
using LinearAlgebra
# using Plots
using Printf
using SparseArrays

using PyCall
os = pyimport("os")


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



# directions = 4
# angles = [n*pi/(directions/2) for n in 1:directions]
# directions = [[cos(x), sin(x)] for x in angles]

phtpath = "./pht_df"
mkfolder(phtpath)
samplepath  = "./newsamples"
samplefiles = os.listdir(samplepath)


########################
## number of directions
########################

directions = 8

for sample in samplefiles
    datapath = samplepath*"/"*sample
    println("here")
    Boundary = CSV.read(datapath)
    println("and here")
    P_D = PHT(Boundary, directions)
    # println(P_D)
    for n in 1:directions
        P_Dn = P_D[n]
        PDn = DataFrame(P_Dn)
        nindex = string(n)
        CSV.write(phtpath*"/"*nindex*sample, PDn)
    end
    # println(typeof(P_D1))
    # println(typeof(PD_1))
    # CSV.write(phtpath*"/"*"1"*sample, P_D1)
    # P_D2 = P_D[2]
    # CSV.write(phtpath*"/"*"2"*sample, P_D2)
    # P_D3 = P_D[3]
    # CSV.write(phtpath*"/"*"3"*sample, P_D3)
    # P_D4 = P_D[4]
    # CSV.write(phtpath*"/"*"4"*sample, P_D4)
end