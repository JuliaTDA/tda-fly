# module Tipulidae


# Diagramas Persistencia Direcoes


### JuliaVersion ###
# JuliaVersion
### Requirements ###

using CSV
using DataFrames
using JLD2
using LinearAlgebra
using Printf
using PyCall
using Ripserer
using SparseArrays
os = pyimport("os")
np = pyimport("numpy")



DIRECTIONS = 2
## cut parameter
cut_parm = 10

#######################################
epsilon = 15 #epsilon-neighborhood parameter we can choose
########################################


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


#########################################
# 1 - read the diagram
# 2 - detect short intervals
# 3 - remove said intervals
###########################################




##########################
function thin_den(dendogram, cut_parm)
    del_list = []
    nn = size(dendogram)[1]
    # println(nn)
    for n in 1:1:size(dendogram)[1]
        comp = dendogram[n,2]-dendogram[n,1]
        if comp < cut_parm
            push!(del_list, n)
        end
    end
    red_dendogram = deleterows!(dendogram, del_list) 
    # # deleteat!for recent julia version as deleterows is deprecated
    println(nn,"/", size(red_dendogram)[1])      
    return red_dendogram
end

########################################



# angles = [n*pi/(directions/2) for n in 1:directions]
# directions = [[cos(x), sin(x)] for x in angles]

phtpath = "./pht_df"
mkfolder(phtpath)
listpath = "./dflist"
mkfolder(listpath)
samplepath  = "./newsamples"
samplefiles = os.listdir(samplepath)
save_object(listpath*"/list.jld2", samplefiles)  #save the file
#load_object("list.jld2") # load the file

########################
## number of directions
########################



##############################
### our "implementation" of PHT
###
# """ extracting the expected complex for our case """
# """ we need the epsilon-neighborhood complex, but couldnt find in eirene format"""


### obs: same directions are used multiple times, so calculate only once.
function Directions(directions)
    if typeof(directions) ==  Int64
		println("auto generating directions")
		dirs = Array{Float64}(undef, directions,2)
		for n in 1:directions
			dirs[n,1] = cos(n*pi/(directions/2))
			dirs[n,2] = sin(n*pi/(directions/2))
		end
		println("Directions are:", dirs)
	else
		println("using directions provided")
		dirs = copy(directions)
	end
    return dirs
end

directions = Directions(DIRECTIONS)





###### to do:
# number the cells 1, ..., N in ascending order, according to dimension
# let D be the N x N zero/one matrix with D[i,j] = 1 iff i is a codimension-1 face of cell j
# define vectors rv and cp by the following procedure (see Julia documentation for a more direct interpretation)

# function Distance_Matrix(pointset)
#     number_of_points = size(pointset[:,1])[1] #number of points
#     point = Matrix(ordered_points)
#     distmtx = zeros((number_of_points, number_of_points))
#     for i in 1:(number_of_points-1)
#         for j in (i+1):number_of_points
#             distmtx[i,j] = norm(point[i,:]-point[j,:])
#         end
#     end
#     return distmtx
# end

function Direction_Filtration(ordered_points, direction, distmtx; out = "barcode", one_cycle = false )
	number_of_points = size(ordered_points[:,1])[1] #number of points
    # println("here", number_of_points)
	heights = zeros(number_of_points) #empty array to be changed to heights for filtration
	# fv = zeros(2*number_of_points) #blank fv Eirene
    fv = zeros(number_of_points) ## size of fv will change later
	for i in 1:number_of_points
		heights[i]= ordered_points[i,1]*direction[1] + ordered_points[i,2]*direction[2] #calculate heights in specificed direction
	end
	
	for i in 1:number_of_points
		fv[i]= heights[i] # for a point the filtration step is the height (0-cells)
	end
    ###################################################
	#from here, we start working changes: defining 1-cells and 2-cells
    ####################################################
    # point = Matrix(ordered_points) # matrix from dataframe
    ####################################################

    FiltList =[]
    ###################
    ## 0-cells
    for i in 1:number_of_points
        vertice = [(i,) => heights[i]]
        append!(FiltList,vertice)
    end
    ###################
    ## 1-cells
    global odc
    odc = 0 #number of one dimensional cells
    # listpairs = [] # start list of the 1-cells
    for i in 1:(number_of_points-1)
        # println(point[i,:])
        for j in (i+1):number_of_points
            # println((point[i],point[j]))
            # println(point)
            # println(point[i,:])
            if distmtx[i,j]<epsilon
                # odc = (odc+1) # work in python, not so sure in julia
				global odc
				odc += 1
                # append!(listpairs, [(i,j)])
                h = maximum([heights[i], heights[j]])
                aresta = [(i,j) => h]
                append!(FiltList,aresta)
            end
        end
    end # length of fv = number_of_points + odc.
    println("filtlist ", size(FiltList))
    # println(FiltList[1500])
    ######################
    ## 2-cells 
    println("2 cells")
    global tdc
    tdc = 0 #number of two dimensional cells
    # listrio = [] # start list of the 1-cells
    for i in 1:(number_of_points-2)
        for j in (i+1):(number_of_points-1)
            for k in (j+1): number_of_points
                diam = maximum([distmtx[i,j], distmtx[i,k], distmtx[k,j]])
                # println(diam)
                if diam < epsilon 
					# diameter(point[i],point[j],point[k])<epsilon
                    # tdc = (tdc+1) # work in python, not so sure in julia
                    global tdc
					tdc += 1
					# append!(listrio, [(i,j,k)])
                	h = maximum([heights[i], heights[j], heights[k]])
                    face = [(i,j,k) => h]
                    append!(FiltList, face)
                	# append!(fv,h)
                end
            end
        end
    end # length of fv = number_of_points + odc + tdc.
    # thresh = maximum(heights)+1
    flt = Custom(FiltList;threshold = 800)
    C = ripserer(flt)[2]
    
    println("ok")
    return C
end


#### Wrapper for the PHT function ####

function PHT(curve_points, directions; one_cycle = false) #accepts an ARRAY of points

    pht = []
	
	if one_cycle == true
		c_1 = []
	end
	dirs = copy(directions)
    M = size(dirs,1)
	for i in 1:M
        Points = Matrix(curve_points)
        N = size(Points,1)
        Pointset = []
        for i in 1:N
            append!(Pointset,Points[i,:])
        end
        println("aqui?")
		dismat = DistanceMatrix(Pointset)
        direc = dirs[i,:]
		if one_cycle == true
			pd,c_1_1 = Direction_Filtration(curve_points, direc, dismat, one_cycle = true)
			append!(c_1, c_1_1)
			pht = vcat(pht, [pd])
		else
			pd = Direction_Filtration(curve_points, direc, dismat)
			pht = vcat(pht, [pd])
		end
	end
    # println(pht)
    return pht
end




############################################
### CALCULATIONS
###




for sample in samplefiles
    datapath = samplepath*"/"*sample
    # println("here")
    println(datapath)
    Boundary = CSV.read(datapath)
    println(size(Boundary))
    # println("and here")
    P_D = PHT(Boundary, directions)
    # println(P_D[1])
    # println(typeof(P_D[1]))
    for n in 1:DIRECTIONS
        P_Dn = P_D[n]
        println("ook")
        PDn = DataFrame(P_Dn)
        # println(PDn)
        println("okk")
        replace!(PDn.death, Inf => 800) ##### infinite deathtimes cause problems
        PDnn = thin_den(PDn, cut_parm)
        nindex = string(n)
        CSV.write(phtpath*"/"*nindex*sample, PDnn)
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