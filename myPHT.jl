""" extracting the expected complex for our case """
""" we need the epsilon-neighborhood complex, but couldnt find in eirene format"""

#######################################
epsilon = 6 #epsilon-neighborhood parameter we can choose
########################################

###### to do:
# number the cells 1, ..., N in ascending order, according to dimension
# let D be the N x N zero/one matrix with D[i,j] = 1 iff i is a codimension-1 face of cell j
# define vectors rv and cp by the following procedure (see Julia documentation for a more direct interpretation)

# julia> S  = sparse(D)
# julia> rv = S.rowval
# julia> cp = S.colptr


# cells are ordered by dimension. so 0-cells first, then 1-cells and then 2-cells.
# dv[i] is the dimension of cell i (dimension vector)
# fv[i] is the birthtime of cell i  (first-appearence vector)

# ev[k] is the total number of cells with dimension k-1
# dp[k] is 1 plus the number of cells of dimension strictly less than k-1

"""aplicação final:
# julia> C = eirene(rv=rv,cp=cp,dv=dv,fv=fv)
#  you can substitute either ev = ev or dp=dp in place of dv=dv. Eirene only needs one of the three
"""

function Direction_Filtration(ordered_points, direction; out = "barcode", one_cycle = false )
	number_of_points = length(ordered_points[:,1]) #number of points
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

    ###################
    ## 1-cells
    odc = 0 #number of one dimensional cells
    listpairs = [] # start list of the 1-cells
    for i in 1:(number_of_points-1)
        for j in (i+1):number_of_points
            if norm(point[i],point[j])<epsilon
                # odc = (odc+1) # work in python, not so sure in julia
				global odc
				odc += 1
                append!(listpairs, [(i,j)])
                h = maximum([heights[i], heights[j]])
                append!(fv,h)
            end
        end
    end # length of fv = number_of_points + odc.

    ######################
    ## 2-cells 
    tdc = 0 #number of two dimensional cells
    listrio = [] # start list of the 1-cells
    for i in 1:(number_of_points-2)
        for j in (i+1):(number_of_points-1)
            for k in (j+1): number_of_points
                if norm(point[i],point[j])<epsilon 
					if norm(point[k],point[j])<epsilon
						if norm(point[i],point[k])<epsilon
					# diameter(point[i],point[j],point[k])<epsilon
                    # tdc = (tdc+1) # work in python, not so sure in julia
                    	global tdc
						tdc += 1
						append!(listrio, [(i,j,k)])
                    	h = maximum([heights[i], heights[j], heights[k]])
                    	append!(fv,h)
						end
					end
                end
            end
        end
    end # length of fv = number_of_points + odc + tdc.
	total_size = number_of_points + odc + tdc

	dv =[]  # dv for Eirene
	for i in 1:number_of_points
		append!(dv,0) # every point is 0 dimensional
	end
	for i in 1:odc
		append!(dv,1) # edges are 1 dimensional
	end
	for i in 1:tdc
		append!(dv,2) # faces are 2 dimensional
	end

	D = zeros((total_size, total_size))

	# edges in the matrix
	for i in 1:odc
		D[listpairs[i][1],(i+number_of_points)]=1 # create boundary matrix and put in entries
		D[listpairs[i][2],(i+number_of_points)]=1
	end

	# faces in the matrix
	for i in 1:tdc
		D[listrio[i][1],(i+odc+number_of_points)]=1 # put in entries for boundary matrix
		D[listrio[i][2],(i+odc+number_of_points)]=1
		D[listrio[i][3],(i+odc+number_of_points)]=1
	end

	ev = [number_of_points, odc, tdc] # template ev for Eirene
	S  = sparse(D) 	# converting as required for Eirene
	rv = S.rowval 	# converting as required for Eirene
	cp = S.colptr	# converting as required for Eirene
	C = Eirene.eirene(rv=rv,cp=cp,ev=ev,fv=fv) # put it all into Eirene
	
	if out == "barcode"
		if one_cycle == true
			return barcode(C, dim=0), maximum(heights)
		else
			return barcode(C, dim=0)
		end
	else
		if one_cycle == true
			return C, maximum(heights)
		else
			return C
		end
	end
end #Direction_Filtration

#     #####################################################
# 	#################(original, will be erased later)####
# 	#####################################################
# 	for i in 1:(number_of_points-1)
# 		fv[(i+number_of_points)]=maximum([heights[i], heights[i+1]]) # for an edge between two adjacent points it enters when the 2nd of the two points does
# 	end
	
# 	fv[2*number_of_points] = maximum([heights[1] , heights[number_of_points]]) #last one is a special snowflake
# 	dv = [] # template dv for Eirene
	
# 	for i in 1:number_of_points
# 		append!(dv,0) # every point is 0 dimensional
# 	end
	
# 	for i in (1+number_of_points):(2*number_of_points)
# 		append!(dv,1) # edges are 1 dimensional
# 	end
	
# 	D = zeros((2*number_of_points, 2*number_of_points))
	
# 	for i in 1:number_of_points
# 		D[i,(i+number_of_points)]=1 # create boundary matrix and put in entries
# 	end
	
# 	for i in 2:(number_of_points)
# 		D[i, (i+number_of_points-1)]=1 # put in entries for boundary matrix
# 	end
	
# 	D[1, (2*number_of_points)]=1
	
# 	ev = [number_of_points, number_of_points] # template ev for Eirene
	
# 	S  = sparse(D) 	# converting as required for Eirene
# 	rv = S.rowval 	# converting as required for Eirene
# 	cp = S.colptr	# converting as required for Eirene
# 	C = Eirene.eirene(rv=rv,cp=cp,ev=ev,fv=fv) # put it all into Eirene
	
# 	if out == "barcode"
# 		if one_cycle == true
# 			return barcode(C, dim=0), maximum(heights)
# 		else
# 			return barcode(C, dim=0)
# 		end
# 	else
# 		if one_cycle == true
# 			return C, maximum(heights)
# 		else
# 			return C
# 		end
# 	end
# end #Direction_Filtration


#### Wrapper for the PHT function ####

function PHT(curve_points, directions; one_cycle = false) #accepts an ARRAY of points
	
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
	pht = []
	
	if one_cycle == true
		c_1 = []
	end
	
	for i in 1:size(dirs,1)
		
		if one_cycle == true
			pd,c_1_1 = Direction_Filtration(curve_points, dirs[i,:], one_cycle = true)
			append!(c_1, c_1_1)
			pht = vcat(pht, [pd])
		else
			pd = Direction_Filtration(curve_points, dirs[i,:])
			pht = vcat(pht, [pd])
		end
	end
	
	if one_cycle == true
		return pht, c_1
		
	else
		return pht
	end
end