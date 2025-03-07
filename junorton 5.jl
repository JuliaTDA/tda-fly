""" julia 4 """


using PyCall
os = pyimport("os")
np = pyimport("numpy")
pd = pyimport("pandas")
shutil = pyimport("shutil")
@pyimport persim
# scipy = pyimport("scipy")
# @pyimport scipy.cluster.hierarchy.linkage as linkage
@pyimport matplotlib.pyplot as plt

using SciPy

distancemtxpath = "./jundistancemtx"

ripspath = "./junripsdgm"  #this is the file with the H1 diagrams


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

############################################################
############################################################

thelist = loadList("./jundistancemtx/namelist.npy")
# print(thelist)
###the list of names



# datapath = distancemtxpath*"/"*"distmatrix_df.csv"
# distmat =  pd.read_csv(datapath)

red_dm =  loadList(distancemtxpath*"/"*"cond_dist_mat.npy")
# red_dm =  loadList(distancemtxpath*"/"*"cond_dist_mat1.npy")
# red_dm =  loadList(distancemtxpath*"/"*"cond_dist_mat2.npy")
# red_dm =  loadList(distancemtxpath*"/"*"cond_dist_mat3.npy")
# red_dm =  loadList(distancemtxpath*"/"*"cond_dist_mat4.npy")
y=red_dm
Z =SciPy.cluster.hierarchy.linkage(y, method="single", metric="euclidean", optimal_ordering=false)

fig = plt.figure(figsize=(25, 10))

dn = SciPy.cluster.hierarchy.dendrogram(Z)
plt.show()





println("fin")