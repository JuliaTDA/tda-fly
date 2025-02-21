""" split images in quarters """


using Printf
using Images
using PyCall
using CSV
using JLD2
os = pyimport("os")
cv2 = pyimport("cv2")
np = pyimport("numpy")

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


listpath = "./junlist"
mkfolder(listpath)
imgpath = "./binaryimages/" 
imgfiles = os.listdir(imgpath)
save_object(listpath*"/list.jld2", imgfiles) 
mkfolder("./junimg")



for img in imgfiles
    img_path = imgpath*"/" *img
    image = cv2.imread(img_path,0) 
    w, h = size(image)
    # println(typeof(w))
    x = Int64(w/2)
    y= Int64(h/2)
    println(x,y)
    ims = [image[1:x, 1:y], image[1:x, y+1:h], image[x+1:w, 1:y], 
        image[x+1:w, y+1:h] ]
    # ims = [@view img[iw:iw + 100, ih:ih + 100] 
    #     for iw in 1:100:w - 100, 
    #     ih in 1:100:h - 100]
    for (i, im) in enumerate(ims)
        save("./junimg/"*img*"-$(lpad(i, 3, "0")).png", im)
    end
end

println("here")
lista = load_object(listpath*"/list.jld2") # load the file
# println(lista)