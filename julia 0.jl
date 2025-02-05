""" cropping images """


using PyCall

@pyimport PIL.Image as Image 
@pyimport PIL.ImageOps as ImageOps
# shutil = pyimport("shutil")
os = pyimport("os")
np = pyimport("numpy")
cv2 = pyimport("cv2")
using MatrixNetworks
using NumPyArrays
using Printf

####################################
## empty the indicated folder
function clear_folder(folder)
# folder = '/path/to/folder'
    for filename in os.listdir(folder)
        file_path = os.path.join(folder, filename)
        try
            rm(file_path, force=true, recursive=true)
        catch e #Exception as e
            printf("Failed to delete %s. Reason: %s", file_path, e)
        end
    end
end


# from thinning import clear_folder
#######################################################################
### creates a new folder
### newpath = './newfoldername'
### if it already exists, empties the folder
function mkfolder(newpath)
# check whether directory already exists BEFORE trying to create file
    # if not os.path.exists(newpath)
    #     os.mkdir(newpath)
    #     print("Folder %s created!" % newpath)
    # else
    #     print("Folder %s already exists" % newpath)
    #     clear_folder(newpath)
    # end
    if os.path.exists(newpath)
        println("Folder $newpath already exists")
        clear_folder(newpath)
    else
        os.mkdir(newpath)
        println("Folder $newpath created!")
    end
end

# using OpenCV
# using Images

# Trim Whitespace Section
function trim_whitespace_image(image)
    imageo =Image.open(image)
    println(typeof(imageo))
    # Convert the image to grayscale
    gray_image = imageo.convert("L")
    println("66marker")
    # gray_image = Gray.(imageo)
    # gray_image = cv2.imread(imageo,0)
    # Convert the grayscale image to a NumPy array
    img_array = np.array(gray_image)
    # img_array = np.array(image)
    # Apply binary thresholding to create a binary image 
    # (change the value here default is 250)    ↓
    _, binary_array = cv2.threshold(img_array, 250, 255, cv2.THRESH_BINARY_INV)
    # Find connected components in the binary image
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_array)
    # Find the largest connected component (excluding the background)
    largest_component_label = np.argmax(stats[2:num_labels, 5]) +2
    lcl = largest_component_label
    println(lcl)
    println(stats[2,1:5])
    # labels == largest_component_label
    # println(labels)
    # binary_matrix = np.matrix(binary_array)
    # vector = size(binary_array)
    # println(typeof(labels))
    # tempx = labels[largest_component_label]
    # println(typeof(tempx))
    # println(largest_component_label)
    # A = load_matrix_network(binary_matrix)
    # _, p = largest_component(binary_array) ###
    # largest_component_mask = (labels == largest_component_label)
    # println(2)
    # println(typeof(largest_component_mask))
    # largest_component_mask = largest_component_label.astype("uint8")
    # largest_component_mask = (labels == largest_component_label).pyastype(np.uint8) * 255
    # Find the bounding box of the largest connected component
    LCL = largest_component_label
    # stata = Int64(stats)
    println(np.shape(stats))
    println(typeof(LCL))
    # println(typeof(cv2.CC_STAT_LEFT))
    # x = np.argmax(stats[largest_component_label, cv2.CC_STAT_LEFT])
    # println(cv2.CC_STAT_LEFT)
    x= stats[largest_component_label, 1]
    y = stats[LCL, 2]
    w = stats[LCL, 3]
    h = stats[LCL, 4]
    println("fuu")
    println(LCL)
    println(x,"/", y,"/", w,"/", h)
    # x, y, w, h = (50, 50, 50, 50)
    println(typeof(y))
    # x, y, w, h = cv2.boundingRect(largest_component_mask)
    # Crop the image to the bounding box
    cropped_image = imageo.crop((x, y, x + w, y + h))
    return cropped_image
end


######################################################
## list of images in the images file
## if the file isnot ./images change it here
######################################################
pathname = "./images"
files = os.listdir(pathname)

listimg = []


for file in files
    # make sure file is an image
    if endswith(file, r".jpg|.png|.jpeg|.JPG")
        img_path = pathname*"/"*file
        push!(listimg, img_path) #this adds img_path to listimg #in place of list.append
    end
end


######################################################################
# create new directory named re_im
# os.mkdir(path)        

newpath = "./re_im"

mkfolder(newpath)


######################################################################
#######################################################################
#$ read image files - resize then save in re_im file
########################################################################
#######################################################################

### testing imageopen and img.size
# for file in listimg:
#     # image = pathname +'/' +file
#     # image = resize(image)
#     img=Image.open(file)
#     x, y =img.size
#     print(x,y)

# for name in listimg:
#     print(name) 


#######################################################################
## 1 - crop the images (remove white border)
## 2 - resize to the desired size
## 3 - save
######################################################################



############################################################
## todo: resize images if necessary
##
############################################################

sizes = (800,300)


## function that resize the image 
function resize(cropped)
    # imageo =Image.open(image)
    # cropped = trim_whitespace_image(image)
    # imageBox = image.getbbox()
    # cropped = image.crop(imageBox)
    x, y =cropped.size
    # out = im.resize(size)
    println(x,",",y)
    if (x,y) != sizes 
        reim = ImageOps.pad(cropped, sizes, color="#fff")#.save("imageops_pad.png")
        # reim = ImageOps.pad(cropped, size, color="#fff")#.save("imageops_pad.png")
        # out.save(image)
    else
        reim = cropped
    end
    return reim
end

# println("foo")
# print(files)
for file in files
    # println(file)
    if endswith(file, r".png|.jpg|.eps|.jpeg")
        println(file)
        image = pathname *"/" * file
        println("fii")
        cropped = trim_whitespace_image(image)
        println("fool")
        reimg = resize(cropped)
        println("foo")
        savefile = newpath *"/"*file
        reimg.save(savefile)
    end
end
