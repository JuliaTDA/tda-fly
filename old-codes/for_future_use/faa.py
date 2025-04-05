# -*- coding: utf-8 -*-

''' we wil put all the functions here'''

import shutill

####################################
## empty the indicated folder
def clear_folder(folder):
# folder = '/path/to/folder'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return

#########################################################################
def mkfolder(newpath):
# check whether directory already exists BEFORE trying to create file
    if not os.path.exists(newpath):
        os.mkdir(newpath)
        print("Folder %s created!" % newpath)
    else:
        print("Folder %s already exists" % newpath)
        clear_folder(newpath)

########################################################################

# Trim Whitespace Section
def trim_whitespace_image(image):
    # Convert the image to grayscale
    gray_image = image.convert('L')

    # Convert the grayscale image to a NumPy array
    img_array = np.array(gray_image)
    # img_array = np.array(image)

    # Apply binary thresholding to create a binary image 
    # (change the value here default is 250)    ↓
    _, binary_array = cv2.threshold(img_array, 250, 255, cv2.THRESH_BINARY_INV)

    # Find connected components in the binary image
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_array)

    # Find the largest connected component (excluding the background)
    largest_component_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    largest_component_mask = (labels == largest_component_label).astype(np.uint8) * 255

    # Find the bounding box of the largest connected component
    x, y, w, h = cv2.boundingRect(largest_component_mask)

    # Crop the image to the bounding box
    cropped_image = image.crop((x, y, x + w, y + h))

    return cropped_image

######################################################################################
##thin the dendogram
def thin_den(dendogram, cparm):
    rip_list = []
    for n in range(len(dendogram)):
        # print(n)
        # print(dendogram[n])
        comp = dendogram[n][1]-dendogram[n][0]
        # print(n, comp)
        ## comp >5 seens to be the ideal
        if comp> cparm:          ##### parametro ajustavel aqui ######
            rip_list.append(dendogram[n])
    red_dendogram = np.array(rip_list)      
    print(len(dendogram), len(red_dendogram))      
    return red_dendogram

