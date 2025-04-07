using TDAfly
using TDAfly.Preprocessing

path = "images/processed/Asilidae 11.png"
img_gray = image_to_grey(path)
pts = image_to_r2(img_gray)
pts = image_to_r2(path)

