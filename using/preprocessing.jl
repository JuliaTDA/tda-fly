using TDAfly
using TDAfly.Preprocessing
import MetricSpaces as MS
using CairoMakie
using Ripserer
import Plots as PD
using Images, ImageFiltering, ImageTransformations


# reading image and converting to points in R^2
path = "images/processed/Asilidae 11.png"
img_gray = image_to_grey(path, blur = 1)
pts = image_to_r2(img_gray)

# no need to read image in other step
path = "images/processed/Asilidae 11.png"
pts = image_to_r2(path)

X = MS.EuclideanSpace(pts)
# MS.farthest_points_sample(X, 1000)

X2 = MS.random_sample(X, 1000)

scatter(X2, axis = (;aspect=DataAspect()))

img_gray


img = imresize(img_gray, ratio = ratio)
length(img)

img = zeros(150, 150)
for i ∈ 1:size(img)[1], j ∈ 1:size(img)[2]
    img[i, j] = 1/((i-50)^2 + (j-50)^2)
end

plot(img)
pd = ripserer(Cubical(img))
PD.plot(pd)
barcode(pd)

img_gray = image_to_grey(path, blur = 1)
img = resize_image(img_gray)
M = image_to_array(img)
M .= (1 .- M)
heatmap(M)

ids = findall(>(0.5), M)
ids[1]

### fazer função que colore a imagem dada um critério:
# distância até reta, etc.
