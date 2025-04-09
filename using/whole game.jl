using TDAfly, TDAfly.Preprocessing, TDAfly.TDA, TDAfly.Analysis
using MetricSpaces
using CairoMakie
using Ripserer
import Plots as PL

# reading image and converting to points in R^2
path = "images/processed/Asilidae 11.png"
img_gray = image_to_grey(path, blur = 1)
pts = image_to_r2(img_gray)

# no need to read image in other step
path = "images/processed/Asilidae 11.png"
pts = image_to_r2(path)
X = random_sample(pts, 500)

scatter(X, axis = (;aspect=DataAspect()))

pd = persistence_diagram(X)
barcode(pd)
PL.plot(pd)