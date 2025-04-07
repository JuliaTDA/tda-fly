using TDAfly
using TDAfly.Preprocessing
import MetricSpaces as MS

# reading image and converting to points in R^2
path = "images/processed/Asilidae 11.png"
img_gray = image_to_grey(path, blur = 1)

pts = image_to_r2(img_gray)

# no need to read image in other step
path = "images/processed/Asilidae 11.png"
pts = image_to_r2(path)

X = MS.EuclideanSpace(pts)
MS.farthest_points_sample(X, 1000)

MS.random_sample(X, 1)