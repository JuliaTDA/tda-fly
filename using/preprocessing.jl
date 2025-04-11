using TDAfly
using TDAfly.Preprocessing
using TDAfly.TDA

# reading image and converting to points in R^2
path = "images/processed/Asilidae 11.png"
img = load_wing(path, blur = 1, pixels = 150)
A = img |> image_to_array
A

X = image_to_r2(img)
X_rand = random_sample(X, 100)

rips_pd(X_rand)

# poinsts
f = dist_to_point(0, 0)
A2 = modify_array(A, f)
heatmap(A2)

f = dist_to_point(25, 20)
A2 = modify_array(A, f)
heatmap(A2)

f = dist_to_point(55, 420)
A2 = modify_array(A, f)
heatmap(A2)

# lines
f = dist_to_line((0, 0), (1, 1))
A2 = modify_array(A, f)
heatmap(A2)

f = dist_to_line((0, 0), (25, 100))
A2 = modify_array(A, f)
heatmap(A2)
pd = cubical_pd(A2)
plot_barcode(pd)

f = dist_to_line((0, 0), (0, 1))
A2 = modify_array(A, f)
heatmap(A2)
pd = cubical_pd(A2)
plot_barcode(pd)