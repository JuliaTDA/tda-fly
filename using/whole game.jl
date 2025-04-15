using TDAfly
using TDAfly.Preprocessing
using TDAfly.TDA
# using CairoMakie: heatmap
using Plots: heatmap, plot, @layout

# reading image and converting to points in R^2
paths = readdir("images/processed", join = true)
path = paths[1]
img = load_wing(path)
A = img |> image_to_array
A

X = image_to_r2(img)
X_rand = random_sample(X, 100)

rips_pd(X_rand)
pd = rips_pd(X_rand, cutoff = 10, threshold = 200)

plot_barcode(pd)
plot_pd(pd)


# barcodes
## rips case
using PersistenceDiagrams
paths = readdir("images/processed", join = true)

rips = @showprogress map(paths) do path
    # load the image
    img = load_wing(path)
    
    # convert to points in R^2
    X = image_to_r2(img)
    
    # select a random sample
    X_rand = random_sample(X, 750)
    
    # calculate the vietoris-rips barcode
    pd = rips_pd(X_rand, cutoff = 5, threshold = 200)

    # return only 1-dimensional pds
    pd[2]
end

images = load_wing.(paths)

PI = PersistenceImage(rips, size = (10, 10))

pers_images = PI.(rips)

i = 3
rip = rips[i];
pers_img = pers_images[i];
img = images[i];

l = @layout [a b; c]

plot(
    plot_pd(rip, persistence = true)
    ,heatmap(pers_img)
    ,heatmap(img)
    ,layout = l
)



[pers_im pers_im] |> heatmap


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



for  in 
    
end