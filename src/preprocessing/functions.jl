using Images, ImageFiltering, ImageTransformations
using Random
using MetricSpaces

function image_to_grey(path::AbstractString; blur = 1)
    imfilter(load(path), Kernel.gaussian(blur)) .|> Gray
end

function image_to_array(path::AbstractString)
    image = image_to_grey(path)
    convert(Array{Float32}, image)
end

function image_to_array(image::Matrix)
    convert(Array{Float32}, image)
end

function image_to_r2(image::AbstractString; threshold = 0.5, blur = 1)
    image = image_to_grey(image, blur = blur)
    image_to_r2(image, threshold = threshold)
end

function image_to_r2(image::Matrix; threshold = 0.5)
    m = image_to_array(image)
    pts = [[i[1], i[2]] for i ∈ findall(<(0.5), m)]
    EuclideanSpace(pts)
end

function resize_image(img; pixels = 150)
    ratio = 150 / size(img)[1]
    imresize(img, ratio = ratio)    
end
