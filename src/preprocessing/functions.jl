using Images, ImageFiltering, ImageTransformations
using Random
using MetricSpaces
using ..TDAfly

function crop_image(img; threshold = 0.5)
    ids = findall_ids(<(threshold), img)
    x1, x2 = extrema(first.(ids))
    y1, y2 = extrema(last.(ids))
    img[x1:x2, y1:y2]
end

function load_wing(path::AbstractString; blur = 1, pixels = 150)
    img = imfilter(load(path), Kernel.gaussian(blur)) .|> Gray
    cropped_img = crop_image(img)
    resize_image(cropped_img, pixels = pixels)
end

function resize_image(img; pixels = 150)
    ratio = pixels / size(img)[1]
    imresize(img, ratio = ratio)
end

function image_to_array(img::Matrix)
    A = convert(Array{Float32}, img)
    A .|> (x -> 1-x)        
end

function image_to_r2(img::Matrix; threshold = 0.5)
    A = image_to_array(img)
    findall_ids(<(0.5), A) |> EuclideanSpace
end

