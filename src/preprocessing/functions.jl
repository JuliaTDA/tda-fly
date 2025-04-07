using Images, ImageFiltering

function image_to_grey(path; blur = 1)        
    imfilter(load(path), Kernel.gaussian(blur)) .|> Gray
end

function image_to_r2(image::AbstractString; threshold = 0.5, blur = 1)
    image = image_to_grey(image, blur = blur)
    image_to_r2(image, threshold = threshold)
end

function image_to_r2(image::Matrix; threshold = 0.5)
    m = convert(Array{Float32}, image)
    pts = [[i[1], i[2]] for i ∈ findall(<(0.5), m)]

    pts
end
