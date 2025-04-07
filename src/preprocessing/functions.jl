using Images

function image_to_grey(path)        
    img_gray = load(path) .|> Gray    

    img_gray
end

function image_to_r2(image::AbstractString, threshold = 0.5)
    image = image_to_grey(image)
    image_to_r2(image, threshold)    
end

function image_to_r2(image::Matrix, threshold = 0.5)    
    m = convert(Array{Float32}, image)
    pts = [[i[1], i[2]] for i ∈ findall(<(0.5), m)]

    pts
end

