using TDAfly
using TDAfly.Preprocessing
using Images: Gray, load, save

const INPUT_DIR = "images/processed"
const OUTPUT_DIR = "images/connected"
const THRESHOLD = 0.2
const CONNECTIVITY = 8

function ids_to_mask(
    ids::Vector{<:AbstractVector{<:Integer}},
    dims::Tuple{Int, Int},
)
    M = zeros(Float32, dims...)
    for p in ids
        i = Int(p[1])
        j = Int(p[2])
        if 1 <= i <= dims[1] && 1 <= j <= dims[2]
            M[i, j] = 1f0
        end
    end
    M
end

function connect_and_save_images(;
    input_dir::AbstractString = INPUT_DIR,
    output_dir::AbstractString = OUTPUT_DIR,
    threshold::Real = THRESHOLD,
    connectivity::Integer = CONNECTIVITY,
)
    mkpath(output_dir)

    paths = filter(readdir(input_dir; join = true)) do p
        endswith(lowercase(p), ".png")
    end
    sort!(paths)

    if isempty(paths)
        println("No PNG images found in: $(input_dir)")
        return
    end

    total_added_pixels = 0
    total_reduced_components = 0

    println("Processing $(length(paths)) image(s)...")
    println("Input:  $(input_dir)")
    println("Output: $(output_dir)")
    println("threshold=$(threshold), connectivity=$(connectivity)")
    println()

    for path in paths
        img = load(path)
        gray = Gray.(img)
        A = image_to_array(gray)

        ids_before = findall_ids(>(threshold), A)
        ids_after = connect_pixel_components(ids_before; connectivity = connectivity)

        n_comp_before = length(pixel_components(ids_before; connectivity = connectivity))
        n_comp_after = length(pixel_components(ids_after; connectivity = connectivity))
        added_pixels = length(ids_after) - length(ids_before)

        total_added_pixels += added_pixels
        total_reduced_components += (n_comp_before - n_comp_after)

        mask_after = ids_to_mask(ids_after, size(gray))
        connected_img = Gray.(1 .- mask_after)

        outpath = joinpath(output_dir, basename(path))
        save(outpath, connected_img)

        println("$(basename(path)): components $(n_comp_before) -> $(n_comp_after), added pixels=$(added_pixels)")
    end

    println()
    println("Done.")
    println("Total added pixels: $(total_added_pixels)")
    println("Total component reduction: $(total_reduced_components)")
end

connect_and_save_images()
