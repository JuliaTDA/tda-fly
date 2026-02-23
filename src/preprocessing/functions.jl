using Images, ImageFiltering, ImageTransformations
using Random
using MetricSpaces
using ..TDAfly

const NEIGHBOR_OFFSETS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
const NEIGHBOR_OFFSETS_8 = (
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
)

function crop_image(img; threshold = 0.8)
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

function _neighbor_offsets(connectivity::Integer)
    connectivity == 4 && return NEIGHBOR_OFFSETS_4
    connectivity == 8 && return NEIGHBOR_OFFSETS_8
    throw(ArgumentError("connectivity must be 4 or 8. Got $(connectivity)."))
end

function pixel_components(ids::Vector{<:AbstractVector{<:Integer}}; connectivity::Integer = 8)
    isempty(ids) && return Vector{Vector{NTuple{2, Int}}}()

    offsets = _neighbor_offsets(connectivity)
    points = Set{NTuple{2, Int}}((Int(p[1]), Int(p[2])) for p in ids)
    visited = Set{NTuple{2, Int}}()
    components = Vector{Vector{NTuple{2, Int}}}()
    stack = NTuple{2, Int}[]

    for point in points
        point in visited && continue
        push!(visited, point)
        push!(stack, point)
        component = NTuple{2, Int}[]

        while !isempty(stack)
            current = pop!(stack)
            push!(component, current)

            for (dx, dy) in offsets
                neighbor = (current[1] + dx, current[2] + dy)
                if (neighbor in points) && !(neighbor in visited)
                    push!(visited, neighbor)
                    push!(stack, neighbor)
                end
            end
        end

        push!(components, component)
    end

    components
end

function _closest_points(c1::Vector{NTuple{2, Int}}, c2::Vector{NTuple{2, Int}})
    best_p = c1[1]
    best_q = c2[1]
    best_d = typemax(Int)

    for p in c1, q in c2
        dx = p[1] - q[1]
        dy = p[2] - q[2]
        d = dx * dx + dy * dy
        if d < best_d
            best_p = p
            best_q = q
            best_d = d
        end
    end

    best_p, best_q, best_d
end

function _line_pixels_8(p::NTuple{2, Int}, q::NTuple{2, Int})
    x1, y1 = p
    x2, y2 = q
    points = NTuple{2, Int}[]

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = x1 < x2 ? 1 : -1
    sy = y1 < y2 ? 1 : -1
    err = dx - dy
    x, y = x1, y1

    while true
        push!(points, (x, y))
        (x == x2 && y == y2) && break
        e2 = 2 * err
        if e2 > -dy
            err -= dy
            x += sx
        end
        if e2 < dx
            err += dx
            y += sy
        end
    end

    points
end

function _line_pixels_4(p::NTuple{2, Int}, q::NTuple{2, Int})
    x1, y1 = p
    x2, y2 = q
    points = NTuple{2, Int}[(x1, y1)]
    x, y = x1, y1

    while x != x2
        x += x < x2 ? 1 : -1
        push!(points, (x, y))
    end
    while y != y2
        y += y < y2 ? 1 : -1
        push!(points, (x, y))
    end

    points
end

function connect_pixel_components(ids::Vector{<:AbstractVector{<:Integer}}; connectivity::Integer = 8)
    components = pixel_components(ids; connectivity = connectivity)
    length(components) <= 1 && return ids

    edges = Tuple{Int, Int, Int, NTuple{2, Int}, NTuple{2, Int}}[]
    for i in 1:(length(components) - 1)
        for j in (i + 1):length(components)
            p, q, d = _closest_points(components[i], components[j])
            push!(edges, (d, i, j, p, q))
        end
    end
    sort!(edges, by = first)

    parent = collect(1:length(components))
    rank = zeros(Int, length(components))

    function find_root(i::Int)
        while parent[i] != i
            parent[i] = parent[parent[i]]
            i = parent[i]
        end
        i
    end

    function union_sets(i::Int, j::Int)
        ri = find_root(i)
        rj = find_root(j)
        ri == rj && return false

        if rank[ri] < rank[rj]
            parent[ri] = rj
        elseif rank[ri] > rank[rj]
            parent[rj] = ri
        else
            parent[rj] = ri
            rank[ri] += 1
        end
        true
    end

    points = Set{NTuple{2, Int}}((Int(p[1]), Int(p[2])) for p in ids)
    n_edges = 0
    for (_, i, j, p, q) in edges
        union_sets(i, j) || continue
        bridge_pixels = connectivity == 4 ? _line_pixels_4(p, q) : _line_pixels_8(p, q)
        union!(points, bridge_pixels)
        n_edges += 1
        n_edges == length(components) - 1 && break
    end

    sorted_points = sort!(collect(points), by = t -> (t[1], t[2]))
    [[p[1], p[2]] for p in sorted_points]
end

function image_to_r2(img::Matrix; threshold = 0.2, ensure_connected = false, connectivity::Integer = 8)
    A = image_to_array(img)
    ids = findall_ids(>(threshold), A)
    if ensure_connected
        ids = connect_pixel_components(ids; connectivity = connectivity)
    end
    EuclideanSpace(ids)
end
