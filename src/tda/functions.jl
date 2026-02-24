using ..TDAfly
using Reexport
using Ripserer
using Plots
using MetricSpaces
using PersistenceDiagrams
using StatsBase: mean, median, std, quantile
using LinearAlgebra: norm
using Images: feature_transform, distance_transform

@reexport using MetricSpaces: random_sample

# =============================================================================
# Persistent Homology Computations
# =============================================================================

"""
    rips_pd_1d(X; kwargs...)

Compute 1-dimensional persistence diagram via Vietoris-Rips filtration.
Returns only the H1 diagram (loops).
"""
function rips_pd_1d(X; kwargs...)
    result = ripserer(X; dim_max=1, kwargs...)
    result[2]
end

"""
    cubical_pd(img_array; dim_max=1)

Compute persistence diagrams via cubical (sublevel-set) filtration on a grayscale image.
The image is inverted so that dark pixels (veins) have low filtration values.
Returns a tuple (pd0, pd1).
"""
function cubical_pd(A::AbstractMatrix; dim_max=1)
    A2 = 1.0 .- Float64.(A)
    result = ripserer(Cubical(A2); dim_max=dim_max)
    dim_max == 0 ? (result[1],) : (result[1], result[2])
end

"""
    cubical_pd_1d(img_array)

Compute only the 1-dimensional cubical persistence diagram.
"""
function cubical_pd_1d(A::AbstractMatrix)
    pd0, pd1 = cubical_pd(A; dim_max=1)
    pd1
end

# =============================================================================
# Directional (Height) Filtrations
# =============================================================================

"""
    height_filtration(A, direction)

Build a filtration array by assigning each foreground pixel a value
equal to its projection onto `direction` (a 2-vector, e.g. [1,0] for horizontal).
Background pixels get a high filtration value (they appear last).

The resulting array can be fed into `Cubical()` for sublevel-set persistence.
"""
function height_filtration(A::AbstractMatrix, direction::Vector{<:Real}; threshold=0.5)
    d = direction ./ norm(direction)
    rows, cols = size(A)
    F = fill(Inf, rows, cols)
    for i in 1:rows, j in 1:cols
        if (1.0 - Float64(A[i, j])) > threshold
            F[i, j] = d[1] * i + d[2] * j
        end
    end
    # Normalize to [0, 1] for foreground pixels
    foreground = F[isfinite.(F)]
    if !isempty(foreground)
        lo, hi = extrema(foreground)
        if hi > lo
            F[isfinite.(F)] .= (foreground .- lo) ./ (hi - lo)
        else
            F[isfinite.(F)] .= 0.0
        end
    end
    # Set background to value > 1 so it appears last
    F[.!isfinite.(F)] .= 2.0
    F
end

"""
    directional_pd_1d(img_array, direction)

Compute 1-dimensional persistence via a directional (height) filtration.
`direction` is a 2-vector, e.g. `[1,0]` (top-to-bottom), `[0,1]` (left-to-right),
`[1,1]` (diagonal).
"""
function directional_pd_1d(A::AbstractMatrix, direction::Vector{<:Real})
    F = height_filtration(A, direction)
    result = ripserer(Cubical(F); dim_max=1)
    result[2]
end

"""
    directional_pd_0d(img_array, direction)

Compute 0-dimensional persistence via a directional (height) filtration.
Captures when disconnected vein segments merge as the sweep progresses ---
directly related to vein count and branching patterns.
"""
function directional_pd_0d(A::AbstractMatrix, direction::Vector{<:Real})
    F = height_filtration(A, direction)
    result = ripserer(Cubical(F); dim_max=1)
    result[1]
end

"""
    radial_filtration(A; threshold=0.5)

Build a filtration where each foreground pixel is assigned its distance to the
centroid of all foreground pixels. Captures radial structure of the wing.
"""
function radial_filtration(A::AbstractMatrix; threshold=0.5)
    rows, cols = size(A)
    foreground_pixels = Tuple{Int,Int}[]
    for i in 1:rows, j in 1:cols
        if (1.0 - Float64(A[i, j])) > threshold
            push!(foreground_pixels, (i, j))
        end
    end

    isempty(foreground_pixels) && return fill(2.0, rows, cols)

    cx = mean(first.(foreground_pixels))
    cy = mean(last.(foreground_pixels))

    F = fill(2.0, rows, cols)
    for (i, j) in foreground_pixels
        F[i, j] = sqrt((i - cx)^2 + (j - cy)^2)
    end
    # Normalize foreground to [0, 1]
    foreground_vals = [F[i, j] for (i, j) in foreground_pixels]
    lo, hi = extrema(foreground_vals)
    if hi > lo
        for (i, j) in foreground_pixels
            F[i, j] = (F[i, j] - lo) / (hi - lo)
        end
    else
        for (i, j) in foreground_pixels
            F[i, j] = 0.0
        end
    end
    F
end

"""
    radial_pd_1d(img_array)

Compute 1-dimensional persistence via a radial (distance-from-centroid) filtration.
"""
function radial_pd_1d(A::AbstractMatrix)
    F = radial_filtration(A)
    result = ripserer(Cubical(F); dim_max=1)
    result[2]
end

"""
    radial_pd_0d(img_array)

Compute 0-dimensional persistence via a radial (distance-from-centroid) filtration.
Captures how disconnected vein segments merge as the radial sweep grows outward.
"""
function radial_pd_0d(A::AbstractMatrix)
    F = radial_filtration(A)
    result = ripserer(Cubical(F); dim_max=1)
    result[1]
end

# =============================================================================
# Euclidean Distance Transform (EDT) Filtration
# =============================================================================

"""
    edt_filtration(A::AbstractMatrix; threshold=0.5)

Build a filtration from the Euclidean Distance Transform (EDT).
Each foreground pixel is assigned its distance to the nearest background pixel.
Thick veins (C, R, M) get high values. Negating the EDT and using sublevel-set
persistence captures the vein thickness hierarchy.

Returns a matrix suitable for `Cubical()`.
"""
function edt_filtration(A::AbstractMatrix; threshold=0.5)
    # Binary mask: foreground = true (dark pixels = veins)
    binary = (1.0 .- Float64.(A)) .> threshold
    rows, cols = size(A)

    if !any(binary) || all(binary)
        return fill(2.0, rows, cols)
    end

    # Compute EDT using Images.jl
    ft = feature_transform(binary)
    edt_vals = distance_transform(ft)

    # Negate so thick veins (high EDT) appear first (low filtration value)
    # Background gets high filtration value
    F = fill(2.0, rows, cols)
    max_edt = maximum(edt_vals[binary])
    if max_edt > 0
        for i in 1:rows, j in 1:cols
            if binary[i, j]
                F[i, j] = 1.0 - edt_vals[i, j] / max_edt  # Normalize to [0, 1]
            end
        end
    end
    F
end

"""
    edt_pd_1d(img_array)

Compute 1-dimensional persistence from EDT filtration.
Captures loops formed by thick vein structures.
"""
function edt_pd_1d(A::AbstractMatrix)
    F = edt_filtration(A)
    result = ripserer(Cubical(F); dim_max=1)
    result[2]
end

"""
    edt_pd_0d(img_array)

Compute 0-dimensional persistence from EDT filtration.
Captures how vein thickness components merge.
"""
function edt_pd_0d(A::AbstractMatrix)
    F = edt_filtration(A)
    result = ripserer(Cubical(F); dim_max=1)
    result[1]
end

# =============================================================================
# Plotting
# =============================================================================

plot_barcode(pd) = barcode(pd)
plot_pd(pd; kwargs...) = plot(pd; kwargs...)

# =============================================================================
# Array manipulation (for custom filtrations)
# =============================================================================

function modify_array(A, f::Function)
    ids = findall_ids(>(0.3), A)
    A2 = zero(A)
    for (x, y) in ids
        A2[x, y] = f(x, y)
    end
    A2 ./ maximum(A2)
end

function dist_to_point(a, b)
    (x, y) -> sqrt((x - a)^2 + (y - b)^2)
end

function dist_to_line((a1, b1), (a2, b2))
    function (x, y)
        T1 = (b2 - b1) * x - (a2 - a1) * y + a2 * b1 - b2 * a1
        T2 = sqrt((a2 - a1)^2 + (b2 - b1)^2)
        abs(T1) / T2
    end
end

# =============================================================================
# TDA Statistics - Feature extraction from persistence diagrams
# =============================================================================

"""
    count_intervals(pd; threshold=0.0)

Count the number of intervals with persistence > threshold.
"""
function count_intervals(pd; threshold=0.0)
    count(i -> isfinite(persistence(i)) && persistence(i) > threshold, pd)
end

"""
    max_persistence(pd)

Maximum persistence value in the diagram.
"""
function max_persistence(pd)
    vals = [persistence(i) for i in pd if isfinite(persistence(i))]
    isempty(vals) ? 0.0 : maximum(vals)
end

"""
    total_persistence(pd; p=1)

Sum of persistence^p values. p=1 is total persistence, p=2 is total squared persistence.
"""
function total_persistence(pd; p=1)
    vals = [persistence(i) for i in pd if isfinite(persistence(i))]
    isempty(vals) ? 0.0 : sum(v^p for v in vals)
end

"""
    median_persistence(pd)

Median persistence value.
"""
function median_persistence(pd)
    vals = [persistence(i) for i in pd if isfinite(persistence(i))]
    isempty(vals) ? 0.0 : median(vals)
end

"""
    quantile_persistence(pd, q)

q-th quantile of persistence values.
"""
function quantile_persistence(pd, q)
    vals = [persistence(i) for i in pd if isfinite(persistence(i))]
    isempty(vals) ? 0.0 : quantile(vals, q)
end

"""
    persistence_entropy(pd)

Normalized Shannon entropy of persistence values.
Higher entropy = more uniform distribution of lifetimes.
"""
function persistence_entropy(pd)
    vals = [persistence(i) for i in pd if isfinite(persistence(i))]
    isempty(vals) && return 0.0
    L = sum(vals)
    L == 0 && return 0.0
    probs = [v / L for v in vals]
    -sum(p > 0 ? p * log(p) : 0.0 for p in probs)
end

"""
    mean_birth(pd)

Mean birth time across all intervals.
"""
function mean_birth(pd)
    vals = [birth(i) for i in pd if isfinite(persistence(i))]
    isempty(vals) ? 0.0 : mean(vals)
end

"""
    mean_death(pd)

Mean death time across all intervals.
"""
function mean_death(pd)
    vals = [death(i) for i in pd if isfinite(death(i))]
    isempty(vals) ? 0.0 : mean(vals)
end

"""
    std_persistence(pd)

Standard deviation of persistence values.
"""
function std_persistence(pd)
    vals = [persistence(i) for i in pd if isfinite(persistence(i))]
    isempty(vals) ? 0.0 : std(vals)
end

"""
    pd_statistics(pd; threshold=0.0)

Extract a comprehensive feature vector from a persistence diagram.
Returns 11 features:
[count, max_pers, total_pers, total_pers2, q10, q25, median, q75, q90, entropy, std_pers]
"""
function pd_statistics(pd; threshold=0.0)
    [
        Float64(count_intervals(pd; threshold=threshold)),
        max_persistence(pd),
        total_persistence(pd; p=1),
        total_persistence(pd; p=2),
        quantile_persistence(pd, 0.10),
        quantile_persistence(pd, 0.25),
        median_persistence(pd),
        quantile_persistence(pd, 0.75),
        quantile_persistence(pd, 0.90),
        persistence_entropy(pd),
        std_persistence(pd)
    ]
end
