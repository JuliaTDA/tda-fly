using ..TDAfly
using Reexport
using Ripserer
using Plots
using MetricSpaces
using PersistenceDiagrams


@reexport using MetricSpaces: random_sample

# persistent homology
function rips_pd(X::MetricSpace; kwargs...)
    ripserer(X; kwargs...)
end

function cubical_pd(A::Array; kwargs...)
    A2 = -copy(A) .+ 1    
    ripserer(Cubical(A2); kwargs...)

end

# plotting
plot_barcode(pd) = barcode(pd)

plot_pd(pd; kwargs...) = plot(pd; kwargs...)

# array manipulation
function modify_array(A, f::Function)
    ids = findall_ids(>(0.3), A)
    A2 = zero(A)
    for (x, y) in ids
        A2[x, y] = f(x, y)
    end
    
    A2 ./ maximum(A2)
end

# closures to make filtration
function dist_to_point(a, b)
    function (x, y)
        sqrt((x - a)^2 + (y - b)^2)
    end
end

function dist_to_line((a1, b1), (a2, b2))
    function (x, y)
        T1 = (b2-b1)*x - (a2-a1)*y + a2*b1 - b2*a1
        T2 = sqrt((a2-a1)^2 + (b2-b1)^2)

        abs(T1) / T2
    end
end

# =============================================================================
# TDA Statistics - Feature extraction from persistence diagrams
# =============================================================================

"""
    count_intervals(pd; threshold=0.0)

Count the number of intervals in a persistence diagram with persistence > threshold.
"""
function count_intervals(pd; threshold=0.0)
    count(i -> persistence(i) > threshold, pd)
end

"""
    max_persistence(pd)

Return the maximum persistence value in the diagram.
"""
function max_persistence(pd)
    isempty(pd) ? 0.0 : maximum(persistence, pd)
end

"""
    total_persistence(pd)

Return the sum of all persistence values in the diagram.
"""
function total_persistence(pd)
    isempty(pd) ? 0.0 : sum(persistence, pd)
end

"""
    mean_persistence(pd)

Return the mean persistence value in the diagram.
"""
function median_persistence(pd)
    isempty(pd) ? 0.0 : median(persistence.(pd))
end

"""
    persistence_entropy(pd)

Compute the normalized Shannon entropy of persistence values.
Higher entropy indicates more uniform distribution of persistence values.
"""
function persistence_entropy(pd)
    isempty(pd) && return 0.0
    L = total_persistence(pd)
    L == 0 && return 0.0
    probs = [persistence(i) / L for i in pd]
    -sum(p > 0 ? p * log(p) : 0.0 for p in probs)
end

"""
    pd_statistics(pd; threshold=0.0)

Extract a feature vector of TDA statistics from a persistence diagram.
Returns: [count, max_pers, total_pers, median_pers, entropy]
"""
function pd_statistics(pd; threshold=0.0)
    [
        Float64(count_intervals(pd; threshold=threshold)),
        max_persistence(pd),
        total_persistence(pd),
        median_persistence(pd),
        persistence_entropy(pd)
    ]
end
