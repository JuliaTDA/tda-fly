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
