using ..TDAfly
using Reexport
using Ripserer
import Plots as PD
using MetricSpaces

@reexport using MetricSpaces: random_sample

function rips_pd(X::MetricSpace; cutoff = 0.1)
    pd = ripserer(X; cutoff = cutoff)
    pd
end

function cubical_pd(A::Array; cutoff = 0.1)
    pd = ripserer(Cubical(-A); cutoff = cutoff)
    pd
end

function plot_barcode(pd)
    barcode(pd)
end

function plot_pd(pd)
    PD.plot(pd)
end


function modify_array(A, f::Function)
    ids = findall_ids(>(0.5), A)
    A2 = zero(A)
    for (x, y) in ids
        A2[x, y] = f(x, y, A = A)
    end
    
    A2
end

function dist_to_point(a, b)
    function (x, y; kwargs...)
        sqrt((x - a)^2 + (y - b)^2)
    end
end

function dist_to_line((a1, b1), (a2, b2))
    function (x, y; kwargs...)
        T1 = (b2-b1)*x - (a2-a1)*y + a2*b1 - b2*a1
        T2 = sqrt((a2-a1)^2 + (b2-b1)^2)

        abs(T1) / T2
    end
end
