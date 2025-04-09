using Ripserer
using MetricSpaces

function persistence_diagram(X::MetricSpace; cutoff = 0.1)
    pd = ripserer(X; cutoff = cutoff)
    pd
end

function plot_pd_barcode(pd)
    barcode(pd)
end

function plot_pd(pd)
    plot(pd)
end
