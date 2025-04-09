using CairoMakie

function plot_wing(X)
    scatter(X, axis = (;aspect=DataAspect()), )
end