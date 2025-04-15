# import CairoMakie as CM
using Distances
using Plots
using ..TDAfly.TDA: plot_pd

# function plot_wing(X)
#     CM.scatter(X, axis = (;aspect=DataAspect()), )
# end

function pairwise_distance(Ms, d = euclidean)
    n = length(Ms)
    D = zeros(n, n)
    for i ∈ 1:n
        for j ∈ i:n
            D[j, i] = D[i, j] = d(Ms[i], Ms[j])            
        end
    end
    
    D
end

function plot_wing_with_pd(pd, image, sample, title)
    l = @layout [a b; c]
  
    plot(
        plot_pd(pd, persistence = true)
        ,heatmap(image)
        ,scatter(last.(sample), first.(sample))
        ,layout = l
        ,plot_title = title
    )
end;

function plot_heatmap(D, labels, title = "")
    xticks = ([1:size(D)[1];], labels)

    heatmap(D, xticks = xticks, yticks = xticks, title = title)
end