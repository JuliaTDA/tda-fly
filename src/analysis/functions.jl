# import CairoMakie as CM
using Distances
using Plots
using Random: shuffle
using ..TDAfly.TDA: plot_pd
using StatsBase: mean, median, std;

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

# =============================================================================
# Distance Matrices for Persistence Diagrams
# =============================================================================

using PersistenceDiagrams: Bottleneck, Wasserstein

"""
    bottleneck_distance_matrix(pds)

Compute the pairwise bottleneck distance matrix for a collection of persistence diagrams.
"""
function bottleneck_distance_matrix(pds)
    n = length(pds)
    D = zeros(n, n)
    bn = Bottleneck()
    for i in 1:n
        for j in i+1:n
            D[i, j] = D[j, i] = bn(pds[i], pds[j])
        end
    end
    D
end

"""
    wasserstein_distance_matrix(pds; q=1)

Compute the pairwise Wasserstein-q distance matrix for a collection of persistence diagrams.
Common choices: q=1 (Earth mover's distance) or q=2.
"""
function wasserstein_distance_matrix(pds; q=1)
    n = length(pds)
    D = zeros(n, n)
    ws = Wasserstein(q)
    for i in 1:n
        for j in i+1:n
            D[i, j] = D[j, i] = ws(pds[i], pds[j])
        end
    end
    D
end

# =============================================================================
# Feature Normalization
# =============================================================================

# using Statistics: mean, std

"""
    minmax_normalize(X::Matrix)

Scale features to [0, 1] range.
"""
function minmax_normalize(X::Matrix)
    mins = minimum(X, dims=1)
    maxs = maximum(X, dims=1)
    ranges = maxs .- mins
    ranges[ranges .== 0] .= 1  # Avoid division by zero
    (X .- mins) ./ ranges
end

"""
    zscore_normalize(X::Matrix)

Normalize features to zero mean and unit variance.
"""
function zscore_normalize(X::Matrix)
    mu = mean(X, dims=1)
    sigma = std(X, dims=1)
    sigma[sigma .== 0] .= 1  # Avoid division by zero
    (X .- mu) ./ sigma
end

# =============================================================================
# Classification - k-NN with precomputed distance matrices
# =============================================================================

using StatsBase: countmap

"""
    knn_predict(D::Matrix, y::Vector, test_idx::Int; k=1)

Predict the label for a single test sample using k-NN with a precomputed distance matrix.
"""
function knn_predict(D::Matrix, y::Vector, test_idx::Int; k=1)
    n = size(D, 1)
    train_idx = setdiff(1:n, test_idx)

    # Distances from test point to all training points
    dists = [(D[test_idx, j], y[j]) for j in train_idx]
    sort!(dists, by=first)

    # k nearest neighbors - majority vote
    neighbors = dists[1:min(k, length(dists))]
    neighbor_labels = [x[2] for x in neighbors]

    counts = countmap(neighbor_labels)
    argmax(counts)
end

"""
    loocv_knn(D::Matrix, y::Vector; k=1)

Perform leave-one-out cross-validation with k-NN using a precomputed distance matrix.
Returns (accuracy, predictions).
"""
function loocv_knn(D::Matrix, y::Vector; k=1)
    n = length(y)
    predictions = [knn_predict(D, y, i; k=k) for i in 1:n]
    accuracy = mean(predictions .== y)
    (accuracy=accuracy, predictions=predictions)
end

# =============================================================================
# Statistical Validation
# =============================================================================

using Distributions: Normal, quantile

"""
    wilson_ci(k::Int, n::Int; alpha=0.05)

Compute the Wilson score confidence interval for a proportion.
k = number of successes, n = total trials.
"""
function wilson_ci(k::Int, n::Int; alpha=0.05)
    z = quantile(Normal(), 1 - alpha/2)
    p_hat = k / n

    denom = 1 + z^2/n
    center = (p_hat + z^2/(2n)) / denom
    margin = z * sqrt(p_hat*(1-p_hat)/n + z^2/(4n^2)) / denom

    (lower = max(0.0, center - margin), upper = min(1.0, center + margin))
end

"""
    confusion_matrix(y_true::Vector, y_pred::Vector)

Compute the confusion matrix for classification results.
Returns (matrix, classes).
"""
function confusion_matrix(y_true::Vector, y_pred::Vector)
    classes = sort(unique(vcat(y_true, y_pred)))
    n_classes = length(classes)
    cm = zeros(Int, n_classes, n_classes)

    for (true_label, pred_label) in zip(y_true, y_pred)
        i = findfirst(==(true_label), classes)
        j = findfirst(==(pred_label), classes)
        cm[i, j] += 1
    end

    (matrix=cm, classes=classes)
end

"""
    permutation_test(D::Matrix, y::Vector; k=1, n_permutations=10000)

Perform a permutation test to assess statistical significance of classification accuracy.
"""
function permutation_test(D::Matrix, y::Vector; k=1, n_permutations=10000)
    # Observed accuracy
    observed = loocv_knn(D, y; k=k).accuracy

    # Permutation distribution
    perm_accuracies = Vector{Float64}(undef, n_permutations)
    for i in 1:n_permutations
        y_perm = shuffle(y)
        perm_accuracies[i] = loocv_knn(D, y_perm; k=k).accuracy
    end

    # p-value: proportion of permuted accuracies >= observed
    p_value = mean(perm_accuracies .>= observed)

    (observed=observed, p_value=p_value,
     perm_mean=mean(perm_accuracies), perm_std=std(perm_accuracies))
end

"""
    classification_report(D::Matrix, y::Vector; k=1, n_permutations=10000)

Generate a complete classification report with LOOCV accuracy, confidence intervals,
confusion matrix, and permutation test.
"""
function classification_report(D::Matrix, y::Vector; k=1, n_permutations=10000)
    # LOOCV
    result = loocv_knn(D, y; k=k)
    acc = result.accuracy
    preds = result.predictions
    n_correct = sum(preds .== y)
    n = length(y)

    # Confusion matrix
    cm_result = confusion_matrix(y, preds)
    cm = cm_result.matrix
    classes = cm_result.classes

    # Per-class metrics (for binary classification)
    if length(classes) == 2
        tp = cm[1, 1]
        fn = cm[1, 2]
        fp = cm[2, 1]
        tn = cm[2, 2]
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
    else
        sensitivity = specificity = NaN
    end

    # Confidence interval
    ci = wilson_ci(n_correct, n)

    # Permutation test
    perm = permutation_test(D, y; k=k, n_permutations=n_permutations)

    (accuracy=acc, n_correct=n_correct, n_total=n,
     ci_95=ci, confusion_matrix=cm, classes=classes,
     sensitivity=sensitivity, specificity=specificity,
     p_value=perm.p_value, chance_level=perm.perm_mean)
end