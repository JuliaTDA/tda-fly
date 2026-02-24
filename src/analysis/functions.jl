using Distances
using Plots
using Random: shuffle, shuffle!, rand, MersenneTwister
using ..TDAfly.TDA: plot_pd
using StatsBase: mean, median, std;

function pairwise_distance(Ms, d = euclidean)
    n = length(Ms)
    D = zeros(n, n)
    for i ∈ 1:n
        for j ∈ i:n
            D[j, i] = D[i, j] = d(Ms[i], Ms[j])            
        end
    end
    
    sanitize_distance_matrix(D)
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

"""
    sanitize_feature_matrix(X::Matrix)

Convert to `Float64` and replace non-finite values by the median of finite values.
If no finite value exists, replaces with 0.0.
"""
function sanitize_feature_matrix(X::Matrix)
    Xf = Float64.(copy(X))
    finite_vals = Xf[isfinite.(Xf)]
    fill_value = isempty(finite_vals) ? 0.0 : median(finite_vals)
    Xf[.!isfinite.(Xf)] .= fill_value
    Xf
end

"""
    sanitize_distance_matrix(D::Matrix)

Ensure a distance matrix is finite, symmetric and non-negative.
Non-finite off-diagonal entries are replaced by a large finite distance.
"""
function sanitize_distance_matrix(D::Matrix)
    Ds = Float64.(copy(D))
    n, m = size(Ds)
    n == m || error("Distance matrix must be square")

    finite_offdiag = Float64[]
    for i in 1:n, j in 1:n
        if i != j && isfinite(Ds[i, j])
            push!(finite_offdiag, Ds[i, j])
        end
    end
    fallback = isempty(finite_offdiag) ? 1.0 : maximum(finite_offdiag)
    fallback = fallback <= 0 ? 1.0 : 1.05 * fallback

    for i in 1:n
        Ds[i, i] = 0.0
        for j in i+1:n
            dij = Ds[i, j]
            dji = Ds[j, i]
            if !isfinite(dij) || !isfinite(dji)
                Ds[i, j] = Ds[j, i] = fallback
            else
                v = max(0.0, (dij + dji) / 2)
                Ds[i, j] = Ds[j, i] = v
            end
        end
    end

    Ds
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

"""
    minmax_normalize(X::Matrix)

Scale features to [0, 1] range.
"""
function minmax_normalize(X::Matrix)
    Xf = sanitize_feature_matrix(X)
    mins = minimum(Xf, dims=1)
    maxs = maximum(Xf, dims=1)
    ranges = maxs .- mins
    ranges[ranges .== 0] .= 1
    (Xf .- mins) ./ ranges
end

"""
    zscore_normalize(X::Matrix)

Normalize features to zero mean and unit variance.
"""
function zscore_normalize(X::Matrix)
    Xf = sanitize_feature_matrix(X)
    mu = mean(Xf, dims=1)
    sigma = std(Xf, dims=1)
    sigma[sigma .== 0] .= 1
    (Xf .- mu) ./ sigma
end

function _fit_zscore(X::Matrix)
    Xf = sanitize_feature_matrix(X)
    mu = mean(Xf, dims=1)
    sigma = std(Xf, dims=1)
    sigma[sigma .== 0] .= 1
    (mu=mu, sigma=sigma)
end

function _apply_zscore(X::Matrix, params)
    Xf = sanitize_feature_matrix(X)
    (Xf .- params.mu) ./ params.sigma
end

function _majority_label(y::Vector)
    argmax(countmap(y))
end

# =============================================================================
# Statistical Validation
# =============================================================================

using Distributions: Normal, quantile
using StatsBase: countmap

"""
    wilson_ci(k::Int, n::Int; alpha=0.05)

Compute the Wilson score confidence interval for a proportion.
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
    classification_metrics(y_true, y_pred)

Compute accuracy, balanced accuracy, macro-F1, and per-class recall.
"""
function classification_metrics(y_true::Vector{String}, y_pred::Vector{String})
    cm_result = confusion_matrix(y_true, y_pred)
    cm = cm_result.matrix
    classes = cm_result.classes
    n_classes = length(classes)

    recalls = Float64[]
    f1s = Float64[]
    for i in 1:n_classes
        tp = cm[i, i]
        fn = sum(cm[i, :]) - tp
        fp = sum(cm[:, i]) - tp

        recall = tp + fn == 0 ? 0.0 : tp / (tp + fn)
        precision = tp + fp == 0 ? 0.0 : tp / (tp + fp)
        f1 = precision + recall == 0 ? 0.0 : 2 * precision * recall / (precision + recall)

        push!(recalls, recall)
        push!(f1s, f1)
    end

    per_class_recall = Dict(classes[i] => recalls[i] for i in eachindex(classes))

    (
        accuracy = mean(y_true .== y_pred),
        balanced_accuracy = mean(recalls),
        macro_f1 = mean(f1s),
        per_class_recall = per_class_recall,
        confusion_matrix = cm,
        classes = classes
    )
end

# =============================================================================
# ML Classifiers: LDA and Decision Tree
# =============================================================================

using MultivariateStats: fit, predict, MulticlassLDA, projection, PCA
import DecisionTree
using DecisionTree: build_tree, apply_tree

"""
    encode_labels(labels::Vector{String})

Convert string labels to integer codes. Returns (codes, class_names).
"""
function encode_labels(labels::Vector{String})
    classes = sort(unique(labels))
    codes = [findfirst(==(l), classes) for l in labels]
    (codes=codes, classes=classes)
end

"""
    loocv_lda(X::Matrix, labels::Vector{String})

Leave-one-out cross-validation with Linear Discriminant Analysis.
X is n×p (samples × features).
"""
function loocv_lda(X::Matrix, labels::Vector{String}; standardize=true)
    Xclean = sanitize_feature_matrix(X)
    n = size(X, 1)
    enc = encode_labels(labels)
    predictions = Vector{String}(undef, n)

    for i in 1:n
        train_idx = setdiff(1:n, i)
        X_train_raw = Xclean[train_idx, :]
        X_test_raw = Xclean[i:i, :]
        if standardize
            params = _fit_zscore(X_train_raw)
            X_train_use = _apply_zscore(X_train_raw, params)
            X_test_use = _apply_zscore(X_test_raw, params)
        else
            X_train_use = X_train_raw
            X_test_use = X_test_raw
        end

        X_train = X_train_use'  # MultivariateStats expects p×n
        y_train = enc.codes[train_idx]

        n_classes = length(unique(y_train))
        outdim = min(n_classes - 1, size(X_train, 1))
        outdim < 1 && (outdim = 1)

        try
            model = fit(MulticlassLDA, X_train, y_train; outdim=outdim)
            X_test = X_test_use'
            proj_train = predict(model, X_train)
            proj_test = predict(model, X_test)

            # 1-NN in projected space
            best_dist = Inf
            best_label = 1
            for j in 1:size(proj_train, 2)
                d = sum((proj_train[:, j] .- proj_test[:, 1]).^2)
                if d < best_dist
                    best_dist = d
                    best_label = y_train[j]
                end
            end
            predictions[i] = enc.classes[best_label]
        catch
            # Fallback: majority vote
            predictions[i] = _majority_label(labels[train_idx])
        end
    end

    accuracy = mean(predictions .== labels)
    (accuracy=accuracy, predictions=predictions)
end

"""
    loocv_decision_tree(X::Matrix, y::Vector{String}; max_depth, min_samples_leaf, min_samples_split)

Leave-one-out cross-validation with a Decision Tree.
"""
function loocv_decision_tree(X::Matrix, y::Vector{String};
                             max_depth::Int = 6,
                             min_samples_leaf::Int = 2,
                             min_samples_split::Int = 2,
                             rng_seed::Int = 20260223)
    Xclean = sanitize_feature_matrix(X)
    n = size(Xclean, 1)
    predictions = Vector{String}(undef, n)

    for i in 1:n
        train_idx = setdiff(1:n, i)
        X_train = Xclean[train_idx, :]
        y_train = y[train_idx]

        tree = build_tree(
            y_train,
            X_train,
            size(X_train, 2),
            max_depth,
            min_samples_leaf,
            min_samples_split,
            0.0;
            loss = DecisionTree.util.gini,
            rng = MersenneTwister(rng_seed + i),
            impurity_importance = true
        )

        predictions[i] = apply_tree(tree, Xclean[i, :])
    end

    (accuracy = mean(predictions .== y), predictions = predictions)
end

# =============================================================================
# Feature Engineering
# =============================================================================

"""
    build_feature_matrix(; matrices...)

Concatenate multiple feature representations into a single feature matrix.
"""
function build_feature_matrix(; kwargs...)
    parts = Matrix[]
    for (_, v) in kwargs
        if v isa Matrix
            push!(parts, v)
        elseif v isa Vector
            mat = hcat([vec(x) for x in v]...)' |> Matrix
            push!(parts, mat)
        end
    end
    isempty(parts) ? error("No feature matrices provided") : hcat(parts...)
end

# =============================================================================
# Permutation test
# =============================================================================

"""
    permutation_test_lda(X, labels; n_permutations=1000)

Permutation test for LDA: shuffles labels and recomputes LOOCV accuracy.
"""
function permutation_test_lda(X::Matrix, labels::Vector{String};
                              n_permutations=1000)
    observed = loocv_lda(X, labels).accuracy

    perm_accuracies = Vector{Float64}(undef, n_permutations)
    for p in 1:n_permutations
        y_perm = shuffle(labels)
        perm_accuracies[p] = loocv_lda(X, y_perm).accuracy
    end

    p_value = mean(perm_accuracies .>= observed)
    (observed=observed, p_value=p_value,
     perm_mean=mean(perm_accuracies), perm_std=std(perm_accuracies),
     perm_max=maximum(perm_accuracies))
end
