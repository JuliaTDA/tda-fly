# import CairoMakie as CM
using Distances
using Plots
using Random: shuffle, shuffle!, rand, MersenneTwister
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
    Xf = sanitize_feature_matrix(X)
    mins = minimum(Xf, dims=1)
    maxs = maximum(Xf, dims=1)
    ranges = maxs .- mins
    ranges[ranges .== 0] .= 1  # Avoid division by zero
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
    sigma[sigma .== 0] .= 1  # Avoid division by zero
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

function _tie_break(neighbors)
    labels = [x[2] for x in neighbors]
    counts = countmap(labels)
    max_count = maximum(values(counts))
    tied = [lbl for (lbl, cnt) in counts if cnt == max_count]
    length(tied) == 1 && return tied[1]

    avg_d = Dict(lbl => mean([d for (d, y) in neighbors if y == lbl]) for lbl in tied)
    best = minimum(values(avg_d))
    winners = sort([lbl for (lbl, d) in avg_d if d == best])
    winners[1]
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
    dists = [(D[test_idx, j], y[j]) for j in train_idx if isfinite(D[test_idx, j])]
    isempty(dists) && return _majority_label(y[train_idx])
    sort!(dists, by=first)

    # k nearest neighbors - majority vote
    neighbors = dists[1:min(k, length(dists))]
    _tie_break(neighbors)
end

"""
    loocv_knn(D::Matrix, y::Vector; k=1)

Perform leave-one-out cross-validation with k-NN using a precomputed distance matrix.
Returns (accuracy, predictions).
"""
function loocv_knn(D::Matrix, y::Vector; k=1)
    Dclean = sanitize_distance_matrix(D)
    n = length(y)
    predictions = [knn_predict(Dclean, y, i; k=k) for i in 1:n]
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
# =============================================================================
# Distance Matrix Combination
# =============================================================================

"""
    normalize_by_max(D)

Normalize a distance matrix to [0, 1] by dividing by its maximum value.
"""
function normalize_by_max(D)
    Dclean = sanitize_distance_matrix(D)
    maxd = maximum(Dclean)
    maxd > 0 ? Dclean ./ maxd : Dclean
end

"""
    combined_distance_grid_search(D1, D2, labels; alphas=0.0:0.1:1.0, ks=[1, 3, 5], normalize_fn=normalize_by_max)

Grid search over convex combinations of two distance matrices.
D_combined(α) = α * normalize(D1) + (1-α) * normalize(D2)

Returns a sorted vector of (alpha, k, accuracy, n_correct) named tuples (best first).
"""
function combined_distance_grid_search(D1, D2, labels;
                                        alphas=0.0:0.1:1.0,
                                        ks=[1, 3, 5],
                                        normalize_fn=normalize_by_max)
    D1n = normalize_fn(D1)
    D2n = normalize_fn(D2)
    results = NamedTuple[]
    for alpha in alphas
        D_combined = alpha .* D1n .+ (1 - alpha) .* D2n
        for k in ks
            result = loocv_knn(D_combined, labels; k=k)
            n_correct = sum(result.predictions .== labels)
            push!(results, (alpha=alpha, k=k, accuracy=result.accuracy, n_correct=n_correct))
        end
    end
    sort(results, by=x -> x.accuracy, rev=true)
end

# =============================================================================
# Alternative Classifiers
# =============================================================================

"""
    knn_predict_weighted(D::Matrix, y::Vector, test_idx::Int; k=3)

Predict using weighted k-NN: each neighbor's vote is weighted by 1/distance.
"""
function knn_predict_weighted(D::Matrix, y::Vector, test_idx::Int; k=3)
    n = size(D, 1)
    train_idx = setdiff(1:n, test_idx)

    dists = [(D[test_idx, j], y[j]) for j in train_idx if isfinite(D[test_idx, j])]
    isempty(dists) && return _majority_label(y[train_idx])
    sort!(dists, by=first)

    neighbors = dists[1:min(k, length(dists))]

    eps = 1e-10
    class_weights = Dict{eltype(y), Float64}()
    for (d, label) in neighbors
        w = 1.0 / (d + eps)
        class_weights[label] = get(class_weights, label, 0.0) + w
    end

    argmax(class_weights)
end

"""
    loocv_knn_weighted(D::Matrix, y::Vector; k=3)

Leave-one-out cross-validation with weighted k-NN.
"""
function loocv_knn_weighted(D::Matrix, y::Vector; k=3)
    Dclean = sanitize_distance_matrix(D)
    n = length(y)
    predictions = [knn_predict_weighted(Dclean, y, i; k=k) for i in 1:n]
    accuracy = mean(predictions .== y)
    (accuracy=accuracy, predictions=predictions)
end

"""
    nearest_centroid(D::Matrix, y::Vector, test_idx::Int)

Classify test point by average distance to each class (nearest centroid / medoid).
"""
function nearest_centroid(D::Matrix, y::Vector, test_idx::Int)
    classes = unique(y)
    train_idx = setdiff(1:length(y), test_idx)

    best_class = classes[1]
    best_dist = Inf

    for cls in classes
        cls_idx = [i for i in train_idx if y[i] == cls]
        avg_dist = mean(D[test_idx, cls_idx])
        if avg_dist < best_dist
            best_dist = avg_dist
            best_class = cls
        end
    end
    best_class
end

"""
    loocv_nearest_centroid(D::Matrix, y::Vector)

Leave-one-out cross-validation with nearest centroid classifier.
"""
function loocv_nearest_centroid(D::Matrix, y::Vector)
    Dclean = sanitize_distance_matrix(D)
    n = length(y)
    predictions = [nearest_centroid(Dclean, y, i) for i in 1:n]
    accuracy = mean(predictions .== y)
    (accuracy=accuracy, predictions=predictions)
end

# =============================================================================
# Nested LOOCV for honest evaluation
# =============================================================================

"""
    nested_loocv(D1, D2, labels; alphas=0.0:0.1:1.0, ks=[1, 3, 5], normalize_fn=normalize_by_max)

Nested leave-one-out cross-validation for combined distance matrices.
Outer loop: leave one sample out. Inner loop: select best (alpha, k) via inner LOOCV.
Returns unbiased accuracy estimate despite hyperparameter tuning.
"""
function nested_loocv(D1, D2, labels;
                       alphas=0.0:0.1:1.0,
                       ks=[1, 3, 5],
                       normalize_fn=normalize_by_max)
    n = length(labels)
    D1n = normalize_fn(D1)
    D2n = normalize_fn(D2)

    predictions = Vector{eltype(labels)}(undef, n)
    selected_params = Vector{NamedTuple}(undef, n)

    for i in 1:n
        inner_idx = setdiff(1:n, i)

        best_acc = -1.0
        best_alpha = 0.5
        best_k = 3

        for alpha in alphas
            D_combined = alpha .* D1n .+ (1 - alpha) .* D2n
            D_inner = D_combined[inner_idx, inner_idx]
            y_inner = labels[inner_idx]

            for k in ks
                inner_result = loocv_knn(D_inner, y_inner; k=k)
                if inner_result.accuracy > best_acc
                    best_acc = inner_result.accuracy
                    best_alpha = alpha
                    best_k = k
                end
            end
        end

        D_best = best_alpha .* D1n .+ (1 - best_alpha) .* D2n
        predictions[i] = knn_predict(D_best, labels, i; k=best_k)
        selected_params[i] = (alpha=best_alpha, k=best_k, inner_acc=best_acc)
    end

    accuracy = mean(predictions .== labels)
    (accuracy=accuracy, predictions=predictions, params=selected_params)
end

# =============================================================================
# Comprehensive Classification Report
# =============================================================================

# =============================================================================
# ML Classifiers: LDA, SVM, Random Forest
# =============================================================================

using MultivariateStats: fit, predict, MulticlassLDA, projection, PCA
using DecisionTree: build_forest, apply_forest, nfoldCV_forest
using LIBSVM

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
    loocv_svm(X::Matrix, labels::Vector{String}; kernel=LIBSVM.Kernel.RadialBasis, cost=1.0)

Leave-one-out cross-validation with SVM.
X is n×p (samples × features).
"""
function loocv_svm(X::Matrix, labels::Vector{String};
                   kernel=LIBSVM.Kernel.RadialBasis, cost=1.0, gamma=nothing,
                   standardize=true)
    Xclean = sanitize_feature_matrix(X)
    n = size(X, 1)
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

        X_train = X_train_use'  # LIBSVM expects p×n
        y_train = labels[train_idx]
        X_test = X_test_use'

        try
            model = isnothing(gamma) ?
                svmtrain(X_train, y_train; kernel=kernel, cost=cost) :
                svmtrain(X_train, y_train; kernel=kernel, cost=cost, gamma=gamma)
            pred, _ = svmpredict(model, X_test)
            predictions[i] = pred[1]
        catch
            predictions[i] = _majority_label(y_train)
        end
    end

    accuracy = mean(predictions .== labels)
    (accuracy=accuracy, predictions=predictions)
end

"""
    loocv_random_forest(X::Matrix, labels::Vector{String}; n_trees=100, max_depth=-1, min_samples_leaf=1)

Leave-one-out cross-validation with Random Forest.
X is n×p (samples × features).
"""
function loocv_random_forest(X::Matrix, labels::Vector{String};
                              n_trees=100, max_depth=-1, min_samples_leaf=1,
                              rng_seed::Union{Nothing, Int}=nothing)
    Xclean = sanitize_feature_matrix(X)
    n = size(Xclean, 1)
    predictions = Vector{String}(undef, n)

    for i in 1:n
        train_idx = setdiff(1:n, i)
        X_train = Xclean[train_idx, :]
        y_train = labels[train_idx]

        n_features = round(Int, sqrt(size(X, 2)))
        n_features = max(1, n_features)
        model = isnothing(rng_seed) ?
            build_forest(y_train, X_train, n_features, n_trees, 0.7, max_depth, min_samples_leaf) :
            build_forest(y_train, X_train, n_features, n_trees, 0.7, max_depth, min_samples_leaf;
                         rng=MersenneTwister(rng_seed + i))
        predictions[i] = apply_forest(model, Xclean[i, :])
    end

    accuracy = mean(predictions .== labels)
    (accuracy=accuracy, predictions=predictions)
end

function _balanced_bootstrap_indices(y::Vector{String}; rng::MersenneTwister, target_balance=:max)
    classes = sort(unique(y))
    counts = [count(==(c), y) for c in classes]
    target_n = target_balance == :median ?
        max(1, round(Int, median(counts))) :
        maximum(counts)

    idx = Int[]
    for cls in classes
        cls_idx = findall(==(cls), y)
        append!(idx, rand(rng, cls_idx, target_n))
    end
    shuffle!(rng, idx)
    idx
end

"""
    loocv_random_forest_balanced(X, labels; ...)

LOOCV random forest with class-balanced bootstrap resampling inside each fold.
Useful for imbalanced multiclass data.
"""
function loocv_random_forest_balanced(X::Matrix, labels::Vector{String};
                                      n_trees=300, max_depth=-1, min_samples_leaf=1,
                                      target_balance=:max, rng_seed=42)
    Xclean = sanitize_feature_matrix(X)
    n = size(Xclean, 1)
    predictions = Vector{String}(undef, n)

    for i in 1:n
        train_idx = setdiff(1:n, i)
        X_train = Xclean[train_idx, :]
        y_train = labels[train_idx]

        rng = MersenneTwister(rng_seed + i)
        bal_idx = _balanced_bootstrap_indices(y_train; rng=rng, target_balance=target_balance)
        X_bal = X_train[bal_idx, :]
        y_bal = y_train[bal_idx]

        n_features = max(1, round(Int, sqrt(size(X_bal, 2))))
        model = build_forest(y_bal, X_bal, n_features, n_trees, 0.7, max_depth, min_samples_leaf; rng=rng)
        predictions[i] = apply_forest(model, Xclean[i, :])
    end

    accuracy = mean(predictions .== labels)
    (accuracy=accuracy, predictions=predictions)
end

"""
    loocv_svm_distance(D::Matrix, labels::Vector{String}; cost=1.0)

Leave-one-out cross-validation with SVM using precomputed distance matrix.
Converts distance matrix to RBF-like kernel: K(i,j) = exp(-D(i,j)^2 / (2*median(D)^2)).
"""
function loocv_svm_distance(D::Matrix, labels::Vector{String}; cost=1.0)
    # Convert distance to kernel
    Dclean = sanitize_distance_matrix(D)
    positive = Dclean[Dclean .> 0]
    med = isempty(positive) ? 1.0 : median(positive)
    gamma = 1.0 / (2 * med^2)
    K = exp.(-gamma .* Dclean.^2)

    n = size(Dclean, 1)
    predictions = Vector{String}(undef, n)

    for i in 1:n
        train_idx = setdiff(1:n, i)
        # Use kernel values as features (kernel trick approximation)
        X_train = K[train_idx, train_idx]'
        y_train = labels[train_idx]
        X_test = reshape(K[i, train_idx], :, 1)

        try
            model = svmtrain(X_train, y_train; kernel=LIBSVM.Kernel.Linear, cost=cost)
            pred, _ = svmpredict(model, X_test)
            predictions[i] = pred[1]
        catch
            predictions[i] = _majority_label(labels[train_idx])
        end
    end

    accuracy = mean(predictions .== labels)
    (accuracy=accuracy, predictions=predictions)
end

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
# Honest evaluation for RF (nested CV)
# =============================================================================

"""
    nested_loocv_random_forest(X, labels; ...)

Outer LOOCV for unbiased performance, inner k-fold CV for hyperparameter selection.
"""
function nested_loocv_random_forest(X::Matrix, labels::Vector{String};
                                    n_trees_grid=[200, 500],
                                    max_depth_grid=[-1],
                                    min_samples_leaf_grid=[1, 2],
                                    inner_folds=5,
                                    balanced=true,
                                    rng_seed=42)
    Xclean = sanitize_feature_matrix(X)
    n = size(Xclean, 1)
    predictions = Vector{String}(undef, n)
    selected_params = Vector{NamedTuple}(undef, n)

    for i in 1:n
        train_idx = setdiff(1:n, i)
        X_train = Xclean[train_idx, :]
        y_train = labels[train_idx]

        best_acc = -Inf
        best_n_trees = n_trees_grid[1]
        best_max_depth = max_depth_grid[1]
        best_min_leaf = min_samples_leaf_grid[1]

        n_features_inner = max(1, round(Int, sqrt(size(X_train, 2))))
        for n_trees in n_trees_grid
            for max_depth in max_depth_grid
                for min_leaf in min_samples_leaf_grid
                    rng_inner = MersenneTwister(rng_seed + i + n_trees + max(max_depth, 0) + 10 * min_leaf)
                    fold_acc = redirect_stdout(devnull) do
                        nfoldCV_forest(
                            y_train, X_train, inner_folds, n_features_inner,
                            n_trees, 0.7, max_depth, min_leaf;
                            verbose=false, rng=rng_inner
                        )
                    end
                    mean_acc = mean(fold_acc)
                    if mean_acc > best_acc
                        best_acc = mean_acc
                        best_n_trees = n_trees
                        best_max_depth = max_depth
                        best_min_leaf = min_leaf
                    end
                end
            end
        end

        rng_outer = MersenneTwister(rng_seed + 10_000 + i)
        if balanced
            bal_idx = _balanced_bootstrap_indices(y_train; rng=rng_outer, target_balance=:max)
            X_fit = X_train[bal_idx, :]
            y_fit = y_train[bal_idx]
        else
            X_fit = X_train
            y_fit = y_train
        end

        n_features_outer = max(1, round(Int, sqrt(size(X_fit, 2))))
        model = build_forest(
            y_fit, X_fit, n_features_outer, best_n_trees, 0.7,
            best_max_depth, best_min_leaf; rng=rng_outer
        )
        predictions[i] = apply_forest(model, Xclean[i, :])
        selected_params[i] = (
            n_trees=best_n_trees,
            max_depth=best_max_depth,
            min_samples_leaf=best_min_leaf,
            inner_acc=best_acc
        )
    end

    metrics = classification_metrics(labels, predictions)
    (
        accuracy=metrics.accuracy,
        balanced_accuracy=metrics.balanced_accuracy,
        macro_f1=metrics.macro_f1,
        predictions=predictions,
        params=selected_params
    )
end

# =============================================================================
# PCA + SVM (dimensionality reduction before classification)
# =============================================================================

"""
    loocv_svm_pca(X, labels; variance_ratio=0.95, kernel=RBF, cost=1.0)

LOOCV with PCA dimensionality reduction fitted on each training fold.
Retains enough components to explain `variance_ratio` of the variance.
"""
function loocv_svm_pca(X::Matrix, labels::Vector{String};
                       variance_ratio=0.95,
                       kernel=LIBSVM.Kernel.RadialBasis, cost=1.0,
                       standardize=true)
    Xclean = sanitize_feature_matrix(X)
    n = size(Xclean, 1)
    predictions = Vector{String}(undef, n)
    n_components_used = Vector{Int}(undef, n)

    for i in 1:n
        train_idx = setdiff(1:n, i)
        X_train_raw = Xclean[train_idx, :]
        X_test_raw = Xclean[i:i, :]

        # Standardize on training fold
        if standardize
            params = _fit_zscore(X_train_raw)
            X_train_std = _apply_zscore(X_train_raw, params)
            X_test_std = _apply_zscore(X_test_raw, params)
        else
            X_train_std = X_train_raw
            X_test_std = X_test_raw
        end

        # Fit PCA on training fold (MultivariateStats expects p×n)
        pca_model = fit(PCA, X_train_std'; maxoutdim=size(X_train_std, 1) - 1, pratio=variance_ratio)
        n_components_used[i] = size(pca_model, 2)

        X_train_pca = predict(pca_model, X_train_std')'  # back to n×p
        X_test_pca = predict(pca_model, X_test_std')'

        # SVM on reduced features (LIBSVM expects p×n)
        y_train = labels[train_idx]
        try
            model = svmtrain(X_train_pca', y_train; kernel=kernel, cost=cost)
            pred, _ = svmpredict(model, X_test_pca')
            predictions[i] = pred[1]
        catch
            predictions[i] = _majority_label(y_train)
        end
    end

    accuracy = mean(predictions .== labels)
    (accuracy=accuracy, predictions=predictions,
     median_n_components=Int(median(n_components_used)))
end

# =============================================================================
# Nested LOOCV for SVM (honest evaluation)
# =============================================================================

"""
    nested_loocv_svm(X, labels; kernels, costs, use_pca, variance_ratio, inner_folds)

Nested LOOCV for SVM: outer loop holds out one sample, inner k-fold CV selects
the best (kernel, cost) combination. Optionally applies PCA inside each fold.
Returns an unbiased accuracy estimate.
"""
function nested_loocv_svm(X::Matrix, labels::Vector{String};
                          kernels=[LIBSVM.Kernel.RadialBasis, LIBSVM.Kernel.Linear],
                          costs=[0.1, 1.0, 10.0, 100.0],
                          use_pca=false,
                          variance_ratio=0.95,
                          inner_folds=5,
                          rng_seed=42)
    Xclean = sanitize_feature_matrix(X)
    n = size(Xclean, 1)
    enc = encode_labels(labels)
    predictions = Vector{String}(undef, n)
    selected_params = Vector{NamedTuple}(undef, n)

    for i in 1:n
        train_idx = setdiff(1:n, i)
        X_outer_train = Xclean[train_idx, :]
        y_outer_train = labels[train_idx]

        # Inner CV to select best hyperparameters
        best_acc = -Inf
        best_kernel = kernels[1]
        best_cost = costs[1]

        # Create inner folds (stratified-ish shuffle)
        rng = MersenneTwister(rng_seed + i)
        inner_n = length(train_idx)
        perm = shuffle(rng, 1:inner_n)
        fold_size = div(inner_n, inner_folds)

        for kernel in kernels
            for cost in costs
                fold_accs = Float64[]
                for f in 1:inner_folds
                    if f < inner_folds
                        val_idx = perm[(f-1)*fold_size+1 : f*fold_size]
                    else
                        val_idx = perm[(f-1)*fold_size+1 : end]
                    end
                    tr_idx = setdiff(1:inner_n, val_idx)

                    X_tr = X_outer_train[tr_idx, :]
                    y_tr = y_outer_train[tr_idx]
                    X_val = X_outer_train[val_idx, :]
                    y_val = y_outer_train[val_idx]

                    # Standardize
                    params = _fit_zscore(X_tr)
                    X_tr_std = _apply_zscore(X_tr, params)
                    X_val_std = _apply_zscore(X_val, params)

                    # Optional PCA
                    if use_pca
                        pca_model = fit(PCA, X_tr_std'; maxoutdim=size(X_tr_std, 1) - 1, pratio=variance_ratio)
                        X_tr_use = predict(pca_model, X_tr_std')'
                        X_val_use = predict(pca_model, X_val_std')'
                    else
                        X_tr_use = X_tr_std
                        X_val_use = X_val_std
                    end

                    try
                        model = svmtrain(X_tr_use', y_tr; kernel=kernel, cost=cost)
                        pred, _ = svmpredict(model, X_val_use')
                        push!(fold_accs, mean(pred .== y_val))
                    catch
                        push!(fold_accs, 0.0)
                    end
                end

                mean_acc = mean(fold_accs)
                if mean_acc > best_acc
                    best_acc = mean_acc
                    best_kernel = kernel
                    best_cost = cost
                end
            end
        end

        # Retrain on full outer training set with best params
        params = _fit_zscore(X_outer_train)
        X_train_std = _apply_zscore(X_outer_train, params)
        X_test_std = _apply_zscore(Xclean[i:i, :], params)

        if use_pca
            pca_model = fit(PCA, X_train_std'; maxoutdim=size(X_train_std, 1) - 1, pratio=variance_ratio)
            X_train_use = predict(pca_model, X_train_std')'
            X_test_use = predict(pca_model, X_test_std')'
        else
            X_train_use = X_train_std
            X_test_use = X_test_std
        end

        try
            model = svmtrain(X_train_use', y_outer_train; kernel=best_kernel, cost=best_cost)
            pred, _ = svmpredict(model, X_test_use')
            predictions[i] = pred[1]
        catch
            predictions[i] = _majority_label(y_outer_train)
        end

        kernel_name = best_kernel == LIBSVM.Kernel.RadialBasis ? "RBF" : "Linear"
        selected_params[i] = (kernel=kernel_name, cost=best_cost, inner_acc=best_acc)
    end

    metrics = classification_metrics(labels, predictions)
    (
        accuracy=metrics.accuracy,
        balanced_accuracy=metrics.balanced_accuracy,
        macro_f1=metrics.macro_f1,
        predictions=predictions,
        params=selected_params
    )
end

# =============================================================================
# Permutation test for feature-based classifiers
# =============================================================================

"""
    permutation_test_svm(X, labels; n_permutations=1000, kernel, cost)

Permutation test for SVM: shuffles labels and recomputes LOOCV accuracy.
Returns observed accuracy, p-value, and null distribution statistics.
"""
function permutation_test_svm(X::Matrix, labels::Vector{String};
                              n_permutations=1000,
                              kernel=LIBSVM.Kernel.RadialBasis,
                              cost=1.0)
    observed = loocv_svm(X, labels; kernel=kernel, cost=cost).accuracy

    perm_accuracies = Vector{Float64}(undef, n_permutations)
    for p in 1:n_permutations
        y_perm = shuffle(labels)
        perm_accuracies[p] = loocv_svm(X, y_perm; kernel=kernel, cost=cost).accuracy
    end

    p_value = mean(perm_accuracies .>= observed)
    (observed=observed, p_value=p_value,
     perm_mean=mean(perm_accuracies), perm_std=std(perm_accuracies),
     perm_max=maximum(perm_accuracies))
end

# =============================================================================
# Ensemble Classifier
# =============================================================================

"""
    ensemble_vote(predictions_list::Vector{Vector{String}}, weights=nothing)

Combine multiple classifier predictions via (weighted) majority voting.
"""
function ensemble_vote(predictions_list::Vector{Vector{String}};
                        weights::Union{Nothing, Vector{Float64}}=nothing)
    n = length(predictions_list[1])
    n_classifiers = length(predictions_list)
    w = isnothing(weights) ? ones(n_classifiers) : weights

    final_predictions = Vector{String}(undef, n)
    for i in 1:n
        vote_weights = Dict{String, Float64}()
        for (j, preds) in enumerate(predictions_list)
            label = preds[i]
            vote_weights[label] = get(vote_weights, label, 0.0) + w[j]
        end
        final_predictions[i] = argmax(vote_weights)
    end
    final_predictions
end

# =============================================================================
# Feature Engineering
# =============================================================================

"""
    build_feature_matrix(; matrices...)

Concatenate multiple feature representations into a single feature matrix.
Accepts keyword arguments where each value is either:
- a Matrix (n×p, used directly)
- a Vector of vectors (converted to n×p matrix)
Returns n×p matrix.
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

    metrics = classification_metrics(y, preds)

    (accuracy=acc, n_correct=n_correct, n_total=n,
     ci_95=ci, confusion_matrix=cm, classes=classes,
     sensitivity=sensitivity, specificity=specificity,
     balanced_accuracy=metrics.balanced_accuracy, macro_f1=metrics.macro_f1,
     p_value=perm.p_value, chance_level=perm.perm_mean)
end
