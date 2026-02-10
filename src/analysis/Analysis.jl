module Analysis

using ..TDAfly

include("functions.jl");
export pairwise_distance,
    plot_wing_with_pd,
    plot_heatmap,
    # Distance matrices
    bottleneck_distance_matrix,
    wasserstein_distance_matrix,
    # Normalization
    minmax_normalize,
    zscore_normalize,
    # Classification
    knn_predict,
    loocv_knn,
    # Statistical validation
    wilson_ci,
    confusion_matrix,
    permutation_test,
    classification_report

end