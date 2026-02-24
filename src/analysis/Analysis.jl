module Analysis

using ..TDAfly

include("functions.jl");
export pairwise_distance,
    plot_wing_with_pd,
    plot_heatmap,
    sanitize_feature_matrix,
    sanitize_distance_matrix,
    # Distance matrices
    bottleneck_distance_matrix,
    wasserstein_distance_matrix,
    # Normalization
    minmax_normalize,
    zscore_normalize,
    # Classification - ML
    encode_labels,
    loocv_lda,
    loocv_decision_tree,
    # Feature engineering
    build_feature_matrix,
    # Statistical validation
    wilson_ci,
    confusion_matrix,
    classification_metrics,
    permutation_test_lda

end
