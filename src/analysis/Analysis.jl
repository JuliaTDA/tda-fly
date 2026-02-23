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
    # Distance combination
    normalize_by_max,
    combined_distance_grid_search,
    # Normalization
    minmax_normalize,
    zscore_normalize,
    # Classification - distance based
    knn_predict,
    loocv_knn,
    knn_predict_weighted,
    loocv_knn_weighted,
    nearest_centroid,
    loocv_nearest_centroid,
    # Classification - ML
    encode_labels,
    loocv_lda,
    loocv_svm,
    loocv_random_forest,
    loocv_random_forest_balanced,
    loocv_svm_distance,
    # Ensemble
    ensemble_vote,
    # Feature engineering
    build_feature_matrix,
    # PCA + SVM
    loocv_svm_pca,
    # Nested LOOCV
    nested_loocv,
    nested_loocv_random_forest,
    nested_loocv_svm,
    # Statistical validation
    wilson_ci,
    confusion_matrix,
    permutation_test,
    permutation_test_svm,
    classification_metrics,
    classification_report

end
