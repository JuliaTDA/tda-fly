module TDA

include("functions.jl");
export # Persistence computations
    rips_pd_1d,
    cubical_pd,
    cubical_pd_1d,
    directional_pd_1d,
    directional_pd_0d,
    radial_pd_1d,
    radial_pd_0d,
    height_filtration,
    radial_filtration,
    # EDT filtration
    edt_filtration,
    edt_pd_1d,
    edt_pd_0d,
    # Plotting
    plot_barcode,
    plot_pd,
    # Array manipulation
    modify_array,
    dist_to_point,
    dist_to_line,
    # TDA statistics
    count_intervals,
    max_persistence,
    min_persistence,
    mean_persistence,
    total_persistence,
    median_persistence,
    std_persistence,
    persistence_entropy,
    persistence_range,
    persistence_iqr,
    persistence_cv,
    persistence_skewness,
    persistence_kurtosis,
    second_max_persistence,
    ratio_top_to_total,
    n_significant,
    mean_birth,
    mean_death,
    std_birth,
    std_death,
    mean_midlife,
    std_midlife,
    pd_statistics,
    pd_stat_names

end
