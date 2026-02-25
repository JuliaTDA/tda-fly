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
    total_persistence,
    median_persistence,
    persistence_entropy,
    pd_statistics,
    pd_statistics_extended,
    skewness_persistence,
    kurtosis_persistence,
    median_death,
    median_birth,
    std_birth,
    std_death,
    mean_midlife,
    persistence_range,
    count_significant

end
