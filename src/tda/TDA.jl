module TDA

include("functions.jl");
export # Persistence computations
    rips_pd_1d,
    cubical_pd,
    cubical_pd_1d,
    directional_pd_1d,
    radial_pd_1d,
    height_filtration,
    radial_filtration,
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
    pd_statistics

end
