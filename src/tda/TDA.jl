module TDA

include("functions.jl");
export persistence_diagram,
    plot_barcode,
    plot_pd,
    modify_array,
    dist_to_line,
    dist_to_point,
    cubical_pd,
    rips_pd,
    # TDA statistics
    count_intervals,
    max_persistence,
    total_persistence,
    median_persistence,
    persistence_entropy,
    pd_statistics

end