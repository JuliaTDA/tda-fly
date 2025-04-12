module TDAfly

using Revise
using Reexport
@reexport using CairoMakie: heatmap
@reexport using Chain: @chain
@reexport using ProgressMeter: @showprogress

include("common/functions.jl");
export findall_ids;

include("preprocessing/Preprocessing.jl");
export Preprocessing;

include("tda/TDA.jl");
export TDA;

include("analysis/Analysis.jl");
export Analysis;

end # module TDAfly
