module ComputationAwareKalman

using Distributions
using LinearAlgebra
using Random
using Statistics

include("model/dynamics.jl")
include("model/measurement.jl")

include("low_rank.jl")
include("state_covariance.jl")
include("truncate.jl")

include("filter/predict.jl")
include("filter/update.jl")
include("filter/cache.jl")
include("filter/loop.jl")

include("smoother.jl")

include("sampling.jl")

# Continuous-time Gauss-Markov Processes
include("gmp/model.jl")
include("gmp/discretize.jl")
include("gmp/interpolate.jl")
include("gmp/sampling.jl")

end
