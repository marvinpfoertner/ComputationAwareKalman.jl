module ComputationAwareKalman

using Distributions
using JLD2
using Kronecker
using LinearAlgebra
using Printf
using Random
using Statistics

include("model/dynamics.jl")
include("model/measurement.jl")

include("low_rank.jl")
include("conditional_gaussian.jl")
include("truncate.jl")

include("filter/predict.jl")
include("filter/policy.jl")
include("filter/update.jl")
include("filter/cache.jl")
include("filter/loop.jl")

include("smoother/cache.jl")
include("smoother/loop.jl")

include("sampling.jl")

# Continuous-time Gauss-Markov Processes
include("gmp/model.jl")
include("gmp/discretize.jl")
include("gmp/interpolate.jl")
include("gmp/sampling.jl")

include("gmp/matern.jl")

# Space-Time Separable Gauss-Markov Processes
include("stsgmp/model.jl")
include("stsgmp/discretize.jl")

end
