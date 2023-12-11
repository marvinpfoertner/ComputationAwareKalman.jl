module ComputationAwareKalman

using Distributions
using LinearAlgebra
using Statistics

include("model.jl")

include("state_covariance.jl")
include("filter.jl")
include("smoother.jl")

include("interpolate.jl")

end
