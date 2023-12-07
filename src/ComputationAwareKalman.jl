module ComputationAwareKalman

using Distributions
using LinearAlgebra
using Statistics

include("model.jl")

include("state_covariance.jl")
include("filter.jl")
include("smoother.jl")

export AbstractDiscreteLGSSM, AbstractDiscretizedContinuousLGSSM

export StateCovariance, ConditionalGaussianBelief
export UpdatedBelief, FilterCache
export SmootherCache

end
