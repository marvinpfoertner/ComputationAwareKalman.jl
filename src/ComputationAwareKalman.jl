module ComputationAwareKalman

using LinearAlgebra

include("state_covariance.jl")
include("filter.jl")
include("smoother.jl")

export StateCovariance
export FilterCache
export SmootherCache

end
