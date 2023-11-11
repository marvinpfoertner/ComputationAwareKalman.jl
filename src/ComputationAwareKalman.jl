module ComputationAwareKalman

using LinearAlgebra

include("state_covariance.jl")
include("filter.jl")

export StateCovariance
export FilterCache

end
