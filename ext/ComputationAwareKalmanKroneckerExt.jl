module ComputationAwareKalmanKroneckerExt

using ComputationAwareKalman
using Kronecker

function Base.:*(
    M::ComputationAwareKalman.LowRankDowndatedMatrix,
    X::GeneralizedKroneckerProduct,
)
    return M.A * X - M.U * (M.V' * X)
end

function Base.:*(
    X::GeneralizedKroneckerProduct,
    M::ComputationAwareKalman.LowRankDowndatedMatrix,
)
    return X * M.A - (X * M.U) * M.V'
end

end # module