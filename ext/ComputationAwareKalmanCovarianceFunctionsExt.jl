module ComputationAwareKalmanCovarianceFunctionsExt

using ComputationAwareKalman
using CovarianceFunctions

function ComputationAwareKalman.covariance_matrix(
    covariance_fn::CovarianceFunctions.AbstractKernel,
    X,
)
    return CovarianceFunctions.Gramian(covariance_fn, X)
end

end  # module