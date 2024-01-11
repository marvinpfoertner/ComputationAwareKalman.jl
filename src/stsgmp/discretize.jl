function discretize(stsgmp::SpaceTimeSeparableGaussMarkovProcess, xs)
    spatial_cov_mat = covariance_matrix(stsgmp.spatial_cov_fn, xs)

    return SpatiallyDiscretizedSTSGMP(
        stsgmp,
        xs,
        mean_vec(stsgmp.spatial_mean_fn, xs),
        spatial_cov_mat,
        sqrt(spatial_cov_mat),
    )
end

function discretize(stsgmp::SpaceTimeSeparableGaussMarkovProcess, ts, xs)
    return discretize(discretize(stsgmp, xs), ts)
end

function mean_vec(mean_fn, xs::AbstractVector)
    return [mean_fn(x) for x in xs]
end

struct SpatiallyDiscretizedSTSGMP{
    T<:AbstractFloat,
    Tstsgmp<:SpaceTimeSeparableGaussMarkovProcess,
    TX<:AbstractVector,
    Tμₓ<:AbstractVector{T},
    TΣₓₓ<:AbstractMatrix{T},
    Tlsqrt_Σₓₓ<:AbstractMatrix{T},
} <: AbstractGaussMarkovProcess
    stsgmp::Tstsgmp

    X::TX

    spatial_mean::Tμₓ
    spatial_cov_mat::TΣₓₓ

    lsqrt_spatial_cov_mat::Tlsqrt_Σₓₓ
end

function μ(sdstsgmp::SpatiallyDiscretizedSTSGMP, t::AbstractFloat)
    return kron(μ(sdstsgmp.stsgmp.tgmp, t), sdstsgmp.spatial_mean)
end

function Σ(sdstsgmp::SpatiallyDiscretizedSTSGMP, t::AbstractFloat)
    return kronecker(Σ(sdstsgmp.stsgmp.tgmp, t), sdstsgmp.spatial_cov_mat)
end

function lsqrt_Σ(sdstsgmp::SpatiallyDiscretizedSTSGMP, t::AbstractFloat)
    return kronecker(lsqrt_Σ(sdstsgmp.stsgmp.tgmp, t), sdstsgmp.lsqrt_spatial_cov_mat)
end

function A(sdstsgmp::SpatiallyDiscretizedSTSGMP, t::AbstractFloat, s::AbstractFloat)
    return kronecker(A(sdstsgmp.stsgmp.tgmp, t, s), I(size(sdstsgmp.spatial_cov_mat, 1)))
end

function A_b_lsqrt_Q(
    sdstsgmp::SpatiallyDiscretizedSTSGMP,
    t::AbstractFloat,
    s::AbstractFloat,
)
    Ãₜₛ, b̃ₜₛ, lsqrt_Q̃ₜₛ = A_b_lsqrt_Q(sdstsgmp.stsgmp.tgmp, t, s)

    return (
        kronecker(Ãₜₛ, I(size(sdstsgmp.spatial_cov_mat, 1))),
        kron(b̃ₜₛ, sdstsgmp.spatial_mean),
        kronecker(lsqrt_Q̃ₜₛ, sdstsgmp.lsqrt_spatial_cov_mat),
    )
end
