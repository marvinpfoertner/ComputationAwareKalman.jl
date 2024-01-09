struct SpatiallyDiscretizedSTSGMP{
    T<:AbstractFloat,
    Ttgmp<:ComputationAwareKalman.AbstractGaussMarkovProcess,
    Tmₓ<:AbstractVector{T},
    TΣₓₓ<:AbstractMatrix{T},
    Tlsqrt_Σₓₓ<:AbstractMatrix{T},
} <: ComputationAwareKalman.AbstractGaussMarkovProcess
    tgmp::Ttgmp

    μₓ::Tmₓ
    Σₓₓ::TΣₓₓ

    lsqrt_Σₓₓ::Tlsqrt_Σₓₓ
end

function ComputationAwareKalman.μ(stgmp::SpatiallyDiscretizedSTSGMP, t::AbstractFloat)
    return kron(ComputationAwareKalman.μ(stgmp.tgmp, t), stgmp.μₓ)
end

function ComputationAwareKalman.Σ(stgmp::SpatiallyDiscretizedSTSGMP, t::AbstractFloat)
    return kronecker(ComputationAwareKalman.Σ(stgmp.tgmp, t), stgmp.Σₓₓ)
end

function ComputationAwareKalman.lsqrt_Σ(stgmp::SpatiallyDiscretizedSTSGMP, t::AbstractFloat)
    return kronecker(ComputationAwareKalman.lsqrt_Σ(stgmp.tgmp, t), stgmp.lsqrt_Σₓₓ)
end

function ComputationAwareKalman.A(
    stgmp::SpatiallyDiscretizedSTSGMP,
    t::AbstractFloat,
    s::AbstractFloat,
)
    return kronecker(ComputationAwareKalman.A(stgmp.tgmp, t, s), I(size(stgmp.Σₓₓ, 1)))
end

function ComputationAwareKalman.A_b_lsqrt_Q(
    stgmp::SpatiallyDiscretizedSTSGMP,
    t::AbstractFloat,
    s::AbstractFloat,
)
    Ãₜₛ, b̃ₜₛ, lsqrt_Q̃ₜₛ = ComputationAwareKalman.A_b_lsqrt_Q(stgmp.tgmp, t, s)

    return (
        kronecker(Ãₜₛ, I(size(stgmp.Σₓₓ, 1))),
        kron(b̃ₜₛ, stgmp.μₓ),
        kronecker(lsqrt_Q̃ₜₛ, stgmp.lsqrt_Σₓₓ),
    )
end
