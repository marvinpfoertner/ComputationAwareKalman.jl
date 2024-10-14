abstract type AbstractGaussMarkovProcess end

function μ(gmp::Tgmp, t::Tt) where {Tgmp<:AbstractGaussMarkovProcess,Tt<:Real} end

function Σ(gmp::Tgmp, t::Tt) where {Tgmp<:AbstractGaussMarkovProcess,Tt<:Real} end

function lsqrt_Σ(gmp::Tgmp, t::Tt) where {Tgmp<:AbstractGaussMarkovProcess,Tt<:Real}
    return sqrt(Σ(gmp, t))
end

function A(gmp::Tgmp, t::Tt, s::Tt) where {Tgmp<:AbstractGaussMarkovProcess,Tt<:Real} end

function A_b_lsqrt_Q(
    gmp::Tgmp,
    t::Tt,
    s::Tt,
) where {Tgmp<:AbstractGaussMarkovProcess,Tt<:Real} end

function Base.rand(
    rng::Trng,
    gmp::Tgmp,
    t::Tt,
) where {Trng<:Random.AbstractRNG,Tgmp<:AbstractGaussMarkovProcess,Tt<:Real}
    lsqrt_Σₜ = lsqrt_Σ(gmp, t)

    return μ(gmp, t) + lsqrt_Σₜ * randn(rng, eltype(lsqrt_Σₜ), size(lsqrt_Σₜ, 2))
end

function Base.rand(
    rng::Trng,
    gmp::Tgmp,
    t::Tt,
    s::Tt,
    xₛ::Txₛ,
) where {
    Trng<:Random.AbstractRNG,
    Tgmp<:AbstractGaussMarkovProcess,
    Tt<:Real,
    Txₛ<:AbstractVector,
}
    Aₜₛ, bₜₛ, lsqrt_Qₜₛ = A_b_lsqrt_Q(gmp, t, s)

    return Aₜₛ * xₛ + bₜₛ + lsqrt_Qₜₛ * randn(rng, eltype(lsqrt_Qₜₛ), size(lsqrt_Qₜₛ, 2))
end
