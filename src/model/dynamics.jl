abstract type AbstractGaussMarkovChain end

# TODO: Document
# function Base.length(gmc::Tgmc) where {Tgmc<:AbstractGaussMarkovChain} end

function prior_mean(gmc::Tgmc, k::Tk) where {Tgmc<:AbstractGaussMarkovChain,Tk<:Integer} end

function prior_cov(gmc::Tgmc, k::Tk) where {Tgmc<:AbstractGaussMarkovChain,Tk<:Integer} end

function lsqrt_Σ(gmc::Tgmc, k::Tk) where {Tgmc<:AbstractGaussMarkovChain,Tk<:Integer}
    return sqrt(prior_cov(gmc, k))
end

# TODO: Document that this must return the identity for k = 0
function A(gmc::Tgmc, k::Tk) where {Tgmc<:AbstractGaussMarkovChain,Tk<:Integer} end

# TODO: Document that this will only be called with k >= 1
function A_b_lsqrt_Q(gmc::Tgmc, k::Tk) where {Tgmc<:AbstractGaussMarkovChain,Tk<:Integer} end

function Base.rand(
    rng::Trng,
    gmc::Tgmc,
    k::Tk,
) where {Trng<:Random.AbstractRNG,Tgmc<:AbstractGaussMarkovChain,Tk<:Integer}
    lsqrt_Σₖ = lsqrt_Σ(gmc, k)

    return prior_mean(gmc, k) + lsqrt_Σₖ * randn(rng, eltype(lsqrt_Σₖ), size(lsqrt_Σₖ, 2))
end

# TODO: Document that this must return xₖ for k = 0
function Base.rand(
    rng::Trng,
    gmc::Tgmc,
    k::Tk,
    xₖ::Txₖ,
) where {
    Trng<:Random.AbstractRNG,
    Tgmc<:AbstractGaussMarkovChain,
    Tk<:Integer,
    Txₖ<:AbstractVector,
}
    if k == 0
        return xₖ
    end

    Aₖ, bₖ, lsqrt_Qₖ = A_b_lsqrt_Q(gmc, k)

    return Aₖ * xₖ + bₖ + lsqrt_Qₖ * randn(rng, eltype(lsqrt_Qₖ), size(lsqrt_Qₖ, 2))
end
