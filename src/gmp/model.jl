abstract type AbstractDiscretizedGaussMarkovProcess <: AbstractGaussMarkovChain end

# TODO: Document that `ts` must be indexed by 1:length(gmp)
function ts(gmp::Tgmp) where {Tgmp<:AbstractDiscretizedGaussMarkovProcess} end

function prior_mean(
    dgmp::Tdgmp,
    t::Tt,
) where {Tdgmp<:AbstractDiscretizedGaussMarkovProcess,Tt<:AbstractFloat} end

function prior_cov(
    dgmp::Tdgmp,
    t::Tt,
) where {Tdgmp<:AbstractDiscretizedGaussMarkovProcess,Tt<:AbstractFloat} end

function lsqrt_Σ(
    dgmp::Tdgmp,
    t::Tt,
) where {Tdgmp<:AbstractDiscretizedGaussMarkovProcess,Tt<:AbstractFloat}
    return sqrt(prior_cov(dgmp, t))
end

function A(
    dgmp::Tdgmp,
    t::Tt,
    s::Tt,
) where {Tdgmp<:AbstractDiscretizedGaussMarkovProcess,Tt<:AbstractFloat} end

function A_b_lsqrt_Q(
    dgmp::Tdgmp,
    t::Tt,
    s::Tt,
) where {Tdgmp<:AbstractDiscretizedGaussMarkovProcess,Tt<:AbstractFloat} end

function Base.rand(
    rng::Trng,
    dgmp::Tdgmp,
    t::Tt,
) where {
    Trng<:Random.AbstractRNG,
    Tdgmp<:AbstractDiscretizedGaussMarkovProcess,
    Tt<:AbstractFloat,
}
    lsqrt_Σₜ = lsqrt_Σ(dgmp, t)

    return prior_mean(dgmp, t) + lsqrt_Σₜ * randn(rng, eltype(lsqrt_Σₜ), size(lsqrt_Σₜ, 2))
end

function Base.rand(
    rng::Trng,
    dgmp::Tdgmp,
    t::Tt,
    s::Tt,
    xₛ::Txₛ,
) where {
    Trng<:Random.AbstractRNG,
    Tdgmp<:AbstractDiscretizedGaussMarkovProcess,
    Tt<:AbstractFloat,
    Txₛ<:AbstractVector,
}
    Aₜₛ, bₜₛ, lsqrt_Qₜₛ = A_b_lsqrt_Q(dgmp, t, s)

    return Aₜₛ * xₛ + bₜₛ + lsqrt_Qₜₛ * randn(rng, eltype(lsqrt_Qₜₛ), size(lsqrt_Qₜₛ, 2))
end

# AbstractDiscretizedGaussMarkovProcess implementation of AbstractGaussMarkovChain interface
function Base.length(dgmp::Tdgmp) where {Tdgmp<:AbstractDiscretizedGaussMarkovProcess}
    return length(ts(dgmp))
end

function prior_mean(
    dgmp::Tdgmp,
    k::Tk,
) where {Tdgmp<:AbstractDiscretizedGaussMarkovProcess,Tk<:Integer}
    return prior_mean(dgmp, ts(dgmp)[max(k, 1)])
end

function prior_cov(
    dgmp::Tdgmp,
    k::Tk,
) where {Tdgmp<:AbstractDiscretizedGaussMarkovProcess,Tk<:Integer}
    return prior_cov(dgmp, ts(dgmp)[max(k, 1)])
end

function lsqrt_Σ(
    dgmp::Tdgmp,
    k::Tk,
) where {Tdgmp<:AbstractDiscretizedGaussMarkovProcess,Tk<:Integer}
    return lsqrt_Σ(dgmp, ts(dgmp)[max(k, 1)])
end

function A(
    dgmp::Tdgmp,
    k::Tk,
) where {Tdgmp<:AbstractDiscretizedGaussMarkovProcess,Tk<:Integer}
    return A(dgmp, ts(dgmp)[k+1], k > 0 ? ts(dgmp)[k] : ts(dgmp)[k+1])
end

function A_b_lsqrt_Q(
    dgmp::Tdgmp,
    k::Tk,
) where {Tdgmp<:AbstractDiscretizedGaussMarkovProcess,Tk<:Integer}
    return A_b_lsqrt_Q(dgmp, ts(dgmp)[k+1], k > 0 ? ts(dgmp)[k] : ts(dgmp)[k+1])
end
