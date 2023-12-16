abstract type AbstractGaussMarkovChain end
abstract type AbstractDiscretizedGaussMarkovProcess <: AbstractGaussMarkovChain end

# AbstractGaussMarkovChain interface

# function Base.length(gmc::Tgmc) where {Tgmc<:AbstractGaussMarkovChain} end

function prior_mean(gmc::Tgmc, k::Tk) where {Tgmc<:AbstractGaussMarkovChain,Tk<:Integer} end

function prior_cov(gmc::Tgmc, k::Tk) where {Tgmc<:AbstractGaussMarkovChain,Tk<:Integer} end

# TODO: Document that this should return the identity for k = 0
function A(gmc::Tgmc, k::Tk) where {Tgmc<:AbstractGaussMarkovChain,Tk<:Integer} end

# function Base.rand(
#     rng::Trng,
#     gmc::Tgmc,
#     k::Tk,
# ) where {Trng<:Random.AbstractRNG,Tgmc<:AbstractGaussMarkovChain,Tk<:Integer} end

# function Base.rand(
#     rng::Trng,
#     gmc::Tgmc,
#     k::Tk,
#     xₖ::Txₖ,
# ) where {
#     Trng<:Random.AbstractRNG,
#     Tgmc<:AbstractGaussMarkovChain,
#     Tk<:Integer,
#     Txₖ<:AbstractVector,
# } end

# AbstractDiscretizedGaussMarkovProcess interface

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

function A(
    dgmp::Tdgmp,
    t::Tt,
    t₀::Tt,
) where {Tdgmp<:AbstractDiscretizedGaussMarkovProcess,Tt<:AbstractFloat} end

# function Base.rand(
#     rng::Trng,
#     dgmp::Tdgmp,
#     t::Tt,
# ) where {
#     Trng<:Random.AbstractRNG,
#     Tdgmp<:AbstractDiscretizedGaussMarkovProcess,
#     Tt<:AbstractFloat,
# } end

# function Base.rand(
#     rng::Trng,
#     dgmp::Tdgmp,
#     t::Tt,
#     s::Tt,
#     xₛ::Txₛ,
# ) where {
#     Trng<:Random.AbstractRNG,
#     Tdgmp<:AbstractDiscretizedGaussMarkovProcess,
#     Tt<:AbstractFloat,
#     Txₛ<:AbstractVector,
# } end

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

function A(
    dgmp::Tdgmp,
    k::Tk,
) where {Tdgmp<:AbstractDiscretizedGaussMarkovProcess,Tk<:Integer}
    return A(dgmp, ts(dgmp)[k+1], k > 0 ? ts(dgmp)[k] : ts(dgmp)[k+1])
end

function Base.rand(
    rng::Trng,
    dgmp::Tdgmp,
    k::Tk,
) where {Trng<:Random.AbstractRNG,Tdgmp<:AbstractDiscretizedGaussMarkovProcess,Tk<:Integer}
    return rand(rng, dgmp, ts(tdgmp)[k])
end

function Base.rand(
    rng::Trng,
    dgmp::Tdgmp,
    k::Tk,
    xₖ::Txₖ,
) where {
    Trng<:Random.AbstractRNG,
    Tdgmp<:AbstractDiscretizedGaussMarkovProcess,
    Tk<:Integer,
    Txₖ<:AbstractVector,
}
    return rand(rng, dgmp, ts(dgmp)[k+1], ts(dgmp)[k], xₖ)
end
