abstract type AbstractGaussMarkovChain end

# TODO: Document
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

# TODO: Document that this should return the identity for k = 0
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
