# AbstractGaussMarkovChain interface
abstract type AbstractGaussMarkovChain end

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

# AbstractDiscretizedGaussMarkovProcess interface
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
    return rand(rng, dgmp, k > 0 ? ts(dgmp)[k] : ts(dgmp)[k+1])
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
    if k == 0
        return xₖ
    end

    return rand(rng, dgmp, ts(dgmp)[k+1], ts(dgmp)[k], xₖ)
end

# AbstractMeasurementModel interface
abstract type AbstractMeasurementModel end

function H(mmod::Tmmod, k::Tk) where {Tmmod<:AbstractMeasurementModel,Tk<:Integer} end

function Λ(mmod::Tmmod, k::Tk) where {Tmmod<:AbstractMeasurementModel,Tk<:Integer} end

function lsqrt_Λ(mmod::Tmmod, k::Tk) where {Tmmod<:AbstractMeasurementModel,Tk<:Integer} end

function Base.rand(
    rng::Trng,
    mmod::Tmmod,
    k::Tk,
    x::Tx,
) where {
    Trng<:Random.AbstractRNG,
    Tmmod<:AbstractMeasurementModel,
    Tk<:Integer,
    Tx<:AbstractVector,
}
    lsqrtΛ = lsqrt_Λ(mmod, k)

    return H(mmod, k) * x + lsqrtΛ * randn(rng, eltype(lsqrtΛ), size(lsqrtΛ, 2))
end

struct UniformMeasurementModel{
    T<:AbstractFloat,
    TH<:AbstractMatrix{T},
    TΛ<:AbstractMatrix{T},
    Tlsqrt_Λ<:AbstractMatrix{T},
} <: AbstractMeasurementModel
    H::TH
    Λ::TΛ
    lsqrt_Λ::Tlsqrt_Λ
end

function UniformMeasurementModel(
    H::TH,
    Λ::TΛ,
) where {T<:AbstractFloat,TH<:AbstractMatrix{T},TΛ<:AbstractMatrix{T}}
    lsqrt_Λ = sqrt(Λ)
    return UniformMeasurementModel(H, Λ, lsqrt_Λ)
end

function H(mmod::Tmmod, k::Tk) where {Tmmod<:UniformMeasurementModel,Tk<:Integer}
    return mmod.H
end

function Λ(mmod::Tmmod, k::Tk) where {Tmmod<:UniformMeasurementModel,Tk<:Integer}
    return mmod.Λ
end

function lsqrt_Λ(mmod::Tmmod, k::Tk) where {Tmmod<:UniformMeasurementModel,Tk<:Integer}
    return mmod.lsqrt_Λ
end
