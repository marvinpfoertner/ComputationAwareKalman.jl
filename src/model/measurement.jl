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
