abstract type AbstractSmootherCache end

function m⁻(scache::AbstractSmootherCache, k)
    return m⁻(fcache(scache), k)
end

function M⁻(scache::AbstractSmootherCache, k)
    return M⁻(fcache(scache), k)
end

function m(scache::AbstractSmootherCache, k)
    return m(fcache(scache), k)
end

function M(scache::AbstractSmootherCache, k)
    # Mˢₖ = Mˢ(scache, k)

    # if k == length(scache)
    #     Mₖ = Mˢₖ
    # else
    #     Wˢₖ₊₁ = Wˢ(scache, k + 1)
    #     Mₖ = Mˢₖ[:, 1:(size(Mˢₖ, 2)-size(Wˢₖ₊₁, 2))]
    # end

    # return Mₖ

    return M(fcache(scache), k)
end

function w(scache::AbstractSmootherCache, k)
    return w(fcache(scache), k)
end

function W(scache::AbstractSmootherCache, k)
    return W(fcache(scache), k)
end

function M⁺(scache::AbstractSmootherCache, k)
    return M⁺(fcache(scache), k)
end

function P⁻W(scache::AbstractSmootherCache, k)
    return P⁻W(fcache(scache), k)
end

function P(gmc::AbstractGaussMarkovChain, scache::AbstractSmootherCache, k)
    return P(gmc, fcache(scache), k)
end

struct SmootherCache{
    Tmˢ<:AbstractVector{<:AbstractFloat},
    TMˢ<:AbstractMatrix{<:AbstractFloat},
    Twˢ<:AbstractVector{<:AbstractFloat},
    TWˢ<:AbstractMatrix{<:AbstractFloat},
    TΠˢ<:AbstractMatrix{<:AbstractFloat},
    Tfcache<:AbstractFilterCache,
} <: AbstractSmootherCache
    mˢs::Vector{Tmˢ}
    Mˢs::Vector{TMˢ}

    wˢs::Vector{Twˢ}
    Wˢs::Vector{TWˢ}

    Πˢs::Vector{TΠˢ}

    fcache::Tfcache
end

function SmootherCache(fcache)
    mˢs = [m(fcache, length(fcache))]
    Mˢs = [M(fcache, length(fcache))]

    wˢs = [w(fcache, length(fcache))]
    Wˢs = [W(fcache, length(fcache))]

    Πˢs = Matrix{eltype(Wˢs[end])}[]

    return SmootherCache(mˢs, Mˢs, wˢs, Wˢs, Πˢs, fcache)
end

function Base.length(scache::SmootherCache)
    return length(scache.mˢs)
end

function mˢ(scache::SmootherCache, k)
    return scache.mˢs[length(fcache(scache))-k+1]
end

function Mˢ(scache::SmootherCache, k)
    return scache.Mˢs[length(fcache(scache))-k+1]
end

function wˢ(scache::SmootherCache, k)
    return scache.wˢs[length(fcache(scache))-k+1]
end

function Wˢ(scache::SmootherCache, k)
    return scache.Wˢs[length(fcache(scache))-k+1]
end

function Πˢ(scache::SmootherCache, k)
    return scache.Πˢs[length(fcache(scache))-k+1]
end

function fcache(scache::SmootherCache)
    return scache.fcache
end

function Base.pushfirst!(
    scache::SmootherCache{Tmˢ,TMˢ,Twˢ,TWˢ,TΠˢ},
    mˢ::Tmˢ,
    Mˢ::TMˢ,
    wˢ::Twˢ,
    Wˢ::TWˢ,
    Πˢ::TΠˢ,
) where {Tmˢ,TMˢ,Twˢ,TWˢ,TΠˢ}
    push!(scache.mˢs, mˢ)
    push!(scache.Mˢs, Mˢ)

    push!(scache.wˢs, wˢ)
    push!(scache.Wˢs, Wˢ)

    push!(scache.Πˢs, Πˢ)
end
