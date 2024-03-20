abstract type AbstractSmootherCache end

function M(scache::AbstractSmootherCache, k)
    Mˢₖ = Mˢ(scache, k)

    if k == length(scache)
        Mₖ = Mˢₖ
    else
        Wˢₖ₊₁ = Wˢ(scache, k + 1)
        Mₖ = Mˢₖ[:, 1:(size(Mˢₖ, 2)-size(Wˢₖ₊₁, 2))]
    end

    return Mₖ
end

struct SmootherCache{
    T<:AbstractFloat,
    Tmˢ<:AbstractVector{T},
    TMˢ<:AbstractMatrix{T},
    Twˢ<:AbstractVector{T},
    TWˢ<:AbstractMatrix{T},
    TΠˢ<:AbstractMatrix{T},
    Tm<:AbstractVector{T},
    TM⁺<:AbstractMatrix{T},
} <: AbstractSmootherCache
    mˢs::Vector{Tmˢ}
    Mˢs::Vector{TMˢ}

    wˢs::Vector{Twˢ}
    Wˢs::Vector{TWˢ}

    Πˢs::Vector{TΠˢ}

    # Quantities from the filter
    ms::Vector{Tm}  # Mean of updated filter belief

    M⁺s::Vector{TM⁺}  # Truncated downdate to covariance of updated filter belief
end

function Base.length(scache::SmootherCache)
    return length(scache.mˢs)
end

function m(scache::SmootherCache, k)
    return scache.ms[k]
end

function mˢ(scache::SmootherCache, k)
    return scache.mˢs[k]
end

function Mˢ(scache::SmootherCache, k)
    return scache.Mˢs[k]
end

function M⁺(scache::SmootherCache, k)
    return scache.M⁺s[k]
end

function wˢ(scache::SmootherCache, k)
    return scache.wˢs[k]
end

function Wˢ(scache::SmootherCache, k)
    return scache.Wˢs[k]
end

function Πˢ(scache::SmootherCache, k)
    return scache.Πˢs[k]
end
