abstract type AbstractSmootherCache{
    Tmˢ<:AbstractVector{<:AbstractFloat},
    TMˢ<:AbstractMatrix{<:AbstractFloat},
    Twˢ<:AbstractVector{<:AbstractFloat},
    TWˢ<:AbstractMatrix{<:AbstractFloat},
} end

function M(cache::AbstractSmootherCache, k)
    Mˢₖ = Mˢ(cache, k)

    if k == length(cache)
        Mₖ = Mˢₖ
    else
        Wˢₖ₊₁ = Wˢ(cache, k + 1)
        Mₖ = Mˢₖ[:, 1:(size(Mˢₖ, 2)-size(Wˢₖ₊₁, 2))]
    end

    return Mₖ
end

function Pˢ(gmc::AbstractGaussMarkovChain, cache::AbstractSmootherCache, k)
    return LowRankDowndatedMatrix(Σ(gmc, k), Mˢ(cache, k))
end

struct SmootherCache{Tmˢ,TMˢ,Twˢ,TWˢ} <: AbstractSmootherCache{Tmˢ,TMˢ,Twˢ,TWˢ}
    final_length::Int

    mˢs::Vector{Tmˢ}
    Mˢs::Vector{TMˢ}

    wˢs::Vector{Twˢ}
    Wˢs::Vector{TWˢ}
end

function SmootherCache{Tmˢ,TMˢ,Twˢ,TWˢ}(final_length::Int) where {Tmˢ,TMˢ,Twˢ,TWˢ}
    return SmootherCache{Tmˢ,TMˢ,Twˢ,TWˢ}(final_length, Tmˢ[], TMˢ[], Twˢ[], TWˢ[])
end

function Base.length(cache::SmootherCache)
    return length(cache.mˢs)
end

function mˢ(cache::SmootherCache, k)
    return cache.mˢs[cache.final_length-k+1]
end

function Mˢ(cache::SmootherCache, k)
    return cache.Mˢs[cache.final_length-k+1]
end

function wˢ(cache::SmootherCache, k)
    return cache.wˢs[cache.final_length-k+1]
end

function Wˢ(cache::SmootherCache, k)
    return cache.Wˢs[cache.final_length-k+1]
end

function Base.pushfirst!(
    cache::SmootherCache{Tmˢ,TMˢ,Twˢ,TWˢ},
    mˢ::Tmˢ,
    Mˢ::TMˢ,
    wˢ::Twˢ,
    Wˢ::TWˢ,
) where {Tmˢ,TMˢ,Twˢ,TWˢ}
    push!(cache.mˢs, mˢ)
    push!(cache.Mˢs, Mˢ)

    push!(cache.wˢs, wˢ)
    push!(cache.Wˢs, Wˢ)
end

mutable struct JLD2SmootherCache{Tmˢ,TMˢ,Twˢ,TWˢ} <: AbstractSmootherCache{Tmˢ,TMˢ,Twˢ,TWˢ}
    path::String

    final_length::Int
    current_length::Int
end

function JLD2SmootherCache{Tmˢ,TMˢ,Twˢ,TWˢ}(
    path::String,
    final_length::Int,
) where {Tmˢ,TMˢ,Twˢ,TWˢ}
    mkpath(path)

    return JLD2SmootherCache{Tmˢ,TMˢ,Twˢ,TWˢ}(path, final_length, 0)
end

function Base.length(cache::JLD2SmootherCache)
    return cache.current_length
end

function read_cache_entry(cache::JLD2SmootherCache, k, key)
    fpath = joinpath(cache.path, @sprintf("smoother_%010d.jld2", k))

    return jldopen(fpath, "r") do file
        file[key]
    end
end

function mˢ(cache::JLD2SmootherCache{Tmˢ,TMˢ,Twˢ,TWˢ}, k)::Tmˢ where {Tmˢ,TMˢ,Twˢ,TWˢ}
    return read_cache_entry(cache, k, "mˢ")
end

function Mˢ(cache::JLD2SmootherCache{Tmˢ,TMˢ,Twˢ,TWˢ}, k)::TMˢ where {Tmˢ,TMˢ,Twˢ,TWˢ}
    return read_cache_entry(cache, k, "Mˢ")
end

function wˢ(cache::JLD2SmootherCache{Tmˢ,TMˢ,Twˢ,TWˢ}, k)::Twˢ where {Tmˢ,TMˢ,Twˢ,TWˢ}
    return read_cache_entry(cache, k, "wˢ")
end

function Wˢ(cache::JLD2SmootherCache{Tmˢ,TMˢ,Twˢ,TWˢ}, k)::TWˢ where {Tmˢ,TMˢ,Twˢ,TWˢ}
    return read_cache_entry(cache, k, "Wˢ")
end

function Base.pushfirst!(
    cache::JLD2SmootherCache{Tmˢ,TMˢ,Twˢ,TWˢ},
    mˢ::Tmˢ,
    Mˢ::TMˢ,
    wˢ::Twˢ,
    Wˢ::TWˢ,
) where {Tmˢ,TMˢ,Twˢ,TWˢ}
    fpath = joinpath(
        cache.path,
        @sprintf("smoother_%010d.jld2", cache.final_length - cache.current_length)
    )

    jldopen(fpath, "w") do file
        file["mˢ"] = mˢ
        file["Mˢ"] = Mˢ
        file["wˢ"] = wˢ
        file["Wˢ"] = Wˢ
    end

    cache.current_length += 1
end
