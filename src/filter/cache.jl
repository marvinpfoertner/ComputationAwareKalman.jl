abstract type AbstractFilterCache{
    Tm⁻<:AbstractVector{<:AbstractFloat},
    Tm<:AbstractVector{<:AbstractFloat},
    TM<:AbstractMatrix{<:AbstractFloat},
    Tu<:AbstractVector{<:AbstractFloat},
    TU<:AbstractMatrix{<:AbstractFloat},
    Tw<:AbstractVector{<:AbstractFloat},
    TW<:AbstractMatrix{<:AbstractFloat},
    TM⁺<:AbstractMatrix{<:AbstractFloat},
    TΠ⁺<:AbstractMatrix{<:AbstractFloat},
} end

function M⁻(cache::AbstractFilterCache, k)
    Mₖ = M(cache, k)
    Wₖ = W(cache, k)

    return Mₖ[:, 1:(size(Mₖ, 2)-size(Wₖ, 2))]
end

function P⁻W(cache::AbstractFilterCache, k)
    Mₖ = M(cache, k)
    Wₖ = W(cache, k)

    return Mₖ[:, (size(Mₖ, 2)-size(Wₖ, 2)+1):end]
end

function P(gmc::AbstractGaussMarkovChain, cache::AbstractFilterCache, k)
    return LowRankDowndatedMatrix(Σ(gmc, k), M(cache, k))
end

struct FilterCache{Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺} <:
       AbstractFilterCache{Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}
    m⁻s::Vector{Tm⁻}

    ms::Vector{Tm}
    Ms::Vector{TM}

    us::Vector{Tu}
    Us::Vector{TU}

    ws::Vector{Tw}  # Hₖᵀuₖ
    Ws::Vector{TW}  # HₖᵀUₖ

    M⁺s::Vector{TM⁺}
    Π⁺s::Vector{TΠ⁺}
end

function FilterCache{Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}() where {Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}
    return FilterCache{Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}(
        Tm⁻[],
        Tm[],
        TM[],
        Tu[],
        TU[],
        Tw[],
        TW[],
        TM⁺[],
        TΠ⁺[],
    )
end

function FilterCache(eltype::Type{<:AbstractFloat} = Float64)
    return FilterCache{
        Vector{eltype},  # Tm⁻
        Vector{eltype},  # Tm
        Matrix{eltype},  # TM
        Vector{eltype},  # Tu
        Matrix{eltype},  # TU
        Vector{eltype},  # Tw
        Matrix{eltype},  # TW
        Matrix{eltype},  # TM⁺
        Matrix{eltype},  # TΠ⁺
    }()
end

function Base.length(cache::FilterCache)
    return length(cache.m⁻s)
end

function m⁻(cache::FilterCache, k)
    return cache.m⁻s[k]
end

function m(cache::FilterCache, k)
    return cache.ms[k]
end

function M(cache::FilterCache, k)
    return cache.Ms[k]
end

function u(cache::FilterCache, k)
    return cache.us[k]
end

function U(cache::FilterCache, k)
    return cache.Us[k]
end

function w(cache::FilterCache, k)
    return cache.ws[k]
end

function W(cache::FilterCache, k)
    return cache.Ws[k]
end

function M⁺(cache::FilterCache, k)
    return cache.M⁺s[k]
end

function Π⁺(cache::FilterCache, k)
    return cache.Π⁺s[k]
end

function Base.push!(
    cache::FilterCache{Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺};
    m⁻::Tm⁻,
    m::Tm,
    M::TM,
    u::Tu,
    U::TU,
    w::Tw,
    W::TW,
    M⁺::TM⁺,
    Π⁺::TΠ⁺,
) where {Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}
    push!(cache.m⁻s, m⁻)
    push!(cache.ms, m)
    push!(cache.Ms, M)
    push!(cache.us, u)
    push!(cache.Us, U)
    push!(cache.ws, w)
    push!(cache.Ws, W)
    push!(cache.M⁺s, M⁺)
    push!(cache.Π⁺s, Π⁺)
end

mutable struct JLD2FilterCache{Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺} <:
               AbstractFilterCache{Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}
    path::String
    length::Int
end

function JLD2FilterCache{Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}(
    path::String,
) where {Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}
    mkpath(path)

    return JLD2FilterCache{Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}(path, 0)
end

function JLD2FilterCache(path::String, eltype::Type{<:AbstractFloat} = Float64)
    return JLD2FilterCache{
        Vector{eltype},  # Tm⁻
        Vector{eltype},  # Tm
        Matrix{eltype},  # TM
        Vector{eltype},  # Tu
        Matrix{eltype},  # TU
        Vector{eltype},  # Tw
        Matrix{eltype},  # TW
        Matrix{eltype},  # TM⁺
        Matrix{eltype},  # TΠ⁺
    }(
        path,
    )
end

function Base.length(cache::JLD2FilterCache)
    return cache.length
end

function read_cache_entry(cache::JLD2FilterCache, k, key)
    fpath = joinpath(cache.path, @sprintf("filter_%010d.jld2", k))

    return jldopen(fpath, "r") do file
        file[key]
    end
end

function m⁻(
    cache::JLD2FilterCache{Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺},
    k,
)::Tm⁻ where {Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}
    return read_cache_entry(cache, k, "m⁻")
end

function m(
    cache::JLD2FilterCache{Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺},
    k,
)::Tm where {Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}
    return read_cache_entry(cache, k, "m")
end

function M(
    cache::JLD2FilterCache{Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺},
    k,
)::TM where {Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}
    return read_cache_entry(cache, k, "M")
end

function u(
    cache::JLD2FilterCache{Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺},
    k,
)::Tu where {Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}
    return read_cache_entry(cache, k, "u")
end

function U(
    cache::JLD2FilterCache{Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺},
    k,
)::TU where {Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}
    return read_cache_entry(cache, k, "U")
end

function w(
    cache::JLD2FilterCache{Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺},
    k,
)::Tw where {Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}
    return read_cache_entry(cache, k, "w")
end

function W(
    cache::JLD2FilterCache{Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺},
    k,
)::TW where {Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}
    return read_cache_entry(cache, k, "W")
end

function M⁺(
    cache::JLD2FilterCache{Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺},
    k,
)::TM⁺ where {Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}
    return read_cache_entry(cache, k, "M⁺")
end

function Π⁺(
    cache::JLD2FilterCache{Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺},
    k,
)::TΠ⁺ where {Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}
    return read_cache_entry(cache, k, "Π⁺")
end

function Base.push!(
    cache::JLD2FilterCache{Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺};
    m⁻::Tm⁻,
    m::Tm,
    M::TM,
    u::Tu,
    U::TU,
    w::Tw,
    W::TW,
    M⁺::TM⁺,
    Π⁺::TΠ⁺,
) where {Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}
    fpath = joinpath(cache.path, @sprintf("filter_%010d.jld2", length(cache) + 1))

    jldopen(fpath, "w") do file
        file["m⁻"] = m⁻
        file["m"] = m
        file["M"] = M
        file["u"] = u
        file["U"] = U
        file["w"] = w
        file["W"] = W
        file["M⁺"] = M⁺
        file["Π⁺"] = Π⁺
    end

    cache.length += 1
end
