abstract type AbstractFilterCache end

function M⁻(fcache::AbstractFilterCache, k)
    Mₖ = M(fcache, k)
    Wₖ = W(fcache, k)

    return Mₖ[:, 1:(size(Mₖ, 2)-size(Wₖ, 2))]
end

function P⁻W(fcache::AbstractFilterCache, k)
    Mₖ = M(fcache, k)
    Wₖ = W(fcache, k)

    return Mₖ[:, (size(Mₖ, 2)-size(Wₖ, 2)+1):end]
end

function P(gmc::AbstractGaussMarkovChain, fcache::AbstractFilterCache, k)
    return LowRankDowndatedMatrix(Σ(gmc, k), M(fcache, k))
end

struct FilterCache{
    T<:AbstractFloat,
    Tm⁻<:AbstractVector{T},
    Tm<:AbstractVector{T},
    TM<:AbstractMatrix{T},
    Tu<:AbstractVector{T},
    TU<:AbstractMatrix{T},
    Tw<:AbstractVector{T},
    TW<:AbstractMatrix{T},
    TM⁺<:AbstractMatrix{T},
    TΠ⁺<:AbstractMatrix{T},
} <: AbstractFilterCache
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

function FilterCache{
    T,
    Tm⁻,
    Tm,
    TM,
    Tu,
    TU,
    Tw,
    TW,
    TM⁺,
    TΠ⁺,
}() where {T,Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}
    return FilterCache{T,Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}(
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

function FilterCache{T}() where {T}
    return FilterCache{
        T,
        Vector{T},  # Tm⁻
        Vector{T},  # Tm
        Matrix{T},  # TM
        Vector{T},  # Tu
        Matrix{T},  # TU
        Vector{T},  # Tw
        Matrix{T},  # TW
        Matrix{T},  # TM⁺
        Matrix{T},  # TΠ⁺
    }()
end

function FilterCache()
    return FilterCache{Float64}()
end

function m⁻(fcache::FilterCache, k)
    return fcache.m⁻s[k]
end

function m(fcache::FilterCache, k)
    return fcache.ms[k]
end

function M(fcache::FilterCache, k)
    return fcache.Ms[k]
end

function u(fcache::FilterCache, k)
    return fcache.us[k]
end

function U(fcache::FilterCache, k)
    return fcache.Us[k]
end

function w(fcache::FilterCache, k)
    return fcache.ws[k]
end

function W(fcache::FilterCache, k)
    return fcache.Ws[k]
end

function M⁺(fcache::FilterCache, k)
    return fcache.M⁺s[k]
end

function Π⁺(fcache::FilterCache, k)
    return fcache.Π⁺s[k]
end

function Base.push!(
    fcache::FilterCache{T,Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺},
    m⁻::Tm⁻,
    x_cache::UpdateCache{T,Tm,TM,Tu,TU,Tw,TW},
    M⁺::TM⁺,
    Π⁺::TΠ⁺,
) where {T,Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺,TΠ⁺}
    push!(fcache.m⁻s, m⁻)

    push!(fcache.ms, x_cache.m)
    push!(fcache.Ms, x_cache.M)
    push!(fcache.us, x_cache.u)
    push!(fcache.Us, x_cache.U)
    push!(fcache.ws, x_cache.w)
    push!(fcache.Ws, x_cache.W)

    push!(fcache.M⁺s, M⁺)
    push!(fcache.Π⁺s, Π⁺)
end
