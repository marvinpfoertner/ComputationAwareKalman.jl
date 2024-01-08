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
}
    m⁻s::Vector{Tm⁻}

    ms::Vector{Tm}
    Ms::Vector{TM}

    us::Vector{Tu}
    Us::Vector{TU}

    ws::Vector{Tw}  # Hₖᵀuₖ
    Ws::Vector{TW}  # HₖᵀUₖ

    M⁺s::Vector{TM⁺}
end

function FilterCache{T,Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺}() where {T,Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺}
    return FilterCache{T,Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺}(
        Tm⁻[],
        Tm[],
        TM[],
        Tu[],
        TU[],
        Tw[],
        TW[],
        TM⁺[],
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
    }()
end

function FilterCache()
    return FilterCache{Float64}()
end

function P(
    gmc::Tgmc,
    fcache::Tfcache,
    k,
) where {Tgmc<:AbstractGaussMarkovChain,Tfcache<:FilterCache}
    return LowRankDowndatedMatrix(prior_cov(gmc, k), fcache.Ms[k])
end

function M⁻(fcache::Tfcache, k) where {Tfcache<:FilterCache}
    Mₖ = fcache.Ms[k]

    if k > 1
        M⁺ₖ₋₁ = fcache.M⁺s[k-1]
        M⁻ₖ = Mₖ[:, 1:size(M⁺ₖ₋₁, 2)]
    else
        M⁻ₖ = Mₖ[:, 1:0]
    end

    return M⁻ₖ
end

function P⁻W(fcache::Tfcache, k) where {Tfcache<:FilterCache}
    Mₖ = fcache.Ms[k]

    if k > 1
        M⁺ₖ₋₁ = fcache.M⁺s[k-1]
        P⁻ₖWₖ = Mₖ[:, (size(M⁺ₖ₋₁, 2)+1):end]
    else
        P⁻ₖWₖ = Mₖ
    end

    return P⁻ₖWₖ
end

function Base.push!(
    fcache::FilterCache{T,Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺},
    m⁻::Tm⁻,
    x_cache::UpdateCache{T,Tm,TM,Tu,TU,Tw,TW},
    M⁺::TM⁺,
) where {T,Tm⁻,Tm,TM,Tu,TU,Tw,TW,TM⁺}
    push!(fcache.m⁻s, m⁻)

    push!(fcache.ms, x_cache.m)
    push!(fcache.Ms, x_cache.M)
    push!(fcache.us, x_cache.u)
    push!(fcache.Us, x_cache.U)
    push!(fcache.ws, x_cache.w)
    push!(fcache.Ws, x_cache.W)

    push!(fcache.M⁺s, M⁺)
end
