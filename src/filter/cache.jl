struct FilterCache{
    T<:AbstractFloat,
    Ty<:AbstractVector{T},
    Tm<:AbstractVector{T},
    TM<:AbstractMatrix{T},
    TU<:AbstractMatrix{T},
    Tw<:AbstractVector{T},
    TW<:AbstractMatrix{T},
    TM⁺<:AbstractMatrix{T},
}
    ys::Vector{Ty}

    ms::Vector{Tm}  # Mean of updated filter belief
    Ms::Vector{TM}  # Downdate to covariance of updated filter belief

    Us::Vector{TU}

    ws::Vector{Tw}  # Hₖᵀuₖ
    Ws::Vector{TW}  # HₖᵀUₖ

    M⁺s::Vector{TM⁺}  # Truncated downdate to covariance of update filter belief
end

function FilterCache{T,Ty,Tm,TM,TU,Tw,TW,TM⁺}() where {T,Ty,Tm,TM,TU,Tw,TW,TM⁺}
    return FilterCache{T,Ty,Tm,TM,TU,Tw,TW,TM⁺}(Ty[], Tm[], TM[], TU[], Tw[], TW[], TM⁺[])
end

function FilterCache{T}() where {T}
    return FilterCache{
        T,
        Vector{T},  # Ty
        Vector{T},  # Tm
        Matrix{T},  # TM
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
    return StateCovariance(prior_cov(gmc, k), fcache.Ms[k])
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
    fcache::FilterCache{T,Ty,Tm,TM,TU,Tw,TW,TM⁺},
    y::Ty,
    x::UpdateCache{T,Tm,TM,TU,Tw,TW},
    M⁺::TM⁺,
) where {T,Ty,Tm,TM,TU,Tw,TW,TM⁺}
    push!(fcache.ys, y)

    push!(fcache.ms, x.m)
    push!(fcache.Ms, x.M)
    push!(fcache.Us, x.U)
    push!(fcache.ws, x.w)
    push!(fcache.Ws, x.W)

    push!(fcache.M⁺s, M⁺)
end
