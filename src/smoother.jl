struct SmootherCache{
    T<:AbstractFloat,
    Tmˢ<:AbstractVector{T},
    TMˢ<:AbstractMatrix{T},
    Twˢ<:AbstractVector{T},
    TWˢ<:AbstractMatrix{T},
    Tm<:AbstractVector{T},
    TM⁺<:AbstractMatrix{T},
}
    mˢs::Vector{Tmˢ}
    Mˢs::Vector{TMˢ}

    wˢs::Vector{Tm}
    Wˢs::Vector{TM}

    # Quantities from the filter
    ms::Vector{Tm}  # Mean of updated filter belief

    is::Vector{Int64}  # Number of PLS iterations

    M⁺s::Vector{TM⁺}  # Truncated downdate to covariance of updated filter belief
end

function smooth(
    gmc::Tgmc,
    fcache::Tfcache,
    svd_cutoff::T = 1e-12,
) where {Tgmc<:AbstractGaussMarkovChain,T<:AbstractFloat,Tfcache<:FilterCache}
    mˢs = [fcache.ms[end]]
    Mˢs = [fcache.Ms[end]]

    wˢs = [fcache.ws[end]]
    Wˢs = [fcache.Ws[end]]

    for k in reverse(indices(gmc))[2:end]
        Aₖ = transition(gmc, k)
        Aₖᵀwˢₖ₊₁ = Aₖ' * wˢs[end]
        AₖᵀWˢₖ₊₁ = Aₖ' * Wˢs[end]

        Pₖ = P(gmc, fcache, k)
        mˢₖ = fcache.ms[k] + Pₖ * Aₖᵀwˢₖ₊₁
        Mˢₖ = [fcache.Ms[k];; Pₖ * AₖᵀWˢₖ₊₁]

        push!(mˢs, mˢₖ)
        push!(Mˢs, Mˢₖ)

        wₖ = fcache.ws[k]
        Wₖ = fcache.Ws[k]
        P⁻Wₖ = P⁻W(fcache, k)
        wˢₖ = wₖ + Aₖᵀwˢₖ₊₁ - Wₖ * (P⁻Wₖ' * Aₖᵀwˢₖ₊₁)
        Wˢₖ = [Wₖ;; AₖᵀWˢₖ₊₁ - Wₖ * (P⁻Wₖ' * AₖᵀWˢₖ₊₁)]

        Wˢₖ = truncate(Wˢₖ, svd_cutoff)

        push!(wˢs, wˢₖ)
        push!(Wˢs, Wˢₖ)
    end

    return SmootherCache(
        reverse!(mˢs),
        reverse!(Mˢs),
        reverse!(wˢs),
        reverse!(Wˢs),
        fcache.ms,
        fcache.is,
        fcache.M⁺s,
    )
end
