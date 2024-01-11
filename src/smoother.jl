struct SmootherCache{
    T<:AbstractFloat,
    Tmˢ<:AbstractVector{T},
    TMˢ<:AbstractMatrix{T},
    Twˢ<:AbstractVector{T},
    TWˢ<:AbstractMatrix{T},
    TΠˢ<:AbstractMatrix{T},
    Tm<:AbstractVector{T},
    TM⁺<:AbstractMatrix{T},
}
    mˢs::Vector{Tmˢ}
    Mˢs::Vector{TMˢ}

    wˢs::Vector{Twˢ}
    Wˢs::Vector{TWˢ}

    Πˢs::Vector{TΠˢ}

    # Quantities from the filter
    ms::Vector{Tm}  # Mean of updated filter belief

    M⁺s::Vector{TM⁺}  # Truncated downdate to covariance of updated filter belief
end

function M(scache::Tscache, k) where {Tscache<:SmootherCache}
    Mˢₖ = scache.Mˢs[k]

    if k == length(scache.Mˢs)
        Mₖ = Mˢₖ
    else
        Wˢₖ₊₁ = scache.Wˢs[k+1]
        Mₖ = Mˢₖ[:, 1:(size(Mˢₖ, 2)-size(Wˢₖ₊₁, 2))]
    end

    return Mₖ
end

function smooth(
    gmc::Tgmc,
    fcache::Tfcache;
    svd_cutoff::T = 1e-12,
) where {Tgmc<:AbstractGaussMarkovChain,T<:AbstractFloat,Tfcache<:FilterCache}
    mˢs = [fcache.ms[end]]
    Mˢs = [fcache.Ms[end]]

    wˢs = [fcache.ws[end]]
    Wˢs = [fcache.Ws[end]]

    Πˢs = Matrix{T}[]

    for k in reverse(1:length(gmc)-1)
        Aₖ = A(gmc, k)
        Aₖᵀwˢₖ₊₁ = Aₖ' * wˢs[end]
        AₖᵀWˢₖ₊₁ = Aₖ' * Wˢs[end]

        Pₖ = P(gmc, fcache, k)
        mˢₖ = fcache.ms[k] + Pₖ * Aₖᵀwˢₖ₊₁
        Mˢₖ = [fcache.M⁺s[k];; Pₖ * AₖᵀWˢₖ₊₁]

        push!(mˢs, mˢₖ)
        push!(Mˢs, Mˢₖ)

        wₖ = fcache.ws[k]
        Wₖ = fcache.Ws[k]
        P⁻Wₖ = P⁻W(fcache, k)
        wˢₖ = wₖ + Aₖᵀwˢₖ₊₁ - Wₖ * (P⁻Wₖ' * Aₖᵀwˢₖ₊₁)
        Wˢₖ = [Wₖ;; AₖᵀWˢₖ₊₁ - Wₖ * (P⁻Wₖ' * AₖᵀWˢₖ₊₁)]

        Wˢₖ, Πˢₖ = truncate(Wˢₖ, svd_cutoff = svd_cutoff)

        push!(wˢs, wˢₖ)
        push!(Wˢs, Wˢₖ)

        push!(Πˢs, Πˢₖ)
    end

    return SmootherCache(
        reverse!(mˢs),
        reverse!(Mˢs),
        reverse!(wˢs),
        reverse!(Wˢs),
        reverse!(Πˢs),
        fcache.ms,
        fcache.M⁺s,
    )
end
