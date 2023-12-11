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
        wˢₖ₊₁ = wˢs[end]
        Wˢₖ₊₁ = Wˢs[end]

        Pₖ = P(gmc, fcache, k)

        mˢₖ = fcache.ms[k] + Pₖ * wˢₖ₊₁
        Mˢₖ = [fcache.Ms[k];; Pₖ * Wˢₖ₊₁]

        push!(mˢs, mˢₖ)
        push!(Mˢs, Mˢₖ)

        Aₖ = transition(gmc, k)
        P⁻Wₖ = P⁻W(fcache, k)

        Aₖᵀwˢₖ₊₁ = Aₖ' * wˢₖ₊₁
        AₖᵀWˢₖ₊₁ = Aₖ' * Wˢₖ₊₁

        wˢₖ = fcache.ws[k] + Aₖᵀwˢₖ₊₁ - fcache.Ws[k] * (P⁻Wₖ' * Aₖᵀwˢₖ₊₁)
        Wˢₖ = [
            fcache.Ws[k];;
            AₖᵀWˢₖ₊₁ - fcache.Ws[k] * (P⁻Wₖ' * AₖᵀWˢₖ₊₁)
        ]

        Wˢₖ = truncate(Wˢₖ, svd_cutoff)

        push!(wˢs, wˢₖ)
        push!(Wˢs, Wˢₖ)
    end

    return SmootherCache(reverse!(mˢs), reverse!(Mˢs), reverse!(wˢs), reverse!(Wˢs))
end

struct SmootherCache{T<:AbstractFloat,Tm<:AbstractVector{T},TM<:AbstractMatrix{T}}
    mˢs::Vector{Tm}
    Mˢs::Vector{TM}

    wˢs::Vector{Tm}
    Wˢs::Vector{TM}
end
