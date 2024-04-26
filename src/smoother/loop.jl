function smooth(
    gmc::AbstractGaussMarkovChain,
    scache::AbstractSmootherCache;
    callback_fn = ((args...; kwargs...) -> nothing),
    truncate_kwargs = (;),
)
    callback_fn(length(gmc))

    wˢₖ₊₁ = wˢ(scache, length(gmc))
    Wˢₖ₊₁ = Wˢ(scache, length(gmc))

    for k in reverse(1:length(gmc)-1)
        Aₖ = A(gmc, k)
        Aₖᵀwˢₖ₊₁ = Aₖ' * wˢₖ₊₁
        AₖᵀWˢₖ₊₁ = Aₖ' * Wˢₖ₊₁

        Pₖ = P(gmc, scache, k)
        mˢₖ = m(scache, k) + Pₖ * Aₖᵀwˢₖ₊₁
        Mˢₖ = [M(scache, k);; Pₖ * AₖᵀWˢₖ₊₁]

        wₖ = w(scache, k)
        Wₖ = W(scache, k)
        P⁻Wₖ = P⁻W(scache, k)
        wˢₖ = wₖ + Aₖᵀwˢₖ₊₁ - Wₖ * (P⁻Wₖ' * Aₖᵀwˢₖ₊₁)
        Wˢₖ = [Wₖ;; AₖᵀWˢₖ₊₁ - Wₖ * (P⁻Wₖ' * AₖᵀWˢₖ₊₁)]

        Wˢₖ, Πˢₖ = truncate(Wˢₖ; truncate_kwargs...)

        pushfirst!(scache, mˢₖ, Mˢₖ, wˢₖ, Wˢₖ, Πˢₖ)

        callback_fn(k)

        wˢₖ₊₁ = wˢₖ
        Wˢₖ₊₁ = Wˢₖ
    end

    return scache
end

function smooth(gmc::AbstractGaussMarkovChain, fcache::AbstractFilterCache; kwargs...)
    return smooth(gmc, SmootherCache(fcache); kwargs...)
end
