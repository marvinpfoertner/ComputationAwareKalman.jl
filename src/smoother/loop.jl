function smooth!(
    gmc::AbstractGaussMarkovChain,
    fcache::AbstractFilterCache,
    scache::AbstractSmootherCache;
    callback_fn = ((args...; kwargs...) -> nothing),
    truncate_kwargs = (;),
)
    @assert length(fcache) == length(gmc)
    @assert length(scache) == 0

    wˢₖ = w(fcache, length(gmc))
    Wˢₖ = W(fcache, length(gmc))

    pushfirst!(scache, m(fcache, length(gmc)), M(fcache, length(gmc)), wˢₖ, Wˢₖ)

    callback_fn(length(gmc))

    wˢₖ₊₁ = wˢₖ
    Wˢₖ₊₁ = Wˢₖ

    for k in reverse(1:length(gmc)-1)
        Aₖ = A(gmc, k)
        Aₖᵀwˢₖ₊₁ = Aₖ' * wˢₖ₊₁
        AₖᵀWˢₖ₊₁ = Aₖ' * Wˢₖ₊₁

        Pₖ = P(gmc, fcache, k)
        mˢₖ = m(fcache, k) + Pₖ * Aₖᵀwˢₖ₊₁
        Mˢₖ = [M(fcache, k);; Pₖ * AₖᵀWˢₖ₊₁]

        wₖ = w(fcache, k)
        Wₖ = W(fcache, k)
        P⁻Wₖ = P⁻W(fcache, k)
        wˢₖ = wₖ + Aₖᵀwˢₖ₊₁ - Wₖ * (P⁻Wₖ' * Aₖᵀwˢₖ₊₁)
        Wˢₖ = [Wₖ;; AₖᵀWˢₖ₊₁ - Wₖ * (P⁻Wₖ' * AₖᵀWˢₖ₊₁)]

        Wˢₖ, _ = truncate(Wˢₖ; truncate_kwargs...)

        pushfirst!(scache, mˢₖ, Mˢₖ, wˢₖ, Wˢₖ)

        callback_fn(k)

        wˢₖ₊₁ = wˢₖ
        Wˢₖ₊₁ = Wˢₖ
    end
end

function smooth(
    gmc::AbstractGaussMarkovChain,
    fcache::AbstractFilterCache{Tm⁻,Tm,TM,Tu,TU,Tw,TW};
    kwargs...,
) where {Tm⁻,Tm,TM,Tu,TU,Tw,TW}
    scache = SmootherCache{Tm,TM,Tw,TW}(length(gmc))
    smooth!(gmc, fcache, scache; kwargs...)
    return scache
end
