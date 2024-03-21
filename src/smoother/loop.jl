function smooth(
    gmc::AbstractGaussMarkovChain,
    fcache::AbstractFilterCache;
    callback_fn = ((args...; kwargs...) -> nothing),
    truncate_kwargs = (;),
)
    mˢs = [m(fcache, length(gmc))]
    Mˢs = [M(fcache, length(gmc))]

    wˢs = [w(fcache, length(gmc))]
    Wˢs = [W(fcache, length(gmc))]

    Πˢs = Matrix{eltype(Wˢs[end])}[]

    callback_fn(length(gmc))

    for k in reverse(1:length(gmc)-1)
        Aₖ = A(gmc, k)
        Aₖᵀwˢₖ₊₁ = Aₖ' * wˢs[end]
        AₖᵀWˢₖ₊₁ = Aₖ' * Wˢs[end]

        Pₖ = P(gmc, fcache, k)
        mˢₖ = m(fcache, k) + Pₖ * Aₖᵀwˢₖ₊₁
        Mˢₖ = [M(fcache, k);; Pₖ * AₖᵀWˢₖ₊₁]

        push!(mˢs, mˢₖ)
        push!(Mˢs, Mˢₖ)

        wₖ = w(fcache, k)
        Wₖ = W(fcache, k)
        P⁻Wₖ = P⁻W(fcache, k)
        wˢₖ = wₖ + Aₖᵀwˢₖ₊₁ - Wₖ * (P⁻Wₖ' * Aₖᵀwˢₖ₊₁)
        Wˢₖ = [Wₖ;; AₖᵀWˢₖ₊₁ - Wₖ * (P⁻Wₖ' * AₖᵀWˢₖ₊₁)]

        Wˢₖ, Πˢₖ = truncate(Wˢₖ; truncate_kwargs...)

        push!(wˢs, wˢₖ)
        push!(Wˢs, Wˢₖ)

        push!(Πˢs, Πˢₖ)

        callback_fn(k)
    end

    return SmootherCache(
        reverse!(mˢs),
        reverse!(Mˢs),
        reverse!(wˢs),
        reverse!(Wˢs),
        reverse!(Πˢs),
        fcache.ms,  # TODO: This only works for `FilterCache`
        fcache.M⁺s,  # TODO: This only works for `FilterCache`
    )
end