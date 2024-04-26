function interpolate(
    dgmp::DiscretizedGaussMarkovProcess,
    fcache::AbstractFilterCache,
    t::AbstractFloat,
)
    k = searchsortedlast(ts(dgmp), t)

    if k < 1
        mₜ = μ(dgmp.gmp, t)
        Mₜ = zeros(eltype(mₜ), size(mₜ, 1), 0)
    elseif t == ts(dgmp)[k]
        mₜ = m(fcache, k)
        Mₜ = M(fcache, k)
    else
        Aₜₖ = A(dgmp.gmp, t, ts(dgmp)[k])

        mₜ = Aₜₖ * m(fcache, k)
        Mₜ = Aₜₖ * M⁺(fcache, k)
    end

    Σₜ = Σ(dgmp.gmp, t)

    return ConditionalGaussianBelief(mₜ, Σₜ, Mₜ)
end

function interpolate(
    dgmp::DiscretizedGaussMarkovProcess,
    fcache::AbstractFilterCache,
    scache::AbstractSmootherCache,
    t::AbstractFloat,
)
    k = searchsortedlast(ts(dgmp), t)

    if k < 1
        mₜ = μ(dgmp.gmp, t)
        Mₜ = zeros(eltype(mₜ), size(m, 1), 0)
    elseif t == ts(dgmp)[k]
        mₜ = m(fcache, k)
        Mₜ = M(fcache, k)
    else
        Aₜₖ = A(dgmp.gmp, t, ts(dgmp)[k])

        mₜ = Aₜₖ * m(fcache, k)
        Mₜ = Aₜₖ * M⁺(fcache, k)
    end

    Σₜ = Σ(dgmp.gmp, t)

    if k >= length(dgmp)
        mˢₜ = mₜ
        Mˢₜ = Mₜ
    else
        Pₜ = LowRankDowndatedMatrix(Σₜ, Mₜ)
        A₍ₖ₊₁₎ₜ = A(dgmp.gmp, ts(dgmp)[k+1], t)

        mˢₜ = mₜ + Pₜ * (A₍ₖ₊₁₎ₜ' * wˢ(scache, k + 1))
        Mˢₜ = [Mₜ;; Pₜ * (A₍ₖ₊₁₎ₜ' * Wˢ(scache, k + 1))]
    end

    return ConditionalGaussianBelief(mˢₜ, Σₜ, Mˢₜ)
end
