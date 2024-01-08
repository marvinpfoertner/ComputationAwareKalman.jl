function interpolate(
    dgmp::DiscretizedGaussMarkovProcess,
    fcache::Tfcache,
    t::Tt,
) where {T<:AbstractFloat,Tfcache<:FilterCache{T},Tt<:AbstractFloat}
    k = searchsortedlast(ts(dgmp), t)

    if k < 1
        mₜ = μ(dgmp.gmp, t)
        Mₜ = zeros(T, size(mₜ, 1), 0)
    elseif t == ts(dgmp)[k]
        mₜ = fcache.ms[k]
        Mₜ = fcache.Ms[k]
    else
        Aₜₖ = A(dgmp.gmp, t, ts(dgmp)[k])

        mₜ = Aₜₖ * fcache.ms[k]
        Mₜ = Aₜₖ * fcache.M⁺s[k]
    end

    Σₜ = Σ(dgmp.gmp, t)

    return ConditionalGaussianBelief(mₜ, Σₜ, Mₜ)
end

function interpolate(
    dgmp::DiscretizedGaussMarkovProcess,
    scache::Tscache,
    t::Tt,
) where {T<:AbstractFloat,Tscache<:SmootherCache{T},Tt<:AbstractFloat}
    k = searchsortedlast(ts(dgmp), t)

    if k < 1
        mₜ = μ(dgmp.gmp, t)
        Mₜ = zeros(T, size(m, 1), 0)
    elseif t == ts(dgmp)[k]
        mₜ = scache.ms[k]
        Mₜ = M(scache, k)
    else
        Aₜₖ = A(dgmp.gmp, t, ts(dgmp)[k])

        mₜ = Aₜₖ * scache.ms[k]
        Mₜ = Aₜₖ * scache.M⁺s[k]
    end

    Σₜ = Σ(dgmp.gmp, t)

    if k >= length(dgmp)
        mˢₜ = mₜ
        Mˢₜ = Mₜ
    else
        Pₜ = LowRankDowndatedMatrix(Σₜ, Mₜ)
        A₍ₖ₊₁₎ₜ = A(dgmp.gmp, ts(dgmp)[k+1], t)

        mˢₜ = mₜ + Pₜ * (A₍ₖ₊₁₎ₜ' * scache.wˢs[k+1])
        Mˢₜ = [Mₜ;; Pₜ * (A₍ₖ₊₁₎ₜ' * scache.Wˢs[k+1])]
    end

    return ConditionalGaussianBelief(mˢₜ, Σₜ, Mˢₜ)
end
