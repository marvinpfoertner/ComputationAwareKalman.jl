function interpolate(
    dgmp::Tdgmp,
    fcache::Tfcache,
    t::T,
) where {
    Tdgmp<:AbstractDiscretizedGaussMarkovProcess,
    T<:AbstractFloat,
    Tfcache<:FilterCache{T},
}
    k = searchsortedlast(ts(dgmp), t)

    if k < 1
        mₜ = prior_mean(dgmp, t)
        Mₜ = zeros(T, size(mₜ, 1), 0)
    elseif t == ts(dgmp)[k]
        mₜ = fcache.ms[k]
        Mₜ = fcache.Ms[k]
    else
        Aₜₖ = transition(dgmp, t, ts(dgmp)[k])

        mₜ = Aₜₖ * fcache.ms[k]
        Mₜ = Aₜₖ * fcache.M⁺s[k]
    end

    Σₜ = prior_cov(dgmp, t)

    return ConditionalGaussianBelief(mₜ, Σₜ, Mₜ)
end

function interpolate(
    dgmp::Tdgmp,
    scache::Tscache,
    t::Tt,
) where {
    Tdgmp<:AbstractDiscretizedGaussMarkovProcess,
    T<:AbstractFloat,
    Tscache<:SmootherCache{T},
    Tt<:AbstractFloat,
}
    k = searchsortedlast(ts(dgmp), t)

    if k < 1
        mₜ = prior_mean(dgmp, t)
        Mₜ = zeros(T, size(m, 1), 0)
    elseif t == ts(dgmp)[k]
        mₜ = scache.ms[k]
        Mₜ = M(scache, k)
    else
        Aₜₖ = transition(dgmp, t, ts(dgmp)[k])

        mₜ = Aₜₖ * scache.ms[k]
        Mₜ = Aₜₖ * scache.M⁺s[k]
    end

    Σₜ = prior_cov(dgmp, t)

    if k >= length(dgmp)
        mˢₜ = mₜ
        Mˢₜ = Mₜ
    else
        Pₜ = StateCovariance(Σₜ, Mₜ)
        A₍ₖ₊₁₎ₜ = transition(dgmp, ts(dgmp)[k+1], t)

        mˢₜ = mₜ + Pₜ * (A₍ₖ₊₁₎ₜ' * scache.wˢs[k+1])
        Mˢₜ = [Mₜ;; Pₜ * (A₍ₖ₊₁₎ₜ' * scache.Wˢs[k+1])]
    end

    return ConditionalGaussianBelief(mˢₜ, Σₜ, Mˢₜ)
end
