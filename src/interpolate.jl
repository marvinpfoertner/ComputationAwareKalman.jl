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

    if k < firstindex(ts(dgmp))
        m = prior_mean(dgmp, t)
        M = zeros(T, size(m, 1), 0)
    else
        A = transition(dgmp, t, ts(dgmp)[k])

        m = A * fcache.ms[k]
        M = A * fcache.Ms[k]
    end

    return ConditionalGaussianBelief(m, prior_cov(dgmp, t), M)
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

    if k < firstindex(ts(dgmp))
        mₜ = prior_mean(dgmp, t)
        Mₜ = zeros(T, size(m, 1), 0)
    else
        Aₖₜ = transition(dgmp, t, ts(dgmp)[k])

        mₜ = Aₖₜ * scache.ms[k]
        Mₜ = Aₖₜ * M(scache, k)
    end

    Σₜ = prior_cov(dgmp, t)

    if k >= lastindex(ts(dgmp))
        mˢₜ = mₜ
        Mˢₜ = Mₜ
    else
        Pₜ = StateCovariance(Σₜ, Mₜ)
        Aₜₖ₊₁ = transition(dgmp, ts(dgmp)[k+1], t)

        mˢₜ = mₜ + Pₜ * (Aₜₖ₊₁' * scache.wˢs[k+1])
        Mˢₜ = [Mₜ;; Pₜ * (Aₜₖ₊₁' * scache.Wˢs[k+1])]
    end

    return ConditionalGaussianBelief(mˢₜ, Σₜ, Mˢₜ)
end
