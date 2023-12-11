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
    fcache::Tfcache,
    scache::Tscache,
    t::Tt,
) where {
    Tdgmp<:AbstractDiscretizedGaussMarkovProcess,
    T<:AbstractFloat,
    Tfcache<:FilterCache{T},
    Tscache<:SmootherCache{T},
    Tt<:AbstractFloat,
}
    k = searchsortedlast(ts(dgmp), t)

    xᶠ = interpolate(dgmp, fcache, t)
    m = Statistics.mean(xᶠ)
    M = xᶠ.M
    P = Statistics.cov(xᶠ)

    if k >= lastindex(ts(dgmp))
        mˢ = m
        Mˢ = M
    else
        A = transition(dgmp, ts(dgmp)[k+1], t)

        mˢ = m + P * (A' * scache.wˢs[k+1])
        Mˢ = [M;; P * (A' * scache.Wˢs[k+1])]
    end

    return ConditionalGaussianBelief(mˢ, prior_cov(dgmp, t), Mˢ)
end
