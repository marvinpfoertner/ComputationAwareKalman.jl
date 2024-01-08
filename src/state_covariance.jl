struct ConditionalGaussianBelief{
    T<:AbstractFloat,
    Tm<:AbstractVector{T},
    TΣ<:AbstractMatrix{T},
    TM<:AbstractMatrix{T},
}
    m::Tm

    Σ::TΣ
    M::TM
end

function Statistics.mean(d::ConditionalGaussianBelief)
    return d.m
end

function Statistics.cov(d::ConditionalGaussianBelief)
    return LowRankDowndatedMatrix(d.Σ, d.M)
end
