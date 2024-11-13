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

function Statistics.var(d::ConditionalGaussianBelief)
    return diag(Statistics.cov(d))
end

function Statistics.std(d::ConditionalGaussianBelief)
    return sqrt.(Statistics.var(d))
end

function Base.:*(A::AbstractMatrix, d::ConditionalGaussianBelief)
    return ConditionalGaussianBelief(A * d.m, A * d.Σ * A', A * d.M)
end
