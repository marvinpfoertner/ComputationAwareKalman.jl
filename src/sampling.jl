function Base.rand(
    rng::Trng,
    gmc::Tgmc,
) where {Trng<:Random.AbstractRNG,Tgmc<:AbstractGaussMarkovChain}
    x_samples = Vector{Float64}[]

    xₖ₋₁_sample = rand(rng, gmc, 0)

    for k = 1:length(gmc)
        # Sample transition
        xₖ_sample = rand(rng, gmc, k - 1, xₖ₋₁_sample)

        push!(x_samples, xₖ_sample)

        xₖ₋₁_sample = xₖ_sample
    end

    return x_samples
end

function Base.rand(
    rng::Trng,
    gmc::Tgmc,
    mmod::Tmmod,
) where {
    Trng<:Random.AbstractRNG,
    Tgmc<:AbstractGaussMarkovChain,
    Tmmod<:AbstractMeasurementModel,
}
    x_samples = rand(rng, gmc)
    y_samples = Vector{eltype(x_samples[1])}[]

    for k = 1:length(gmc)
        xₖ_sample = x_samples[k]
        yₖ_sample = rand(rng, mmod, k, xₖ_sample)

        push!(y_samples, yₖ_sample)
    end

    return x_samples, y_samples
end

function Base.rand(
    rng::Random.AbstractRNG,
    gmc::AbstractGaussMarkovChain,
    mmod::AbstractMeasurementModel,
    ys::AbstractVector{<:AbstractVector{<:AbstractFloat}},
    fcache::AbstractFilterCache,
)
    x_samples = Vector{Float64}[]
    w_samples = Vector{Float64}[]

    xₖ₋₁_sample = rand(rng, gmc, 0)

    for k = 1:length(gmc)
        # Sample transition
        x⁻ₖ_sample = rand(rng, gmc, k - 1, xₖ₋₁_sample)

        # Sample measurement model
        y⁻ₖ_sample = rand(rng, mmod, k, x⁻ₖ_sample)

        # Matheron's rule for update step
        yₖ = ys[k]
        Uₖ = U(fcache, k)
        Wₖ = W(fcache, k)
        P⁻ₖWₖ = P⁻W(fcache, k)

        Uₖᵀsample_rₖ = Uₖ' * (yₖ - y⁻ₖ_sample)

        sample_xₖ = x⁻ₖ_sample + P⁻ₖWₖ * Uₖᵀsample_rₖ
        sample_wₖ = Wₖ * Uₖᵀsample_rₖ

        push!(x_samples, sample_xₖ)
        push!(w_samples, sample_wₖ)

        xₖ₋₁_sample = sample_xₖ
    end

    xˢ_samples = [x_samples[end]]

    wˢₖ₊₁_sample = w_samples[end]

    for k = (length(gmc)-1):-1:1
        Aₖᵀwˢₖ₊₁_sample = A(gmc, k)' * wˢₖ₊₁_sample

        P⁻ₖWₖ = P⁻W(fcache, k)
        WₖᵀP⁻ₖAₖᵀwˢₖ₊₁_sample = P⁻ₖWₖ' * Aₖᵀwˢₖ₊₁_sample

        # Compute sample
        xₖ_sample = x_samples[k]
        P⁻ₖ = LowRankDowndatedMatrix(Σ(gmc, k), M⁻(fcache, k))

        xˢₖ_sample = xₖ_sample + P⁻ₖ * wˢₖ₊₁_sample - P⁻ₖWₖ * WₖᵀP⁻ₖAₖᵀwˢₖ₊₁_sample

        push!(xˢ_samples, xˢₖ_sample)

        # Compute wˢₖ sample
        wₖ_sample = w_samples[k]
        Wₖ = W(fcache, k)

        wˢₖ_sample = wₖ_sample + Aₖᵀwˢₖ₊₁_sample - Wₖ * WₖᵀP⁻ₖAₖᵀwˢₖ₊₁_sample

        wˢₖ₊₁_sample = wˢₖ_sample
    end

    return reverse!(xˢ_samples)
end
