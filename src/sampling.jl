function Base.rand(
    rng::Trng,
    gmc::Tgmc,
    mmod::Tmmod,
    ys::Tys,
    fcache::Tfcache,
) where {
    Trng<:Random.AbstractRNG,
    Tgmc<:AbstractDiscretizedGaussMarkovProcess,
    Tmmod<:AbstractMeasurementModel,
    T<:AbstractFloat,
    Tys<:AbstractVector{<:AbstractVector{T}},
    Tfcache<:FilterCache{T},
}
    x_samples = Vector{T}[]
    w_samples = Vector{T}[]

    xₖ₋₁_sample = rand(rng, gmc, 0)

    for k = 1:length(gmc)
        # Sample transition
        x⁻ₖ_sample = rand(rng, gmc, k - 1, xₖ₋₁_sample)

        # Sample measurement model
        y⁻ₖ_sample = rand(rng, mmod, k, x⁻ₖ_sample)

        # Matheron's rule for update step
        yₖ = ys[k]
        Uₖ = fcache.Us[k]
        Wₖ = fcache.Ws[k]
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
        P⁻ₖ = StateCovariance(prior_cov(gmc, k), M⁻(fcache, k))

        xˢₖ_sample = xₖ_sample + P⁻ₖ * wˢₖ₊₁_sample - P⁻ₖWₖ * WₖᵀP⁻ₖAₖᵀwˢₖ₊₁_sample

        push!(xˢ_samples, xˢₖ_sample)

        # Compute wˢₖ sample
        wₖ_sample = w_samples[k]
        Wₖ = fcache.Ws[k]

        wˢₖ_sample = wₖ_sample + Aₖᵀwˢₖ₊₁_sample - Wₖ * WₖᵀP⁻ₖAₖᵀwˢₖ₊₁_sample

        wˢₖ₊₁_sample = wˢₖ_sample
    end

    return reverse!(xˢ_samples)
end

function Base.rand(
    rng::Trng,
    dgmp::Tdgmp,
    mmod::Tmmod,
    ys::Tys,
    fcache::Tfcache,
    ts_sample,
) where {
    Trng<:Random.AbstractRNG,
    Tdgmp<:AbstractDiscretizedGaussMarkovProcess,
    Tmmod<:AbstractMeasurementModel,
    Tys<:AbstractVector{<:AbstractVector{<:AbstractFloat}},
    Tfcache<:FilterCache,
}
    ts_sample = unique!(sort!(vcat(ts(dgmp), ts_sample), alg = MergeSort))

    samples_x⁻ = Vector{Float64}[]
    samples_x = Vector{Float64}[]

    samples_w = Vector{Float64}[]

    k = 0

    tⱼ₋₁ = ts_sample[1]
    sample_xⱼ₋₁ = rand(rng, dgmp, tⱼ₋₁)

    for j = 1:size(ts_sample, 1)
        tⱼ = ts_sample[j]

        # Sample predict step
        sample_x⁻ⱼ = rand(rng, dgmp, tⱼ, tⱼ₋₁, sample_xⱼ₋₁)

        if (k < length(dgmp) && tⱼ == ts(dgmp)[k+1])
            k = k + 1

            # Sample y⁻ₖ
            sample_y⁻ₖ = rand(rng, mmod, k, sample_x⁻ⱼ)

            # Sample update step
            yₖ = ys[k]
            Uₖ = fcache.Us[k]
            Wₖ = fcache.Ws[k]
            P⁻ₖWₖ = P⁻W(fcache, k)

            Uₖᵀsample_rₖ = Uₖ' * (yₖ - sample_y⁻ₖ)

            sample_xⱼ = sample_x⁻ⱼ + P⁻ₖWₖ * Uₖᵀsample_rₖ
            sample_wₖ = Wₖ * Uₖᵀsample_rₖ

            push!(samples_w, sample_wₖ)
        else
            sample_xⱼ = sample_x⁻ⱼ
        end

        push!(samples_x⁻, sample_x⁻ⱼ)
        push!(samples_x, sample_xⱼ)

        tⱼ₋₁ = tⱼ
        sample_xⱼ₋₁ = sample_xⱼ
    end

    samples_xˢ = [samples_x[end]]

    k = k - 1

    sample_wˢⱼ₊₁ = samples_w[end]

    for j = (size(ts_sample, 1)-1):-1:1
        tⱼ = ts_sample[j]
        tⱼ₊₁ = ts_sample[j+1]


        # Compute sample_wⱼ and P⁻ⱼ
        A₍ⱼ₊₁₎ⱼ = A(dgmp, tⱼ₊₁, tⱼ)
        sample_wˢⱼ = A₍ⱼ₊₁₎ⱼ' * sample_wˢⱼ₊₁

        if tⱼ == ts(dgmp)[k]
            sample_wₖ = samples_w[k]
            Wₖ = fcache.Ws[k]
            P⁻ₖWₖ = P⁻W(fcache, k)

            sample_wˢⱼ = sample_wₖ + sample_wˢⱼ - Wₖ * (P⁻ₖWₖ' * sample_wˢⱼ)

            M⁻ⱼ = M⁻(fcache, k)
        else
            Aⱼₖ = A(dgmp, tⱼ, ts(dgmp)[k])

            M⁻ⱼ = Aⱼₖ * fcache.Ms[k]
        end

        P⁻ⱼ = StateCovariance(prior_cov(dgmp, tⱼ), M⁻ⱼ)

        # Compute sample
        sample_xˢⱼ = samples_x⁻[j] + P⁻ⱼ * sample_wˢⱼ

        push!(samples_xˢ, sample_xˢⱼ)

        sample_wˢⱼ₊₁ = sample_wˢⱼ

        if tⱼ == ts(dgmp)[k]
            k = k - 1
        end
    end

    return reverse!(samples_xˢ)
end
