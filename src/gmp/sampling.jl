function Base.rand(
    rng::Trng,
    dgmp::Tdgmp,
    ts_sample::Tts,
) where {
    Trng<:Random.AbstractRNG,
    Tdgmp<:AbstractDiscretizedGaussMarkovProcess,
    Tts<:AbstractVector{<:AbstractFloat},
}
    samples_x = Vector{Float64}[]

    tⱼ₋₁ = ts_sample[1]
    sample_xⱼ₋₁ = rand(rng, dgmp, tⱼ₋₁)

    for j = 1:size(ts_sample, 1)
        tⱼ = ts_sample[j]
        sample_xⱼ = rand(rng, dgmp, tⱼ, tⱼ₋₁, sample_xⱼ₋₁)

        push!(samples_x, sample_xⱼ)

        tⱼ₋₁ = tⱼ
        sample_xⱼ₋₁ = sample_xⱼ
    end

    return samples_x
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
