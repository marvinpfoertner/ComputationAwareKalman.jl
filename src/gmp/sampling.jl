function Base.rand(rng::Random.AbstractRNG, gmp::AbstractGaussMarkovProcess, ts)
    samples_x = Vector{Float64}[]

    tⱼ₋₁ = ts[1]
    sample_xⱼ₋₁ = rand(rng, gmp, tⱼ₋₁)

    for j = 1:size(ts, 1)
        tⱼ = ts[j]
        sample_xⱼ = rand(rng, gmp, tⱼ, tⱼ₋₁, sample_xⱼ₋₁)

        push!(samples_x, sample_xⱼ)

        tⱼ₋₁ = tⱼ
        sample_xⱼ₋₁ = sample_xⱼ
    end

    return samples_x
end

function Base.rand(
    rng::Random.AbstractRNG,
    dgmp::DiscretizedGaussMarkovProcess,
    ts::AbstractVector{<:AbstractFloat},
)
    return rand(rng, dgmp.gmp, ts)
end

function Base.rand(
    rng::Random.AbstractRNG,
    dgmp::DiscretizedGaussMarkovProcess,
    mmod::AbstractMeasurementModel,
    ys::AbstractVector{<:AbstractVector{<:AbstractFloat}},
    fcache::AbstractFilterCache,
    ts_sample,
)
    ts_sample = unique!(sort!(vcat(ts(dgmp), ts_sample), alg = MergeSort))

    samples_x⁻ = Vector{Float64}[]
    samples_x = Vector{Float64}[]

    samples_w = Vector{Float64}[]

    k = 0

    tⱼ₋₁ = ts_sample[1]
    sample_xⱼ₋₁ = rand(rng, dgmp.gmp, tⱼ₋₁)

    for j = 1:size(ts_sample, 1)
        tⱼ = ts_sample[j]

        # Sample predict step
        sample_x⁻ⱼ = rand(rng, dgmp.gmp, tⱼ, tⱼ₋₁, sample_xⱼ₋₁)

        if (k < length(dgmp) && tⱼ == ts(dgmp)[k+1])
            k = k + 1

            # Sample y⁻ₖ
            sample_y⁻ₖ = rand(rng, mmod, k, sample_x⁻ⱼ)

            # Sample update step
            yₖ = ys[k]
            Uₖ = U(fcache, k)
            Wₖ = W(fcache, k)
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
        A₍ⱼ₊₁₎ⱼ = A(dgmp.gmp, tⱼ₊₁, tⱼ)
        sample_wˢⱼ = A₍ⱼ₊₁₎ⱼ' * sample_wˢⱼ₊₁

        if tⱼ == ts(dgmp)[k]
            sample_wₖ = samples_w[k]
            Wₖ = W(fcache, k)
            P⁻ₖWₖ = P⁻W(fcache, k)

            sample_wˢⱼ = sample_wₖ + sample_wˢⱼ - Wₖ * (P⁻ₖWₖ' * sample_wˢⱼ)

            M⁻ⱼ = M⁻(fcache, k)
        else
            Aⱼₖ = A(dgmp.gmp, tⱼ, ts(dgmp)[k])

            M⁻ⱼ = Aⱼₖ * M(fcache, k)
        end

        P⁻ⱼ = LowRankDowndatedMatrix(Σ(dgmp.gmp, tⱼ), M⁻ⱼ)

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
