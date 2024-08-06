function filter!(
    gmc::AbstractGaussMarkovChain,
    mmod::AbstractMeasurementModel,
    ys::AbstractVector{<:AbstractVector},
    cache::AbstractFilterCache;
    update_kwargs = (;),
    truncate_kwargs = (;),
)
    mₖ₋₁ = μ(gmc, 0)
    M⁺ₖ₋₁ = zeros(eltype(mₖ₋₁), size(mₖ₋₁, 1), 0)

    for k = 1:length(gmc)
        # Predict
        m⁻ₖ, M⁻ₖ = predict(gmc, k, mₖ₋₁, M⁺ₖ₋₁)

        # Update
        yₖ = ys[k]

        xₖ = update(m⁻ₖ, Σ(gmc, k), M⁻ₖ, H(mmod, k), Λ(mmod, k), yₖ; update_kwargs...)

        # Truncate
        M⁺ₖ, Π⁺ₖ = truncate(xₖ.M; truncate_kwargs...)

        push!(cache; m⁻ = m⁻ₖ, xₖ..., M⁺ = M⁺ₖ, Π⁺ = Π⁺ₖ)

        mₖ₋₁ = xₖ.m
        M⁺ₖ₋₁ = M⁺ₖ
    end
end

function filter(args...; kwargs...)
    cache = FilterCache()

    filter!(args..., cache; kwargs...)

    return cache
end
