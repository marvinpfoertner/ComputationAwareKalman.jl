function filter(
    gmc::Tgmc,
    mmod::Tmmod,
    ys::Tys;
    update_kwargs = (;),
    truncate_kwargs = (;),
) where {
    Tgmc<:AbstractGaussMarkovChain,
    Tmmod<:AbstractMeasurementModel,
    Tys<:AbstractVector{<:AbstractVector},
}
    fcache = FilterCache()

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

        push!(fcache, yₖ, xₖ, M⁺ₖ, Π⁺ₖ)

        mₖ₋₁ = xₖ.m
        M⁺ₖ₋₁ = M⁺ₖ
    end

    return fcache
end
