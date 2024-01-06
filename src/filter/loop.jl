function filter(
    gmc::Tgmc,
    mmod::Tmmod,
    ys::Tys;
    abstol = 1e-6,
    reltol = 1e-8,
    svd_cutoff = 1e-12,
) where {
    Tgmc<:AbstractGaussMarkovChain,
    Tmmod<:AbstractMeasurementModel,
    Tys<:AbstractVector{<:AbstractVector},
}
    fcache = FilterCache()

    mₖ₋₁ = prior_mean(gmc, 0)
    M⁺ₖ₋₁ = zeros(eltype(mₖ₋₁), size(mₖ₋₁, 1), 0)

    for k = 1:length(gmc)
        # Predict
        m⁻ₖ, M⁻ₖ = predict(gmc, k, mₖ₋₁, M⁺ₖ₋₁)

        # Update
        yₖ = ys[k]

        xₖ = update(
            m⁻ₖ,
            prior_cov(gmc, k),
            M⁻ₖ,
            H(mmod, k),
            Λ(mmod, k),
            yₖ,
            abstol = abstol,
            reltol = reltol,
        )

        # Truncate
        M⁺ₖ = truncate(xₖ.M, svd_cutoff = svd_cutoff)

        push!(fcache, yₖ, xₖ, M⁺ₖ)

        mₖ₋₁ = xₖ.m
        M⁺ₖ₋₁ = M⁺ₖ
    end

    return fcache
end