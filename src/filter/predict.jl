function predict(
    gmc::Tgmc,
    k::Tk,
    mₖ₋₁::Tmₖ₋₁,
    Mₖ₋₁::TMₖ₋₁,
) where {
    Tgmc<:AbstractGaussMarkovChain,
    Tk<:Integer,
    T<:AbstractFloat,
    Tmₖ₋₁<:AbstractVector{T},
    TMₖ₋₁<:AbstractMatrix{T},
}
    Aₖ₋₁ = A(gmc, k - 1)
    m⁻ₖ = Aₖ₋₁ * mₖ₋₁
    M⁻ₖ = Aₖ₋₁ * Mₖ₋₁
    return m⁻ₖ, M⁻ₖ
end
