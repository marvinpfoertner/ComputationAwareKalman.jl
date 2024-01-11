function truncate(
    M::TM;
    svd_cutoff::T = 1e-12,
) where {T<:AbstractFloat,TM<:AbstractMatrix{T}}
    U, S, V = svd(M)

    trunc_idx = findlast(S .> svd_cutoff)
    if isnothing(trunc_idx)
        trunc_idx = 0
    end

    M⁺ = U[:, 1:trunc_idx] * Diagonal(S[1:trunc_idx])

    return M⁺, V[:, 1:trunc_idx]
end
