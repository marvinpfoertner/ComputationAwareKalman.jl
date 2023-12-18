function truncate(
    M::TM,
    svd_cutoff::T = 1e-12,
) where {T<:AbstractFloat,TM<:AbstractMatrix{T}}
    U, S, _ = svd(M)

    trunc_idx = findlast(S .> svd_cutoff)
    if isnothing(trunc_idx)
        trunc_idx = 0
    end

    return U[:, 1:trunc_idx] * Diagonal(S[1:trunc_idx])
end
