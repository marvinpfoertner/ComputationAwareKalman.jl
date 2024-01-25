function truncate(
    M::AbstractMatrix{T};
    max_cols::Integer = size(M, 1),
    min_sval::T = eps(T),
) where {T<:AbstractFloat}
    U, S, V = svd(M)

    j_max = findlast(S .>= min_sval)

    if isnothing(j_max)
        j_max = 0
    end

    j_max = min(j_max, max_cols)

    M⁺ = U[:, 1:j_max] * Diagonal(S[1:j_max])

    return M⁺, V[:, 1:j_max]
end
