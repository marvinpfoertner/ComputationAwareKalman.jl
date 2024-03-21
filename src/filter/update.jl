struct UpdateCache{
    T<:AbstractFloat,
    Tm<:AbstractVector{T},
    TM<:AbstractMatrix{T},
    Tu<:AbstractVector{T},
    TU<:AbstractMatrix{T},
    Tw<:AbstractVector{T},
    TW<:AbstractMatrix{T},
}
    m::Tm
    M::TM

    u::Tu
    U::TU

    w::Tw
    W::TW
end

function update(
    m⁻::Tm⁻,
    Σ::TΣ,
    M⁻::TM⁻,
    H::TH,
    Λ::TΛ,
    y::Ty;
    abstol::T = 1e-10,
    reltol::T = 1e-12,
    max_iter::Integer = size(H, 1),
    policy = CGPolicy(),
) where {
    T<:AbstractFloat,
    Tm⁻<:AbstractVector{T},
    TΣ<:AbstractMatrix{T},
    TM⁻<:AbstractMatrix{T},
    TH<:AbstractMatrix,
    TΛ<:AbstractMatrix{T},
    Ty<:AbstractVector{T},
}
    P⁻Hᵀ = LowRankDowndatedMatrix(Σ * H', M⁻, H * M⁻)
    HP⁻Hᵀ = LowRankDowndatedMatrix(H * P⁻Hᵀ.A, P⁻Hᵀ.V)

    S(v) = HP⁻Hᵀ * v + Λ * v

    i = 0

    u = zeros(size(H, 1))
    U = zeros(size(H, 1), 0)

    r₀ = y - H * m⁻
    r = r₀

    tol = max(abstol, reltol * norm(y, 2))

    while i < max_iter && norm(r, 2) > tol
        v = policy(i = i, u = u, U = U, r = r)

        α = v' * r
        d = v - U * (U' * S(v))
        η = v' * S(d)

        u = u + (α / η) * d
        U = [U;; sqrt(1 / η) * d]

        i = i + 1

        r = r₀ - S(u)
    end

    w = H' * u
    W = H' * U

    P⁻w = P⁻Hᵀ * u
    P⁻W = P⁻Hᵀ * U

    m = m⁻ + P⁻w
    M = [M⁻;; P⁻W]

    return UpdateCache{T,typeof(m),typeof(M),typeof(u),typeof(U),typeof(w),typeof(W)}(
        m,
        M,
        u,
        U,
        w,
        W,
    )
end
