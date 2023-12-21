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
    y::Ty,
    abstol::T = 1e-6,
    reltol::T = 1e-8,
) where {
    T<:AbstractFloat,
    Tm⁻<:AbstractVector{T},
    TΣ<:AbstractMatrix{T},
    TM⁻<:AbstractMatrix{T},
    TH<:AbstractMatrix{T},
    TΛ<:AbstractMatrix{T},
    Ty<:AbstractVector{T},
}
    P⁻ = StateCovariance(Σ, M⁻)

    S = H * (P⁻ * H') + Λ

    i = 0

    u = zeros(size(H, 1))
    U = zeros(size(H, 1), 0)

    r₀ = y - H * m⁻
    r = r₀

    tol = max(abstol, reltol * norm(y, 2))

    while i < size(u, 1) && norm(r, 2) > tol
        v = r

        α = v' * r
        d = v - U * (U' * (S * v))
        η = v' * S * d

        u = u + (α / η) * d
        U = [U;; sqrt(1 / η) * d]

        i = i + 1

        r = r₀ - S * u
    end

    w = H' * u
    W = H' * U

    P⁻w = P⁻ * w
    P⁻W = P⁻ * W

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