function predict(
    m::Tm,
    M::TM,
    A::TA,
) where {T<:AbstractFloat,Tm<:AbstractVector{T},TM<:AbstractMatrix{T},TA<:AbstractMatrix{T}}
    return A * m, A * M
end

struct UpdateCache{
    T<:AbstractFloat,
    Tm<:AbstractVector{T},
    TM<:AbstractMatrix{T},
    TU<:AbstractMatrix{T},
    Tw<:AbstractVector{T},
    TW<:AbstractMatrix{T},
}
    m::Tm
    M::TM

    i::Int64

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

    return UpdateCache{T,typeof(m),typeof(M),typeof(U),typeof(w),typeof(W)}(
        m,
        M,
        i,
        U,
        w,
        W,
    )
end

struct FilterCache{
    T<:AbstractFloat,
    Tm<:AbstractVector{T},
    TM<:AbstractMatrix{T},
    TM⁺<:AbstractMatrix{T},
    TU<:AbstractMatrix{T},
    Tw<:AbstractVector{T},
    TW<:AbstractMatrix{T},
}
    ms::Vector{Tm}  # Mean of updated filter belief
    Ms::Vector{TM}  # Downdate to covariance of updated filter belief

    is::Vector{Int64}  # Number of PLS iterations

    M⁺s::Vector{TM⁺}  # Truncated downdate to covariance of update filter belief

    Us::Vector{TU}

    ws::Vector{Tw}  # Hₖᵀuₖ
    Ws::Vector{TW}  # HₖᵀUₖ
end

function FilterCache{T,Tm,TM,TM⁺,TU,Tw,TW}() where {T,Tm,TM,TM⁺,TU,Tw,TW}
    return FilterCache{T,Tm,TM,TM⁺,TU,Tw,TW}(Tm[], TM[], Int64[], TM⁺[], TU[], Tw[], TW[])
end

function FilterCache{T}() where {T}
    return FilterCache{T,Vector{T},Matrix{T},Matrix{T},Matrix{T},Vector{T},Matrix{T}}()
end

function FilterCache()
    return FilterCache{Float64}()
end

function Base.push!(
    fcache::FilterCache{T,Tm,TM,TM⁺,TU,Tw,TW},
    x::UpdateCache{T,Tm,TM,TU,Tw,TW},
    M⁺::TM⁺,
) where {T,Tm,TM,TM⁺,TU,Tw,TW}
    push!(fcache.ms, x.m)
    push!(fcache.Ms, x.M)
    push!(fcache.is, x.i)
    push!(fcache.M⁺s, M⁺)
    push!(fcache.Us, x.U)
    push!(fcache.ws, x.w)
    push!(fcache.Ws, x.W)
end
