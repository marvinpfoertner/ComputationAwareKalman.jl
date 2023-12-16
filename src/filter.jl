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

struct UpdateCache{
    T<:AbstractFloat,
    Tm<:AbstractVector{T},
    TM<:AbstractMatrix{T},
    Ty<:AbstractVector{T},
    TU<:AbstractMatrix{T},
    Tw<:AbstractVector{T},
    TW<:AbstractMatrix{T},
}
    m::Tm
    M::TM

    y::Ty

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

    return UpdateCache{T,typeof(m),typeof(M),Ty,typeof(U),typeof(w),typeof(W)}(
        m,
        M,
        y,
        U,
        w,
        W,
    )
end

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

struct FilterCache{
    T<:AbstractFloat,
    Tm<:AbstractVector{T},
    TM<:AbstractMatrix{T},
    TM⁺<:AbstractMatrix{T},
    Ty<:AbstractVector{T},
    TU<:AbstractMatrix{T},
    Tw<:AbstractVector{T},
    TW<:AbstractMatrix{T},
}
    ms::Vector{Tm}  # Mean of updated filter belief
    Ms::Vector{TM}  # Downdate to covariance of updated filter belief

    M⁺s::Vector{TM⁺}  # Truncated downdate to covariance of update filter belief

    ys::Vector{Ty}

    Us::Vector{TU}

    ws::Vector{Tw}  # Hₖᵀuₖ
    Ws::Vector{TW}  # HₖᵀUₖ
end

function FilterCache{T,Tm,TM,TM⁺,Ty,TU,Tw,TW}() where {T,Tm,TM,TM⁺,Ty,TU,Tw,TW}
    return FilterCache{T,Tm,TM,TM⁺,Ty,TU,Tw,TW}(Tm[], TM[], TM⁺[], Ty[], TU[], Tw[], TW[])
end

function FilterCache{T}() where {T}
    return FilterCache{
        T,
        Vector{T},  # Tm
        Matrix{T},  # TM
        Matrix{T},  # TM⁺
        Vector{T},  # Ty
        Matrix{T},  # TU
        Vector{T},  # Tw
        Matrix{T},  # TW
    }()
end

function FilterCache()
    return FilterCache{Float64}()
end

function P(
    gmc::Tgmc,
    fcache::Tfcache,
    k,
) where {Tgmc<:AbstractGaussMarkovChain,Tfcache<:FilterCache}
    return StateCovariance(prior_cov(gmc, k), fcache.Ms[k])
end

function M⁻(fcache::Tfcache, k) where {Tfcache<:FilterCache}
    Mₖ = fcache.Ms[k]

    if k > 1
        M⁺ₖ₋₁ = fcache.M⁺s[k-1]
        M⁻ₖ = Mₖ[:, 1:size(M⁺ₖ₋₁, 2)]
    else
        M⁻ₖ = Mₖ[:, 1:0]
    end

    return M⁻ₖ
end

function P⁻W(fcache::Tfcache, k) where {Tfcache<:FilterCache}
    Mₖ = fcache.Ms[k]

    if k > 1
        M⁺ₖ₋₁ = fcache.M⁺s[k-1]
        P⁻ₖWₖ = Mₖ[:, (size(M⁺ₖ₋₁, 2)+1):end]
    else
        P⁻ₖWₖ = Mₖ
    end

    return P⁻ₖWₖ
end

function Base.push!(
    fcache::FilterCache{T,Tm,TM,TM⁺,Ty,TU,Tw,TW},
    x::UpdateCache{T,Tm,TM,Ty,TU,Tw,TW},
    M⁺::TM⁺,
) where {T,Tm,TM,TM⁺,Ty,TU,Tw,TW}
    push!(fcache.ms, x.m)
    push!(fcache.Ms, x.M)
    push!(fcache.M⁺s, M⁺)
    push!(fcache.ys, x.y)
    push!(fcache.Us, x.U)
    push!(fcache.ws, x.w)
    push!(fcache.Ws, x.W)
end
