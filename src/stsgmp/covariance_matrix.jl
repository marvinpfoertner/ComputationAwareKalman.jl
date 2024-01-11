function covariance_matrix(covariance_fn, X)
    return KernelMatrix{eltype(X)}(covariance_fn, X, X)
end

struct KernelMatrix{T<:Real,TK,TX₁,TX₂} <: AbstractMatrix{T}
    K::TK
    X₁::TX₁
    X₂::TX₂
end

function KernelMatrix{T}(K::TK, X₁::TX₁, X₂::TX₂) where {T,TK,TX₁,TX₂}
    return KernelMatrix{T,TK,TX₁,TX₂}(K, X₁, X₂)
end

Base.size(Kₓ₁ₓ₂::KernelMatrix) = (length(Kₓ₁ₓ₂.X₁), length(Kₓ₁ₓ₂.X₂))
Base.IndexStyle(::Type{<:KernelMatrix}) = IndexCartesian()
Base.getindex(Kₓ₁ₓ₂::KernelMatrix, i::Int, j::Int) = Kₓ₁ₓ₂.K(Kₓ₁ₓ₂.X₁[i], Kₓ₁ₓ₂.X₂[j])

LinearAlgebra.adjoint(Kₓ₁ₓ₂::KernelMatrix{T}) where {T} =
    KernelMatrix{T}(Kₓ₁ₓ₂.K, Kₓ₁ₓ₂.X₂, Kₓ₁ₓ₂.X₁)
LinearAlgebra.transpose(Kₓ₁ₓ₂::KernelMatrix{T}) where {T} =
    KernelMatrix{T}(Kₓ₁ₓ₂.K, Kₓ₁ₓ₂.X₂, Kₓ₁ₓ₂.X₁)
