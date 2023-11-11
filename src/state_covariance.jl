using Kronecker

struct StateCovariance{T<:AbstractFloat, TΣ<:AbstractMatrix{T}, TM<:AbstractMatrix{T}} <: AbstractMatrix{T}
    Σ::TΣ
    M::TM

    function StateCovariance(Σ, M)
        if size(Σ, 1) != size(Σ, 2)
            error("`Σ` must be square")
        end

        if size(Σ, 1) != size(M, 1)
            error("`size(Σ, 1)` must be equal to `size(M, 1)")
        end

        return new{eltype(Σ), typeof(Σ), typeof(M)}(Σ, M)
    end
end

function StateCovariance(Σ::AbstractMatrix{<:AbstractFloat})
    M = Matrix{eltype(Σ)}(undef, size(Σ, 1), 0)
    return StateCovariance(Σ, M)
end

Base.size(P::StateCovariance) = size(P.Σ)
Base.IndexStyle(::Type{<:StateCovariance}) = IndexCartesian()
Base.getindex(P::StateCovariance, I::Vararg{Int, 2}) = P.Σ[I[1], I[2]] - transpose(P.M[I[1], :]) * conj(P.M[I[2], :])

const MulMatTypes = [:AbstractMatrix, :Diagonal, :GeneralizedKroneckerProduct]

for TX in [:AbstractVector; MulMatTypes]
    @eval function Base.:*(P::StateCovariance, X::$TX)
        return P.Σ * X - P.M * (P.M' * X)
    end
end

for TX in MulMatTypes
    @eval function Base.:*(X::$TX, P::StateCovariance)
        return X * P.Σ - (X * P.M) * P.M'
    end
end

function Base.:*(vadj::LinearAlgebra.Adjoint{<:Number,<:AbstractVector}, P::StateCovariance)
    return vadj * P.Σ - (vadj * P.M) * P.M'
end
