struct LowRankDowndatedMatrix{
    T<:Number,
    TA<:AbstractMatrix{<:Number},
    TU<:AbstractMatrix{<:Number},
    TV<:AbstractMatrix{<:Number},
} <: AbstractMatrix{T}
    A::TA
    U::TU
    V::TV

    function LowRankDowndatedMatrix(A, U, V)
        if size(A, 1) != size(U, 1)
            error("`size(A, 1)` must be equal to `size(U, 1)")
        end

        if size(A, 2) != size(V, 1)
            error("`size(A, 2)` must be equal to `size(V, 1)")
        end

        if size(U, 2) != size(V, 2)
            error("`size(U, 2)` must be equal to `size(V, 1)")
        end

        T = promote_type(eltype(A), eltype(U), eltype(V))

        return new{T,typeof(A),typeof(U),typeof(V)}(A, U, V)
    end
end

function LowRankDowndatedMatrix(A::AbstractMatrix{<:Number}, U::AbstractMatrix{<:Number})
    if size(A, 1) != size(A, 2)
        error("`A` must be square")
    end

    return LowRankDowndatedMatrix(A, U, U)
end

function LowRankDowndatedMatrix(A::AbstractMatrix{<:Number})
    return LowRankDowndatedMatrix(A, Matrix{eltype(A)}(undef, size(A, 1), 0))
end

Base.size(M::LowRankDowndatedMatrix) = size(M.A)
Base.IndexStyle(::Type{<:LowRankDowndatedMatrix}) = IndexCartesian()
Base.getindex(M::LowRankDowndatedMatrix, i::Int, j::Int) =
    M.A[i, j] - M.V[j, :]' * M.U[i, :]

const MulMatTypes = [:AbstractMatrix, :Diagonal]

for TX in [:AbstractVector; MulMatTypes]
    @eval function Base.:*(M::LowRankDowndatedMatrix, X::$TX)
        return M.A * X - M.U * (M.V' * X)
    end
end

const MulAdjTypes = [
    :(LinearAlgebra.Adjoint{<:Number,<:AbstractVector}),
    :(LinearAlgebra.Transpose{<:Number,<:AbstractVector}),
]

for TX in [MulMatTypes; MulAdjTypes]
    @eval function Base.:*(X::$TX, M::LowRankDowndatedMatrix)
        return X * M.A - (X * M.U) * M.V'
    end
end
