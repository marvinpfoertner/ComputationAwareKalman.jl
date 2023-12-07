struct UpdatedBelief{
    T<:AbstractFloat,
    Tm<:AbstractVector{T},
    TΣ<:AbstractMatrix{T},
    TM<:AbstractMatrix{T},
    TU<:AbstractMatrix{T},
    Tw<:AbstractVector{T},
    TW<:AbstractMatrix{T},
}
    m::Tm

    Σ::TΣ
    M::TM

    i::Int64

    U::TU

    w::Tw
    W::TW
end

struct FilterCache{
    T<:AbstractFloat,
    Tm<:AbstractVector{T},
    TM<:AbstractMatrix{T},
    Tlgssm<:AbstractDiscreteLGSSM,
}
    lgssm::Tlgssm

    ms::Vector{Tm}  # Mean of updated filter belief
    Ms::Vector{TM}  # Downdate to covariance of updated filter belief

    is::Vector{Int64}  # Number of PLS iterations

    M⁺s::Vector{TM}  # Truncated downdate to covariance of update filter belief

    Us::Vector{TM}

    ws::Vector{Tm}  # Hₖᵀuₖ
    Ws::Vector{TM}  # HₖᵀUₖ
end

function FilterCache{T,Tm,TM}(lgssm::Tlgssm) where {T,Tm,TM,Tlgssm}
    return FilterCache{T,Tm,TM,Tlgssm}(lgssm, Tm[], TM[], Int64[], TM[], TM[], Tm[], TM[])
end

function FilterCache{T}(lgssm::Tlgssm) where {T,Tlgssm}
    return FilterCache{T,Vector{T},Matrix{T}}(lgssm)
end
