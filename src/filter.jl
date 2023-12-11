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
    Tlgssm<:AbstractDiscreteLGSSM,
    Tm<:AbstractVector{T},
    TM<:AbstractMatrix{T},
    TM⁺<:AbstractMatrix{T},
    TU<:AbstractMatrix{T},
    Tw<:AbstractVector{T},
    TW<:AbstractMatrix{T},
}
    lgssm::Tlgssm

    ms::Vector{Tm}  # Mean of updated filter belief
    Ms::Vector{TM}  # Downdate to covariance of updated filter belief

    is::Vector{Int64}  # Number of PLS iterations

    M⁺s::Vector{TM⁺}  # Truncated downdate to covariance of update filter belief

    Us::Vector{TU}

    ws::Vector{Tw}  # Hₖᵀuₖ
    Ws::Vector{TW}  # HₖᵀUₖ
end

function FilterCache{T,Tlgssm,Tm,TM,TM⁺,TU,Tw,TW}(
    lgssm::Tlgssm,
) where {T,Tlgssm,Tm,TM,TM⁺,TU,Tw,TW}
    return FilterCache{T,Tlgssm,Tm,TM,TM⁺,TU,Tw,TW}(
        lgssm,
        Tm[],
        TM[],
        Int64[],
        TM⁺[],
        TU[],
        Tw[],
        TW[],
    )
end

function FilterCache{T}(lgssm::Tlgssm) where {T,Tlgssm}
    return FilterCache{T,Tlgssm,Vector{T},Matrix{T},Matrix{T},Matrix{T},Vector{T},Matrix{T}}(
        lgssm,
    )
end

function Base.push!(
    fcache::FilterCache{T,Tlgssm,Tm,TM,TM⁺,TU,Tw,TW},
    x::UpdatedBelief{T,Tm,TΣ,TM,TU,Tw,TW},
    M⁺::TM⁺,
) where {T,Tlgssm,Tm,TM,TM⁺,TU,Tw,TW,TΣ}
    push!(fcache.ms, x.m)
    push!(fcache.Ms, x.M)
    push!(fcache.is, x.i)
    push!(fcache.M⁺s, M⁺)
    push!(fcache.Us, x.U)
    push!(fcache.ws, x.w)
    push!(fcache.Ws, x.W)
end
