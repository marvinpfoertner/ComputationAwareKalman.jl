struct FilterCache{
    T<:AbstractFloat,
    Tm<:AbstractVector{T},
    TM<:AbstractMatrix{T},
}
    ts::Vector{T}

    m⁻s::Vector{Tm}
    M⁻s::Vector{TM}

    ms::Vector{Tm}
    Ms::Vector{TM}

    Hᵀws::Vector{Tm}

    Ws::Vector{TM}
    HᵀWs::Vector{TM}
    P⁻HᵀWs::Vector{TM}
end

function FilterCache{T, Tm, TM}() where {T, Tm, TM}
    return FilterCache{T, Tm, TM}(T[], Tm[], TM[], Tm[], TM[], Tm[], TM[], TM[], TM[])
end

function FilterCache{T}() where {T}
    return FilterCache{T, Vector{T}, Matrix{T}}()
end
