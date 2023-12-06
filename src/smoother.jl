struct SmootherCache{T<:AbstractFloat,Tm<:AbstractVector{T},TM<:AbstractMatrix{T}}
    ts::Vector{T}

    mˢs::Vector{Tm}
    Mˢs::Vector{TM}

    wˢs::Vector{Tm}
    Wˢs::Vector{TM}
end

function SmootherCache(filter_cache::FilterCache{T,Tm,TM}) where {T,Tm,TM}
    return SmootherCache{T,Tm,TM}(
        filter_cache.ts,
        [filter_cache.ms[end]],
        [filter_cache.Ms[end]],
        [filter_cache.Hᵀws[end]],
        [filter_cache.HᵀWs[end]],
    )
end
