struct CGPolicy end

function (policy::CGPolicy)(; r::AbstractVector{<:AbstractFloat}, kwargs...)
    return r
end
