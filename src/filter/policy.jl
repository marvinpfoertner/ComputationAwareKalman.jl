struct CGPolicy end

function (policy::CGPolicy)(; r::AbstractVector{<:AbstractFloat}, kwargs...)
    return r
end

struct CoordinatePolicy
    idcs::Vector{Int}
end

function (policy::CoordinatePolicy)(;
    i::Integer,
    r::AbstractVector{<:AbstractFloat},
    kwargs...,
)
    v = zeros(Bool, size(r))
    v[i+1] = 1
    return v
end

struct RandomGaussianPolicy{Trng<:Random.AbstractRNG}
    rng::Trng
end

function (policy::RandomGaussianPolicy)(; r::AbstractVector{<:AbstractFloat}, kwargs...)
    return randn(policy.rng, eltype(r), length(r))
end
