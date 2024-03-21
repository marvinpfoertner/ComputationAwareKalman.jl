struct CGPolicy end

function (policy::CGPolicy)(; r::AbstractVector{<:AbstractFloat}, kwargs...)
    return r
end

struct RandomGaussianPolicy{Trng<:Random.AbstractRNG}
    rng::Trng
end

function (policy::RandomGaussianPolicy)(; r::AbstractVector{<:AbstractFloat}, kwargs...)
    return randn(policy.rng, eltype(r), length(r))
end
