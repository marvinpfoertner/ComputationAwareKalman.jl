abstract type AbstractGaussMarkovProcess end

"""
    μ(gmp::AbstractGaussMarkovProcess, t::Real)

Mean of the state at time `t`.
"""
μ(::AbstractGaussMarkovProcess, ::Real)

"""
    Σ(gmp::AbstractGaussMarkovProcess, t::Real)

Covariance matrix of the state at time `t`.
"""
Σ(::AbstractGaussMarkovProcess, ::Real)

"""
    lsqrt_Σ(gmp::AbstractGaussMarkovProcess, t::Real)

A left square root of the covariance matrix of the state at time `t`.
"""
function lsqrt_Σ(gmp::AbstractGaussMarkovProcess, t::Real)
    return sqrt(Σ(gmp, t))
end

"""
    A(gmp::AbstractGaussMarkovProcess, t::Real, s::Real)

Transition matrix from time `s` to time `t`.

Must return the identity matrix for `t = s`.
"""
A(::AbstractGaussMarkovProcess, ::Real, ::Real)

"""
    A_b_lsqrt_Q(gmp::AbstractGaussMarkovProcess, t::Real, s::Real)

Transition matrix, bias vector, and left square-root of the process noise covariance from time `s` to time `t`.

Will only be called with `t >= s`.
"""
A_b_lsqrt_Q(::AbstractGaussMarkovProcess, ::Real, ::Real)

"""
    rand(rng::Random.AbstractRNG, gmp::AbstractGaussMarkovProcess, t::Real)

Sample from the state at time `t`.
"""
function Base.rand(rng::Random.AbstractRNG, gmp::AbstractGaussMarkovProcess, t::Real)
    lsqrt_Σₜ = lsqrt_Σ(gmp, t)

    return μ(gmp, t) + lsqrt_Σₜ * randn(rng, eltype(lsqrt_Σₜ), size(lsqrt_Σₜ, 2))
end

"""
    rand(rng::Random.AbstractRNG, gmp::AbstractGaussMarkovProcess, t::Real, s::Real, xₛ::AbstractVector)

Sample from the state at time `t` given the state at time `s`.

Will only be called with `t >= s`.
Must return `xₛ` if `t == s`.
"""
function Base.rand(
    rng::Random.AbstractRNG,
    gmp::AbstractGaussMarkovProcess,
    t::Real,
    s::Real,
    xₛ::AbstractVector,
)
    Aₜₛ, bₜₛ, lsqrt_Qₜₛ = A_b_lsqrt_Q(gmp, t, s)

    return Aₜₛ * xₛ + bₜₛ + lsqrt_Qₜₛ * randn(rng, eltype(lsqrt_Qₜₛ), size(lsqrt_Qₜₛ, 2))
end
