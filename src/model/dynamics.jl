abstract type AbstractGaussMarkovChain end

"""
    Base.length(gmc::AbstractGaussMarkovChain)

Number of states in the Gauss-Markov chain.
"""
Base.length(::AbstractGaussMarkovChain)

# Mean
function μ end

@doc raw"""
    μ(gmc::AbstractGaussMarkovChain, k::Integer)

Mean of the state at time step `k`.
"""
μ(::AbstractGaussMarkovChain, ::Integer)

# Covariance matrix
function Σ end

@doc raw"""
    Σ(gmc::AbstractGaussMarkovChain, k::Integer)

Covariance matrix of the state at time step `k`.
"""
Σ(::AbstractGaussMarkovChain, ::Integer)

@doc raw"""
    lsqrt_Σ(gmc::AbstractGaussMarkovChain, k::Integer)

A left square root of the covariance matrix of the state at time step `k`.
"""
function lsqrt_Σ(gmc::AbstractGaussMarkovChain, k::Integer)
    return sqrt(Σ(gmc, k))
end

# Transition matrix
function A end

@doc raw"""
    A(gmc::AbstractGaussMarkovChain, k::Integer)

Transition matrix from time step `k` to time step `k + 1`.

Must return the identity matrix for `k = 0`.
"""
A(::AbstractGaussMarkovChain, ::Integer)

# Transition model
function A_b_lsqrt_Q end

@doc raw"""
    A_b_lsqrt_Q(gmc::AbstractGaussMarkovChain, k::Integer)

Transition matrix, bias vector, and left square-root of the process noise covariance from time step `k` to time step `k + 1`.

Will only be called with `k >= 1`.
"""
A_b_lsqrt_Q(::AbstractGaussMarkovChain, ::Integer)

"""
    rand(rng::Random.AbstractRNG, gmc::AbstractGaussMarkovChain, k::Integer)

Sample from the state at time step `k`.
"""
function Base.rand(rng::Random.AbstractRNG, gmc::AbstractGaussMarkovChain, k::Integer)
    lsqrt_Σₖ = lsqrt_Σ(gmc, k)

    return μ(gmc, k) + lsqrt_Σₖ * randn(rng, eltype(lsqrt_Σₖ), size(lsqrt_Σₖ, 2))
end

"""
    rand(rng::Random.AbstractRNG, gmc::AbstractGaussMarkovChain, k::Integer, xₖ::AbstractVector)

Sample from the state at time step `k` given the value of the state at time step `k - 1`.

Must return `xₖ` for `k = 0`.
"""
function Base.rand(
    rng::Random.AbstractRNG,
    gmc::AbstractGaussMarkovChain,
    k::Integer,
    xₖ::AbstractVector,
)
    if k == 0
        return xₖ
    end

    Aₖ, bₖ, lsqrt_Qₖ = A_b_lsqrt_Q(gmc, k)

    return Aₖ * xₖ + bₖ + lsqrt_Qₖ * randn(rng, eltype(lsqrt_Qₖ), size(lsqrt_Qₖ, 2))
end
