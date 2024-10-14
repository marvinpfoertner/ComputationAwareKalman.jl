struct MaternProcess{TF,TΣ∞} <: AbstractGaussMarkovProcess
    F::TF
    Σ∞::TΣ∞
end

function MaternProcess(p::Integer, l::Real = 1, σ²::Real = 1)
    D = p + 1

    λ = √(2 * p + 1) / l

    F = [
        zeros(p) I(p)
        [-binomial(D, i) * λ^(D - i) for i = 0:p]'
    ]

    if p == 0
        Σ∞ = [σ²]
    elseif p == 1
        Σ∞ = σ² * Diagonal([1, λ^2])
    else
        L = [zeros(p); 1.0]
        Q = [σ² * (2λ)^(2p + 1) / binomial(2p, p)]

        Σ∞ = Symmetric(lyap(F, L * Q * L'))
    end

    return MaternProcess{Ttypeof(F),typeof(Σ∞)}(F, Σ∞)
end

function μ(gmp::MaternProcess, ::Real)
    D = size(gmp.F, 2)

    return zeros(eltype(gmp.Σ∞), D)
end

function Σ(gmp::MaternProcess, ::Real)
    return gmp.Σ∞
end

function A(gmp::MaternProcess, t::Real, s::Real)
    return exp(gmp.F * (t - s))
end

function A_b_lsqrt_Q(gmp::MaternProcess, t::Real, s::Real)
    Aₜₛ = A(gmp, t, s)
    bₜₛ = zeros(eltype(Aₜₛ), size(Aₜₛ, 1))
    Qₜₛ = gmp.Σ∞ - Aₜₛ * gmp.Σ∞ * Aₜₛ'

    return Aₜₛ, bₜₛ, sqrt(Qₜₛ)
end
