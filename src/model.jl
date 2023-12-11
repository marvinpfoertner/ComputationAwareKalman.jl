abstract type AbstractGaussMarkovChain end
abstract type AbstractDiscretizedGaussMarkovProcess <: AbstractGaussMarkovChain end

# AbstractGaussMarkovChain interface
function indices(gmc::Tgmc) where {Tgmc<:AbstractGaussMarkovChain} end

function prior_mean(
    gmc::Tgmc,
    index::Tindex,
) where {Tgmc<:AbstractGaussMarkovChain,Tindex<:Integer} end

function prior_cov(
    gmc::Tgmc,
    index::Tindex,
) where {Tgmc<:AbstractGaussMarkovChain,Tindex<:Integer} end

function transition(
    gmc::Tgmc,
    index::Tindex,
) where {Tgmc<:AbstractGaussMarkovChain,Tindex<:Integer} end

# AbstractDiscretizedGaussMarkovProcess interface
function ts(gmp::Tgmp) where {Tgmp<:AbstractDiscretizedGaussMarkovProcess} end

function prior_mean(
    dgmp::Tdgmp,
    t::Tt,
) where {Tdgmp<:AbstractDiscretizedGaussMarkovProcess,Tt<:AbstractFloat} end

function prior_cov(
    dgmp::Tdgmp,
    t::Tt,
) where {Tdgmp<:AbstractDiscretizedGaussMarkovProcess,Tt<:AbstractFloat} end

function transition(
    dgmp::Tdgmp,
    t::Tt,
    t₀::Tt,
) where {Tdgmp<:AbstractDiscretizedGaussMarkovProcess,Tt<:AbstractFloat} end

# AbstractDiscretizedGaussMarkovProcess implementation of AbstractGaussMarkovChain interface
function indices(dgmp::Tdgmp) where {Tdgmp<:AbstractDiscretizedGaussMarkovProcess}
    return firstindex(ts(dgmp)):lastindex(ts(dgmp))
end

function prior_cov(
    dgmp::Tdgmp,
    index::Tindex,
) where {Tdgmp<:AbstractDiscretizedGaussMarkovProcess,Tindex<:Integer}
    return prior_cov(dgmp, ts(dgmp)[index])
end

function prior_mean(
    dgmp::Tdgmp,
    index::Tindex,
) where {Tdgmp<:AbstractDiscretizedGaussMarkovProcess,Tindex<:Integer}
    return prior_mean(dgmp, ts(dgmp)[index])
end

function transition(
    dgmp::Tdgmp,
    index::Tindex,
) where {Tdgmp<:AbstractDiscretizedGaussMarkovProcess,Tindex<:Integer}
    return transition(dgmp, ts(dgmp)[index+1], ts(dgmp)[index])
end
