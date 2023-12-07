abstract type AbstractDiscreteLGSSM end
abstract type AbstractDiscretizedContinuousLGSSM <: AbstractDiscreteLGSSM end

function indices(lgssm::Tlgssm) where {Tlgssm<:AbstractDiscreteLGSSM} end

function prior_mean(
    lgssm::Tlgssm,
    index::Tindex,
) where {Tlgssm<:AbstractDiscreteLGSSM,Tindex<:Integer} end

function prior_cov(
    lgssm::Tlgssm,
    index::Tindex,
) where {Tlgssm<:AbstractDiscreteLGSSM,Tindex<:Integer} end

function transition(
    lgssm::Tlgssm,
    index::Tindex,
) where {Tlgssm<:AbstractDiscreteLGSSM,Tindex<:Integer} end

function ts(lgssm::Tlgssm) where {Tlgssm<:AbstractDiscretizedContinuousLGSSM} end

function prior_mean(
    lgssm::Tlgssm,
    t::Tt,
) where {Tlgssm<:AbstractDiscretizedContinuousLGSSM,Tt<:AbstractFloat} end

function prior_cov(
    lgssm::Tlgssm,
    t::Tt,
) where {Tlgssm<:AbstractDiscretizedContinuousLGSSM,Tt<:AbstractFloat} end

function transition(
    lgssm::Tlgssm,
    t::Tt,
    t₀::Tt,
) where {Tlgssm<:AbstractDiscreteLGSSM,Tt<:AbstractFloat} end

function indices(lgssm::Tlgssm) where {Tlgssm<:AbstractDiscretizedContinuousLGSSM}
    return firstindex(ts(lgssm)):lastindex(ts(lgssm))
end

function prior_cov(
    lgssm::Tlgssm,
    index::Tindex,
) where {Tlgssm<:AbstractDiscretizedContinuousLGSSM,Tindex<:Integer}
    return prior_cov(lgssm, ts(lgssm)[index])
end

function prior_mean(
    lgssm::Tlgssm,
    index::Tindex,
) where {Tlgssm<:AbstractDiscretizedContinuousLGSSM,Tindex<:Integer}
    return prior_mean(lgssm, ts(lgssm)[index])
end

function transition(
    lgssm::Tlgssm,
    index::Tindex,
) where {Tlgssm<:AbstractDiscretizedContinuousLGSSM,Tindex<:Integer}
    return transition(lgssm, ts(lgssm)[index+1], ts(lgssm)[index])
end
