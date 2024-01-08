function discretize(gmp::AbstractGaussMarkovProcess, ts)
    return DiscretizedGaussMarkovProcess(gmp, ts)
end

struct DiscretizedGaussMarkovProcess{Tgmp<:AbstractGaussMarkovProcess,Tts} <:
       AbstractGaussMarkovChain
    gmp::Tgmp
    ts::Tts # TODO: Document that `ts` must be indexed by 1:length(dgmp)
end

function ts(dgmp::Tdgmp) where {Tdgmp<:DiscretizedGaussMarkovProcess}
    return dgmp.ts
end

function Base.length(dgmp::Tdgmp) where {Tdgmp<:DiscretizedGaussMarkovProcess}
    return length(dgmp.ts)
end

function μ(dgmp::Tdgmp, k::Tk) where {Tdgmp<:DiscretizedGaussMarkovProcess,Tk<:Integer}
    return μ(dgmp.gmp, dgmp.ts[max(k, 1)])
end

function Σ(dgmp::Tdgmp, k::Tk) where {Tdgmp<:DiscretizedGaussMarkovProcess,Tk<:Integer}
    return Σ(dgmp.gmp, dgmp.ts[max(k, 1)])
end

function lsqrt_Σ(
    dgmp::Tdgmp,
    k::Tk,
) where {Tdgmp<:DiscretizedGaussMarkovProcess,Tk<:Integer}
    return lsqrt_Σ(dgmp.gmp, dgmp.ts[max(k, 1)])
end

function A(dgmp::Tdgmp, k::Tk) where {Tdgmp<:DiscretizedGaussMarkovProcess,Tk<:Integer}
    return A(dgmp.gmp, dgmp.ts[k+1], k > 0 ? dgmp.ts[k] : dgmp.ts[k+1])
end

function A_b_lsqrt_Q(
    dgmp::Tdgmp,
    k::Tk,
) where {Tdgmp<:DiscretizedGaussMarkovProcess,Tk<:Integer}
    return A_b_lsqrt_Q(dgmp.gmp, dgmp.ts[k+1], k > 0 ? dgmp.ts[k] : dgmp.ts[k+1])
end
