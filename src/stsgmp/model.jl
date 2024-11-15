struct SpaceTimeSeparableGaussMarkovProcess{Ttgmp<:AbstractGaussMarkovProcess,Tμₓ,TΣₓ}
    tgmp::Ttgmp

    spatial_mean_fn::Tμₓ
    spatial_cov_fn::TΣₓ
end

statedim(stsgmp::SpaceTimeSeparableGaussMarkovProcess) = statedim(stsgmp.tgmp)
