using Test
using ComputationAwareKalman
using Aqua

@testset "Aqua.jl" begin
    Aqua.test_all(
        ComputationAwareKalman;
        ambiguities = (broken = true,),
        deps_compat = (broken = true, check_weakdeps = (broken = true,)),
    )
end
