using Test
using ComputationAwareKalman
using Aqua, JET

@testset "Aqua.jl" begin
    Aqua.test_all(
        ComputationAwareKalman;
        ambiguities = (broken = true,),
        deps_compat = (broken = true, check_weakdeps = (broken = true,)),
    )
end

@testset "JET.jl" begin
    JET.test_package(ComputationAwareKalman, broken = true)
end
