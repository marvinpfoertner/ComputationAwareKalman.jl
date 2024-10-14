using Test
using AbstractGPs
using KernelFunctions
using Random
using ComputationAwareKalman

@testset "Matérn process regression" begin
    # Define Gaussian process with a Matérn 5/2 kernel
    σ² = 2.0
    l = 0.5

    f = GP(σ² * with_lengthscale(Matern52Kernel(), l))

    # Sample data
    λ² = 0.4

    rng = Random.seed!(42)
    ts = sort(rand(rng, Float64, 20))

    f_ts = f(ts, λ²)
    ys = rand(rng, f_ts)

    # Fit batch GP
    f_posterior = posterior(f_ts, ys)

    # Construct equivalent LGSSM
    gmp = ComputationAwareKalman.MaternProcess(2, l, σ²)
    obs = ComputationAwareKalman.UniformMeasurementModel([1;; 0;; 0], [λ²;;])

    dgmp = ComputationAwareKalman.discretize(gmp, ts)

    # Run CAKF and CAKS
    fcache = ComputationAwareKalman.filter(dgmp, obs, [[y] for y in ys])
    scache = ComputationAwareKalman.smooth(dgmp, fcache)

    # Compare
    for k = 1:length(ts)
        mˢ = ComputationAwareKalman.mˢ(scache, k)
        Pˢ = ComputationAwareKalman.Pˢ(dgmp, scache, k)

        mˢ₁ref = mean(f_posterior, [ts[k]])[1]
        Pˢ₁₁ref = var(f_posterior, [ts[k]])[1]

        @test mˢ[1] ≈ mˢ₁ref
        @test Pˢ[1, 1] ≈ Pˢ₁₁ref
    end
end
