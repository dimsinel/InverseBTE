#= test_inverse.jl — Unit tests for the Inverse module =#
using Test
using DrWatson
@quickactivate "InverseBTE"

using Lux, ComponentArrays

include(joinpath(srcdir(), "dataset.jl"))
include(joinpath(srcdir(), "surrogate.jl"))
include(joinpath(srcdir(), "training.jl"))
include(joinpath(srcdir(), "inverse.jl"))
using .Dataset
using .Surrogate
using .Training
using .Inverse

@testset "Inverse" begin
    # Minimal surrogate for speed
    cfg_s = Surrogate.SurrogateConfig(n_input=5, hidden_dim=16, n_res_blocks=1, n_output=2)
    model, ps, st = Surrogate.init_surrogate(cfg_s)

    # Trivial norm stats (identity transform: y_mean=0, y_std=1)
    ns = Dataset.NormStats([0.0, 0.0], [1.0, 1.0])

    @testset "areal_mass positive" begin
        m = Inverse.areal_mass(10.0, 2.0, 20.0)
        @test m > 0.0
    end

    @testset "build_input length" begin
        S_ext = rand(Float32, 2)   # n_input - 3 = 2
        x = Inverse.build_input(S_ext, 5.0, 1.0, 10.0)
        @test length(x) == 5
    end

    @testset "predict_physical returns 2 scalars" begin
        S_ext = rand(Float32, 2)
        x = Inverse.build_input(S_ext, 5.0, 1.0, 10.0)
        tid, n_flux = Inverse.predict_physical(model, ps, st, x, ns.y_mean, ns.y_std)
        @test tid > 0.0
        @test n_flux > 0.0
    end

    @testset "solve_inverse returns InverseResult" begin
        S_ext = Float32.(ones(2) ./ 2)
        cfg_inv = InverseConfig(
            D_max_Gy       = 1.0,
            penalty_lambda = 1e3,
            t_bounds_Al    = (1.0, 20.0),
            t_bounds_Ta    = (0.5, 5.0),
            t_bounds_Poly  = (5.0, 50.0),
            adam_iters     = 20,    # fast for test
            lbfgs_iters    = 10,
            lr_adam        = 5e-3,
        )
        result = solve_inverse(model, ps, st, S_ext, ns; cfg=cfg_inv)
        @test result isa InverseResult
        @test result.mass_areal > 0.0
        @test result.t_Al_mm >= 1.0
        @test result.t_Al_mm <= 20.0
    end
end
