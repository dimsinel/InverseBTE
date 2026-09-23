#= test_surrogate.jl — Unit tests for the Surrogate and Training modules =#
using Test
using DrWatson
@quickactivate "InverseBTE"

using Lux, Random, Zygote, ComponentArrays

include(joinpath(srcdir(), "surrogate.jl"))
include(joinpath(srcdir(), "training.jl"))
using .Surrogate
using .Training

@testset "Surrogate" begin
    @testset "build_surrogate output shape" begin
        cfg   = SurrogateConfig(n_input=33, hidden_dim=64, n_res_blocks=2, n_output=2)
        model, ps, st = init_surrogate(cfg)
        x = rand(Float32, 33, 8)   # batch of 8
        y, _ = Lux.apply(model, x, ps, st)
        @test size(y) == (2, 8)
    end

    @testset "Zygote gradient non-NaN" begin
        cfg   = SurrogateConfig(n_input=33, hidden_dim=32, n_res_blocks=1, n_output=2)
        model, ps, st = init_surrogate(cfg)
        x = rand(Float32, 33, 4)
        y_true = rand(Float32, 2, 4)
        loss_fn = p -> begin
            y_hat, _ = Lux.apply(model, x, p, st)
            Training.huber_loss(y_hat, y_true)
        end
        g = Zygote.gradient(loss_fn, ps)[1]
        @test !any(isnan, g)
    end

    @testset "Huber loss properties" begin
        y_hat = Float32[0.0, 2.0]
        y_true = Float32[0.0, 0.0]
        # For r=0: 0; for |r|=2 > δ=1: loss = 1.0*(2-0.5) = 1.5; mean([0, 1.5]) = 0.75
        l = Training.huber_loss(reshape(y_hat,2,1), reshape(y_true,2,1); δ=1.0f0)
        @test isapprox(l, 0.75f0; atol=0.01)
    end
end
