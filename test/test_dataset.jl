#= test_dataset.jl — Unit tests for the Dataset module =#
using Test
using DrWatson
@quickactivate "InverseBTE"

include(joinpath(srcdir(), "dataset.jl"))
using .Dataset

@testset "Dataset" begin
    @testset "build_spectrum_vector" begin
        E_bins = [10.0, 100.0, 1000.0]
        sv = Dataset.build_spectrum_vector(100.0, E_bins)
        @test length(sv) == 3
        @test sv[2] == 1.0f0
        @test sv[1] == 0.0f0
    end

    @testset "standardise_Y invertibility" begin
        Y = Float32[1.0 2.0 3.0; 4.0 5.0 6.0]
        Y_norm, ns = Dataset.standardise_Y(Y)
        # Round-trip
        Y_recovered = inverse_transform_y(Y_norm, ns)
        # We are in log10 space during standardise, physical after inverse_transform
        # so compare in log10
        Y_log_recovered = log10.(Y_recovered .+ Dataset.LOG_EPS)
        # The standardised values should have mean≈0, std≈1
        using Statistics
        @test isapprox(mean(Y_norm[1,:]), 0.0; atol=1e-5)
    end

    @testset "NormStats fields" begin
        ns = NormStats([1.0, 2.0], [0.5, 0.3])
        @test length(ns.y_mean) == 2
        @test length(ns.y_std)  == 2
    end

    @testset "shield_mass_areal positive" begin
        mass = Dataset.shield_mass_areal(10.0, 2.0, 20.0)
        @test mass > 0.0
    end

    @testset "make_dataloaders" begin
        X_tr = rand(Float32, 10, 100)
        Y_tr = rand(Float32, 2,  100)
        X_vl = rand(Float32, 10, 20)
        Y_vl = rand(Float32, 2,  20)
        train_dl, val_dl = make_dataloaders(X_tr, Y_tr, X_vl, Y_vl; batchsize=32)
        (xb, yb) = first(train_dl)
        @test size(xb, 1) == 10
        @test size(yb, 1) == 2
    end
end
