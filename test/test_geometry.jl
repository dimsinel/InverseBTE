#= test_geometry.jl — Unit tests for the Geometry module =#
using Test
using DrWatson
@quickactivate "InverseBTE"

include(joinpath(srcdir(), "geometry.jl"))
using .Geometry

@testset "Geometry" begin
    @testset "ShieldParams" begin
        p = ShieldParams(10.0, 2.0, 20.0)
        @test p.t_Al   == 10.0
        @test p.t_Ta   ==  2.0
        @test p.t_Poly == 20.0
    end

    @testset "clamp_params" begin
        p_over = ShieldParams(999.0, 999.0, 999.0)
        p_c    = Geometry.clamp_params(p_over)
        b      = shield_bounds()
        @test p_c.t_Al   == b.Al[2]
        @test p_c.t_Ta   == b.Ta[2]
        @test p_c.t_Poly == b.Poly[2]
    end

    @testset "shield_bounds format" begin
        b = shield_bounds()
        @test haskey(b, :Al)
        @test haskey(b, :Ta)
        @test haskey(b, :Poly)
        @test b.Al[1] < b.Al[2]
    end

    @testset "SatelliteDetector defaults" begin
        det = SatelliteDetector()
        @test det.shield.t_Al == 10.0
        @test det.sensitive_half == 5.0
        @test det.world_margin == 5.0
    end
end
