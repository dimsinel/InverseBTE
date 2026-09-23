using Test
using DrWatson
@quickactivate "InverseBTE"

@testset "InverseBTE" begin
    include("test_geometry.jl")
    include("test_dataset.jl")
    include("test_surrogate.jl")
    include("test_inverse.jl")
end
